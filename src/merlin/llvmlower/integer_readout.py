"""Source-exact CPU readout from signed-i32 accumulators to signed-i8.

Positive ordered binary32 scales and nearest-even saturation form monotone step
functions. Full transition proofs permit integer evaluation with a guarded
fixed-point estimate; the generated C uses no accelerator instructions or ABI.
"""

import bisect
import math

from .requantization import f32, quantized


def derive(scales, lo=-(1 << 31), hi=(1 << 31) - 1, relu=False):
    scales = tuple(f32(s) for s in scales)
    if not scales or any(not math.isfinite(s) or s <= 0 for s in scales):
        raise ValueError("positive finite f32 source scales required")
    if type(lo) is not int or type(hi) is not int or not -(1 << 31) <= lo <= hi < (1 << 31):
        raise ValueError("invalid i32 accumulator domain")
    low = 0 if relu else -128
    thresholds = []
    for q in range(low + 1, 128):
        left, right = lo, hi + 1
        while left < right:
            mid = (left + right) // 2
            if quantized(mid, scales, relu) >= q:
                right = mid
            else:
                left = mid + 1
        if left <= hi and quantized(left, scales, relu) < q:
            raise AssertionError("invalid upper boundary")
        if left > lo and quantized(left - 1, scales, relu) >= q:
            raise AssertionError("invalid lower boundary")
        thresholds.append(left)
    assert thresholds == sorted(thresholds)
    return dict(
        schema="exact_integer_readout_v1",
        source_scales=list(scales),
        accumulator_min=lo,
        accumulator_max=hi,
        output_min=low,
        output_max=127,
        thresholds=thresholds,
        rounding="source f32 nearest even",
        proof="All monotone output transition thresholds and immediate predecessors verified; positive ordered f32 multiplications preserve monotonicity",
        exact=True,
        max_output_lsb_error=0,
    )


def evaluate(acc, proof):
    if not proof["accumulator_min"] <= acc <= proof["accumulator_max"]:
        raise ValueError("outside proven domain")
    return proof["output_min"] + bisect.bisect_right(proof["thresholds"], acc)


def fixedpoint_candidate(proof):
    """Prove a cheap estimate lies within one output step over the full domain."""
    lo, hi = proof["accumulator_min"], proof["accumulator_max"]
    low = proof["output_min"]
    product = math.prod(proof["source_scales"])
    if not math.isfinite(product):
        return None
    for shift in range(30, 7, -1):
        multiplier = round(product * (1 << shift))
        half = 1 << (shift - 1)
        if multiplier <= 0 or max(abs(lo), abs(hi)) * multiplier + half >= (1 << 63):
            continue

        def estimate(a):
            return max(low, min(127, (a * multiplier + half) // (1 << shift)))

        boundaries = {lo, hi, *[x for x in proof["thresholds"] if x <= hi]}
        for q in range(low + 1, 128):
            numerator = q * (1 << shift) - half
            threshold = -((-numerator) // multiplier)
            if lo <= threshold <= hi:
                boundaries.add(threshold)
        error = max(abs(evaluate(a, proof) - estimate(a)) for a in boundaries)
        if error <= 1:
            return dict(
                multiplier=multiplier,
                shift=shift,
                max_estimate_output_error=error,
                proof="Complete union of source and fixed-point estimate transition points and domain endpoints",
                correction="At most one neighboring source threshold correction",
            )
    return None


def emit_readout(proof, symbol, *, fixedpoint=False, saturation_first=False, packet=1, producer_range=None):
    """Emit exact CPU readout, optionally testing saturation before estimation.

    Packets retain each input load/output store pair in source order. They grant
    no pointer nonoverlap and never hoist all lane loads ahead of the stores.
    The default options preserve the existing generated source bytes.
    A producer_range explicitly changes the input contract: the owner must bind
    its typed integer sum-of-products certificate to this exact storage/call.
    Producer storage must remain unchanged until its last read (in particular,
    overlapping output writes must not invalidate the certified input values).
    Arbitrary public i32 inputs must retain the checked default.
    """
    if type(saturation_first) is not bool:
        raise ValueError("saturation_first requires an explicit boolean")
    if type(packet) is not int or packet not in (1, 2, 4, 8):
        raise ValueError("readout packet must be 1, 2, 4 or 8")
    # Re-derive rather than trusting mutable receipt fields from an external file.
    expected = derive(
        proof["source_scales"], proof["accumulator_min"], proof["accumulator_max"], proof["output_min"] == 0
    )
    if any(proof.get(k) != expected[k] for k in expected):
        raise ValueError("readout proof fields changed")
    domain_check = (
        f"if((int64_t)x<{proof['accumulator_min']}LL || (int64_t)x>{proof['accumulator_max']}LL)__builtin_trap();"
    )
    if producer_range is not None:
        from .integer_producer_range import IntegerSumProductsRange

        if type(producer_range) is not IntegerSumProductsRange:
            raise ValueError("typed source-bound integer producer range required")
        producer_range.require_contained(proof["accumulator_min"], proof["accumulator_max"])
        domain_check = "/* Input domain guaranteed by the bound producer contract. */"
    thresholds = ",".join(str(x) + "LL" for x in proof["thresholds"])
    low = proof["output_min"]
    count = len(proof["thresholds"])
    candidate = fixedpoint_candidate(proof) if fixedpoint else None
    if fixedpoint and candidate is None:
        raise ValueError("no proven single-correction fixed-point estimate")
    if candidate:
        body = f"""int64_t q=((int64_t)x*{candidate["multiplier"]}LL+{1 << (candidate["shift"] - 1)}LL)>>{candidate["shift"]};
 if(q<{low})q={low};if(q>127)q=127;
 if(q<127 && (int64_t)x>={symbol}_thresholds[q-({low})])q++;
 else if(q>{low} && (int64_t)x<{symbol}_thresholds[q-({low})-1])q--;
 return (int8_t)q;"""
    else:
        body = f"""unsigned lo=0,hi={count};
 while(lo<hi){{unsigned mid=(lo+hi)/2;if((int64_t)x>={symbol}_thresholds[mid])lo=mid+1;else hi=mid;}}
 return (int8_t)({low}+(int)lo);"""
    if saturation_first:
        body = f"""if((int64_t)x<{proof["thresholds"][0]}LL)return (int8_t){low};
 if((int64_t)x>={proof["thresholds"][-1]}LL)return (int8_t)127;
 {body}"""
    loop = f"for(size_t i=0;i<count;i++)out[i]={symbol}_scalar(acc[i]);"
    if packet != 1:
        lanes = "\n ".join(f"out[i+{lane}]={symbol}_scalar(acc[i+{lane}]);" for lane in range(packet))
        loop = f"""size_t i=0;
 for(;count-i>={packet};i+={packet}){{
 {lanes}
 }}
 for(;i<count;i++)out[i]={symbol}_scalar(acc[i]);"""
    return f"""#include <stdint.h>
#include <stddef.h>
_Static_assert((-1LL>>1)==-1LL,"arithmetic signed right shift required");
static const int64_t {symbol}_thresholds[{count}]={{{thresholds}}};
static inline int8_t {symbol}_scalar(int32_t x){{
 {domain_check}
 {body}
}}
void {symbol}(const int32_t*acc,int8_t*out,size_t count){{
 {loop}
}}
"""
