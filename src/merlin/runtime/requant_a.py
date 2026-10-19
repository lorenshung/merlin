"""requant-A-v1: exact monotone-boundary certificates, independent of target.

Promoted from the provider prototype. LLVM host emitters supply exact ratios,
offsets and proven integer accumulator intervals; this module has no target
dependency. The certificate format is docs/compiler/requant_A_contract.md.
"""

import hashlib
import json
from fractions import Fraction
from functools import lru_cache

SCHEMA = "requant-A-v1"
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1


def exact(x, f, c, zp, qmin, qmax, relu=False):
    value = x * f + c
    return min(qmax, max(qmin, round(max(0, value) if relu else value) + zp))


def transitions(f, c, zp, qmin, qmax, relu, lo, hi):
    """First integer x reaching each successive code, with hi+1 sentinel."""
    f, c = Fraction(f), Fraction(c)
    if f <= 0 or not lo <= hi or not 0 <= qmax - qmin <= 255:
        raise ValueError("invalid requant-A domain, ratio or byte output range")
    out = []
    for code in range(qmin + 1, qmax + 1):
        q = code - zp
        if relu and q <= 0:
            t = lo
        else:
            b = (Fraction(2 * q - 1, 2) - c) / f
            t = -(-b.numerator // b.denominator)
            if b.denominator == 1 and q % 2:
                t += 1
        out.append(min(hi + 1, max(lo, t)))
    return tuple(out)


def segments(ts, qmin, qmax, lo, hi):
    starts, ends = (lo, *ts), (*(t - 1 for t in ts), hi)
    return tuple((q, a, b) for q, a, b in zip(range(qmin, qmax + 1), starts, ends) if a <= b)


def _points(ts, lo, hi, constants=None):
    points = {lo, hi}
    for t in ts:
        points.update(x for x in (t - 1, t) if lo <= x <= hi)
    if constants:
        points.update(x for x in (int(constants["L"]), int(constants["H"])) if lo <= x <= hi)
    return tuple(sorted(points))


def _hash(values):
    return hashlib.sha256(("".join(str(v) + "\n" for v in values)).encode()).hexdigest()


def apply(x, record):
    if record["algorithm"] == "threshold_table":
        import bisect

        return record["quant_min"] + bisect.bisect_right(tuple(map(int, record["thresholds"])), x)
    k = record["constants"]
    v = min(int(k["H"]), max(int(k["L"]), x))
    q = (v * int(k["M"]) + int(k["B"])) >> k["S"]
    if k["out_clamp"]:
        q = min(k["qhi"], max(k["qlo"], q))
    return q


def _solve(f, segs, out_clamp):
    if len(segs) == 1:
        # A constant monotone map is the degenerate M=0, S=0 affine form.
        return dict(L="0", H="0", M="0", B=str(segs[0][0]), S=0, out_clamp=False, qlo=segs[0][0], qhi=segs[0][0])
    L, H = segs[0][2], segs[-1][1]
    if not I64_MIN <= L <= H <= I64_MAX:
        return None
    for S in range(63):
        one = 1 << S
        m0 = (f * one).__floor__()
        for M in sorted({m0 + d for d in (-2, -1, 0, 1, 2, 3) if 0 < m0 + d <= I64_MAX}):
            # Both multiplication and the subsequent addition must fit int64.
            if not I64_MIN <= L * M <= H * M <= I64_MAX:
                continue
            low, high = max(I64_MIN, I64_MIN - L * M), min(I64_MAX, I64_MAX - H * M)
            for q, a, b in segs:
                a, b = max(a, L), min(b, H)
                if a > b:
                    continue
                if not (out_clamp and q == segs[0][0]):
                    low = max(low, q * one - a * M)
                if not (out_clamp and q == segs[-1][0]):
                    high = min(high, (q + 1) * one - 1 - b * M)
                if low > high:
                    break
            if low <= high:
                B = (low + high) // 2
                return dict(
                    L=str(L), H=str(H), M=str(M), B=str(B), S=S, out_clamp=out_clamp, qlo=segs[0][0], qhi=segs[-1][0]
                )
    return None


def validate(record, *, require_proof=True):
    """Reject unproved/tampered constants; regenerate ALL exact boundaries."""

    def decimal(value):
        if not isinstance(value, str) or str(int(value)) != value:
            raise ValueError("noncanonical requant integer")

    for value in record["domain"].values():
        decimal(value)
    for key in ("zero_point", "quant_min", "quant_max"):
        if type(record[key]) is not int:
            raise ValueError("noninteger requant output parameter")
    if type(record["relu"]) is not bool:
        raise ValueError("nonboolean requant ReLU flag")
    if require_proof and not isinstance(record.get("proof"), dict):
        raise ValueError("missing requant boundary proof")
    if record["rounding"] != "half_to_even":
        raise ValueError("unsupported requant rounding")
    if record["schema"] != SCHEMA:
        raise ValueError("unknown requant certificate schema")
    lo, hi = int(record["domain"]["min"]), int(record["domain"]["max"])
    f, c = (Fraction(int(record[key]["numerator"]), int(record[key]["denominator"])) for key in ("ratio", "offset"))
    for key, value in [("ratio", f), ("offset", c)]:
        if record[key] != dict(numerator=str(value.numerator), denominator=str(value.denominator)):
            raise ValueError("noncanonical requant rational")
    zp, qmin, qmax, relu = (record[k] for k in ("zero_point", "quant_min", "quant_max", "relu"))
    ts = transitions(f, c, zp, qmin, qmax, relu, lo, hi)
    k = record.get("constants")
    if record["algorithm"] == "clamp_mul_add_shift":
        for key in ("L", "H", "M", "B"):
            decimal(k[key])
        if (
            type(k["S"]) is not int
            or type(k["out_clamp"]) is not bool
            or type(k["qlo"]) is not int
            or type(k["qhi"]) is not int
            or not qmin <= k["qlo"] <= k["qhi"] <= qmax
        ):
            raise ValueError("invalid requant clamp or shift")
        L, H, M, B, S = (int(k[key]) for key in ("L", "H", "M", "B", "S"))
        if not (
            I64_MIN <= L <= H <= I64_MAX
            and 0 <= M <= I64_MAX
            and I64_MIN <= B <= I64_MAX
            and 0 <= S <= 62
            and I64_MIN <= L * M <= H * M <= I64_MAX
            and I64_MIN <= L * M + B <= H * M + B <= I64_MAX
        ):
            raise ValueError("requant-A int64 overflow or invalid constants")
    elif record["algorithm"] == "threshold_table":
        if k is not None:
            raise ValueError("threshold certificate also contains affine constants")
        for value in record["thresholds"]:
            decimal(value)
        if tuple(map(int, record["thresholds"])) != ts:
            raise ValueError("invalid exact threshold table")
    else:
        raise ValueError("unknown requant algorithm")
    points = _points(ts, lo, hi, k)
    if any(apply(x, record) != exact(x, f, c, zp, qmin, qmax, relu) for x in points):
        raise ValueError("requant-A boundary proof failed")
    proof = dict(
        method="all_monotone_step_boundaries",
        step_count=len(ts),
        checked_points=len(points),
        checked_points_sha256=_hash(points),
        thresholds_sha256=_hash(ts),
        monotone=True,
        integer_arithmetic="signed_i64" if k else "exact_integer_comparisons",
    )
    if k:
        proof.update(product_min=str(L * M), product_max=str(H * M), sum_min=str(L * M + B), sum_max=str(H * M + B))
    if record.get("proof") is not None and record["proof"] != proof:
        raise ValueError("requant-A certificate proof mismatch")
    return proof


@lru_cache(maxsize=16384)
def _plan(f, c, zp, qmin, qmax, relu, lo, hi):
    ts = transitions(f, c, zp, qmin, qmax, relu, lo, hi)
    segs = segments(ts, qmin, qmax, lo, hi)
    k = _solve(f, segs, False) or _solve(f, segs, True)
    rec = dict(
        schema=SCHEMA,
        algorithm="clamp_mul_add_shift" if k else "threshold_table",
        domain=dict(min=str(lo), max=str(hi)),
        ratio=dict(numerator=str(f.numerator), denominator=str(f.denominator)),
        offset=dict(numerator=str(c.numerator), denominator=str(c.denominator)),
        zero_point=zp,
        quant_min=qmin,
        quant_max=qmax,
        relu=relu,
        rounding="half_to_even",
    )
    if k:
        rec["constants"] = k
    else:
        rec.update(thresholds=list(map(str, ts)), refusal="no int64 candidate passed the complete boundary proof")
    rec["proof"] = validate(rec, require_proof=False)
    return json.dumps(rec, sort_keys=True)


def plan(f, c=0, zp=0, qmin=-128, qmax=127, relu=False, lo=-(1 << 31), hi=(1 << 31) - 1):
    if any(int(v) != v for v in (zp, qmin, qmax, lo, hi)):
        raise ValueError("noninteger requant parameter")
    # Return a fresh record: consumers may annotate channel/tile bindings.
    return json.loads(_plan(Fraction(f), Fraction(c), int(zp), int(qmin), int(qmax), bool(relu), int(lo), int(hi)))


def validate_manifest(document):
    """Recheck shared provider certificates before accepting a linked catalog.

    Binding fields stay provider-owned. Deduplicate only the semantic record,
    so an identical class used in many tiles needs one boundary proof check.
    """
    keys = (
        "schema",
        "algorithm",
        "domain",
        "ratio",
        "offset",
        "zero_point",
        "quant_min",
        "quant_max",
        "relu",
        "rounding",
        "constants",
        "thresholds",
        "proof",
    )
    seen = set()
    for kernel in document.get("kernels", ()):
        for record in kernel.get("meta", {}).get("host_requantization_proofs", ()):
            if record.get("schema") != SCHEMA:
                continue
            semantic = {k: record[k] for k in keys if k in record}
            key = json.dumps(semantic, sort_keys=True)
            if key not in seen:
                validate(semantic)
                seen.add(key)
    return len(seen)
