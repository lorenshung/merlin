"""Complete signed-i8 pair certificates for ordered binary32 affine arithmetic.

An integer predictor is a caller supplied arithmetic contract. A target must
independently validate its implementation of that contract before using this
certificate. The correction never changes source rounding or reassociates it.
"""

import hashlib
import math

import numpy as np

from .requantization import f32


def _scales(lhs_scale, rhs_scale, output_scale, relu):
    if type(relu) is not bool:
        raise ValueError("explicit source ReLU boolean required")
    try:
        scales = tuple(f32(v) for v in (lhs_scale, rhs_scale, output_scale))
        reciprocal = f32(1.0 / scales[2])
    except (OverflowError, TypeError, ZeroDivisionError) as error:
        raise ValueError("finite representable binary32 scales required") from error
    if any(not math.isfinite(v) or v <= 0 for v in (*scales, reciprocal)):
        raise ValueError("positive finite binary32 scales and reciprocal required")
    return dict(lhs_scale=scales[0], rhs_scale=scales[1], output_scale=scales[2], relu=relu), reciprocal


def source_table(lhs_scale, rhs_scale, output_scale, *, relu=False):
    """Separate RN-even f32 products, addition, optional ReLU and reciprocal multiply."""
    source, reciprocal = _scales(lhs_scale, rhs_scale, output_scale, relu)
    a = np.arange(-128, 128, dtype=np.float32)[:, None]
    b = np.arange(-128, 128, dtype=np.float32)[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        lhs = np.multiply(a, np.float32(source["lhs_scale"]), dtype=np.float32)
        rhs = np.multiply(b, np.float32(source["rhs_scale"]), dtype=np.float32)
        total = np.add(lhs, rhs, dtype=np.float32)
        if relu:
            total = np.maximum(total, np.float32(0))
        scaled = np.multiply(total, np.float32(reciprocal), dtype=np.float32)
    if any(not np.isfinite(value).all() for value in (lhs, rhs, total, scaled)):
        raise ValueError("source pair arithmetic may overflow binary32")
    return np.clip(np.rint(scaled), 0 if relu else -128, 127).astype(np.int8)


def predictor_table(p, q, scale, *, relu=False):
    """Exact i32 affine sum, i32-to-f32, f32 product, RNE and signed-i8 clamp."""
    if type(p) is not int or type(q) is not int or 128 * (abs(p) + abs(q)) > (1 << 31) - 1:
        raise ValueError("integer predictor coefficients must prove signed-i32 accumulation")
    try:
        scale = f32(scale)
    except (OverflowError, TypeError) as error:
        raise ValueError("finite binary32 predictor scale required") from error
    if not math.isfinite(scale) or scale <= 0 or type(relu) is not bool:
        raise ValueError("positive finite predictor scale and ReLU boolean required")
    a = np.arange(-128, 128, dtype=np.int64)[:, None]
    b = np.arange(-128, 128, dtype=np.int64)[None, :]
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.multiply((a * p + b * q).astype(np.float32), np.float32(scale), dtype=np.float32)
    if not np.isfinite(scaled).all():
        raise ValueError("predictor pair arithmetic may overflow binary32")
    return np.clip(np.rint(scaled), 0 if relu else -128, 127).astype(np.int8)


def _centered_guard(values):
    # Powers of two support safe byte-parallel comparisons. No sample bound is
    # used: every ambiguous source pair must be outside this interval.
    if 0 in values:
        return None
    negative = max((v for v in values if v < 0), default=-129)
    positive = min((v for v in values if v > 0), default=128)
    lower = max(v for v in (0, 1, 2, 4, 8, 16, 32, 64, 128) if -v > negative)
    upper = max(v for v in (1, 2, 4, 8, 16, 32, 64, 128) if v <= positive)
    return dict(negative_radius=lower, positive_radius=upper, safe_min=-lower, safe_max=upper - 1)


def derive(lhs_scale, rhs_scale, output_scale, *, p, q, scale, relu=False):
    source, reciprocal = _scales(lhs_scale, rhs_scale, output_scale, relu)
    expected = source_table(**source)
    predicted = predictor_table(p, q, scale, relu=relu)
    mismatch = expected != predicted
    locations = np.argwhere(mismatch) - 128
    guards = []
    for axis, index in (("lhs", 0), ("rhs", 1)):
        guard = _centered_guard(set(map(int, locations[:, index])))
        if guard is not None:
            guards.append(dict(axis=axis, **guard))
    guard = max(guards, key=lambda row: row["negative_radius"] + row["positive_radius"], default=None)
    if guard is not None:
        index = 0 if guard["axis"] == "lhs" else 1
        if any(guard["safe_min"] <= int(v) <= guard["safe_max"] for v in locations[:, index]):
            raise ValueError("prefix guard hides a mismatched source pair")
    raw_bitmap = np.packbits(np.roll(mismatch, 128, axis=(0, 1)).ravel(), bitorder="little").tobytes()
    bit_guard = None
    if mismatch.any():
        raw = predicted.view(np.uint8)
        informative = int(np.bitwise_or.reduce(raw.ravel()) ^ np.bitwise_and.reduce(raw.ravel()))
        ones = int(np.bitwise_and.reduce(raw[mismatch]))
        zeros = int(~np.bitwise_or.reduce(raw[mismatch])) & 255
        mask = (ones | zeros) & informative
        if mask:
            bit_guard = dict(mask=mask, value=ones & mask)
    return dict(
        schema="quantized_affine_pair_certificate_v1",
        source=source,
        reciprocal=reciprocal,
        predictor=dict(p=p, q=q, scale=f32(scale)),
        pairs=65536,
        mismatched_pairs=int(mismatch.sum()),
        max_predictor_output_error=int(np.abs(expected.astype(np.int16) - predicted.astype(np.int16)).max()),
        source_table_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
        predictor_table_sha256=hashlib.sha256(predicted.tobytes()).hexdigest(),
        correction_bitmap_hex=raw_bitmap.hex(),
        correction_bitmap_sha256=hashlib.sha256(raw_bitmap).hexdigest(),
        packed_prefix_guard=guard,
        predictor_bit_guard=bit_guard,
        correction_exact_for_all_pairs=True,
        source_arithmetic=(
            "binary32 RN-even with gradual underflow; separate multiply, multiply, add, optional ReLU, "
            "reciprocal multiply, RNE, clamp"
        ),
        predictor_arithmetic=(
            "exact signed-i32 affine sum, binary32 conversion/product RN-even, RNE, clamp; "
            "hardware implementation must be independently validated"
        ),
        runtime_ambiguity_rate="UNKNOWN: full-domain count is not a runtime input distribution",
    )


def _prefix_expression(guard):
    """A byte-parallel C expression; high bits mark values outside the safe interval."""
    terms = []
    left, right = guard["negative_radius"], guard["positive_radius"]
    if left == 0:
        terms.append("raw")
    left_bits = int(math.log2(left)) if left else 7
    right_bits = int(math.log2(right))
    for bit in range(max(left_bits, right_bits), 7):
        terms.append(f"(raw^(raw<<{7 - bit}))")
    for bit in range(min(left_bits, right_bits), max(left_bits, right_bits)):
        terms.append(f"(raw&~(raw<<{7 - bit}))" if left_bits < right_bits else f"(~raw&(raw<<{7 - bit}))")
    return "(" + ("|".join(terms) if terms else "UINT64_C(0)") + ")&UINT64_C(0x8080808080808080)"


_GATHER_EXPRESSION = "((outside>>7)*UINT64_C(0x0102040810204080))>>56"


def _valid_c_identifier(symbol):
    return (
        type(symbol) is str
        and symbol.isascii()
        and symbol.isidentifier()
        and symbol
        not in {
            "for",
            "if",
            "while",
            "return",
            "switch",
            "case",
            "void",
            "int",
            "float",
            "static",
            "const",
            "struct",
            "typedef",
            "do",
            "else",
            "break",
            "continue",
            "goto",
            "sizeof",
            "char",
            "long",
            "short",
            "double",
            "signed",
            "unsigned",
            "union",
            "enum",
            "volatile",
            "extern",
            "auto",
            "register",
            "default",
            "restrict",
            "inline",
            "_Bool",
            "_Complex",
            "_Imaginary",
            "_Alignas",
            "_Alignof",
            "_Atomic",
            "_Generic",
            "_Noreturn",
            "_Static_assert",
            "_Thread_local",
        }
    )


def derive_output_value_guard(proof):
    """Prove which predictor byte values may need original source replay.

    The complete pair relation derives a conservative raw-byte bit predicate.
    Rejected output values equal the original source for every signed-i8 pair;
    accepted values still require the original immutable inputs and mismatch
    bitmap. No observed input distribution participates in the predicate.
    """
    if proof != derive(**proof["source"], **proof["predictor"]):
        raise ValueError("unchanged complete pair certificate required")
    expected = source_table(**proof["source"])
    predicted = predictor_table(**proof["predictor"], relu=proof["source"]["relu"])
    values = sorted(map(int, np.unique(predicted[expected != predicted].view(np.uint8))))
    ones, zeros = 255, 255
    for value in values:
        ones &= value
        zeros &= value ^ 255
    guard = dict(mask=ones | zeros, value=ones) if values else None
    if not values:
        accepted = []
    elif guard is None:
        accepted = list(range(256))
    else:
        accepted = [byte for byte in range(256) if byte & guard["mask"] == guard["value"]]
    if not set(values) <= set(accepted):
        raise ValueError("output-value guard hides a mismatched source pair")
    return dict(
        schema="quantized_affine_output_value_guard_v1",
        source_table_sha256=proof["source_table_sha256"],
        predictor_table_sha256=proof["predictor_table_sha256"],
        correction_bitmap_sha256=proof["correction_bitmap_sha256"],
        pairs=65536,
        predictor_exact=not values,
        inexact_raw_byte_values=values,
        predicate=guard,
        accepted_raw_byte_values=accepted,
        scope="Predicate rejection proves source identity over all signed-byte pairs; runtime rate and cost UNKNOWN",
    )


def derive_sparse_pair_predicate(proof, *, limit):
    """Prove a bounded source-pair predicate over the complete signed-byte domain.

    This selects no profitable branch budget. Callers explicitly supply the
    permitted cardinality and separately qualify the complete producer/decoder.
    Groups use raw bytes so generated comparisons remain independent of char
    signedness. Ordered replay and source floating obligations are unchanged.
    """
    if type(limit) is not int or not 1 <= limit <= 16:
        raise ValueError("explicit sparse pair limit1..16 required")
    try:
        unchanged = proof == derive(**proof["source"], **proof["predictor"])
    except (KeyError, TypeError, ValueError) as failure:
        raise ValueError("unchanged complete pair certificate required") from failure
    if not unchanged:
        raise ValueError("unchanged complete pair certificate required")
    bitmap = bytes.fromhex(proof["correction_bitmap_hex"])
    keys = [key for key in range(65536) if bitmap[key >> 3] & (1 << (key & 7))]
    if len(keys) > limit:
        raise ValueError("complete correction relation exceeds explicit sparse pair limit")
    groups = {}
    for key in keys:
        groups.setdefault(key >> 8, []).append(key & 255)
    rows = [dict(lhs_raw=lhs, rhs_raw=rhs) for lhs, rhs in sorted(groups.items())]
    # Compare the grouped predicate with every original bitmap decision rather
    # than granting eligibility from examples or observed operand ranges.
    for key in range(65536):
        accepted = any(key >> 8 == row["lhs_raw"] and (key & 255) in row["rhs_raw"] for row in rows)
        if accepted != bool(bitmap[key >> 3] & (1 << (key & 7))):
            raise ValueError("sparse pair predicate does not cover the complete relation")
    expression = (
        " || ".join(
            f"((uint8_t)a[i]=={row['lhs_raw']} && ("
            + " || ".join(f"(uint8_t)b[i]=={rhs}" for rhs in row["rhs_raw"])
            + "))"
            for row in rows
        )
        or "0"
    )
    return dict(
        schema="quantized_affine_sparse_pair_predicate_v1",
        source_table_sha256=proof["source_table_sha256"],
        predictor_table_sha256=proof["predictor_table_sha256"],
        correction_bitmap_sha256=proof["correction_bitmap_sha256"],
        pairs=65536,
        cardinality=len(keys),
        limit=limit,
        groups=rows,
        predicate_c=expression,
        scope=(
            "Exact mismatch predicate; unchanged source replay, alias/input/FP obligations "
            "and complete runtime cost remain caller proofs"
        ),
    )


def derive_source_word_guard(proof, *, limit):
    """Derive source bytes which every complete correction pair must contain.

    A rejected word cannot contain a mismatch, so it requires no access to the
    other source or predicted output. Choose the axis with fewer distinct bytes
    as an explicit code-size ranking; runtime hit rate and cost remain unknown.
    The boolean zero-byte test accepts every hit, including adjacent borrow
    cases. Individual suspect lanes still require the exact pair predicate.
    """
    sparse = derive_sparse_pair_predicate(proof, limit=limit)
    lhs = sorted(row["lhs_raw"] for row in sparse["groups"])
    rhs = sorted({byte for row in sparse["groups"] for byte in row["rhs_raw"]})
    axis, values = ("lhs", lhs) if len(lhs) <= len(rhs) else ("rhs", rhs)
    return dict(
        schema="quantized_affine_source_word_guard_v1",
        source_table_sha256=proof["source_table_sha256"],
        predictor_table_sha256=proof["predictor_table_sha256"],
        correction_bitmap_sha256=proof["correction_bitmap_sha256"],
        pairs=65536,
        sparse_cardinality=sparse["cardinality"],
        axis=axis,
        raw_byte_values=values,
        scope=(
            "No-hit source words prove every predicted lane source-exact; "
            "suspect words require exact pair tests; immutable sources and disjoint fresh output required"
        ),
        runtime_hit_rate="UNKNOWN",
    )


def _source_word_hit_expression(values):
    """Boolean any-byte equality; high-bit positions may include borrow flags."""
    terms = []
    for byte in values:
        difference = f"(raw^UINT64_C(0x{byte * 0x0101010101010101:016x}))"
        terms.append(f"(({difference}-UINT64_C(0x0101010101010101))&~{difference}&UINT64_C(0x8080808080808080))")
    return "|".join(terms) or "UINT64_C(0)"


def emit_correction(
    proof, symbol, *, packed_prefix=True, output_value_guard=False, sparse_pair_limit=0, source_word_guard=False
):
    """Correct only certified mismatches; an optional safe word guard skips scans.

    Inputs must retain the original source bytes after predictor execution. The
    caller must establish nonoverlapping output/input storage for that predictor.
    Floating arithmetic must use RN-even with gradual underflow and preserve the
    emitted separate operations; a target's prediction requires its own proof.
    """
    valid_identifier = _valid_c_identifier(symbol)
    if (
        not valid_identifier
        or type(packed_prefix) is not bool
        or type(output_value_guard) is not bool
        or type(sparse_pair_limit) is not int
        or not 0 <= sparse_pair_limit <= 16
        or type(source_word_guard) is not bool
        or (source_word_guard and (not packed_prefix or not sparse_pair_limit))
        or proof != derive(**proof["source"], **proof["predictor"])
    ):
        raise ValueError("valid identifier and unchanged complete pair certificate required")
    values = ",".join(map(str, bytes.fromhex(proof["correction_bitmap_hex"])))
    sparse = derive_sparse_pair_predicate(proof, limit=sparse_pair_limit) if sparse_pair_limit else None
    source_guard = derive_source_word_guard(proof, limit=sparse_pair_limit) if source_word_guard else None
    source = proof["source"]
    relu = "if (sum < 0.0f) sum = 0.0f;" if source["relu"] else ""
    low = 0 if source["relu"] else -128
    output_guard = derive_output_value_guard(proof) if output_value_guard else None
    if (output_guard is not None and output_guard["predictor_exact"]) or (
        source_guard is not None and not source_guard["raw_byte_values"]
    ):
        return f"""#include <stdint.h>
#include <stddef.h>
void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 (void)a;(void)b;(void)c;(void)n;
}}
"""
    bit_predicate = output_guard["predicate"] if output_guard is not None else None
    byte_guard = (
        f" if(((uint8_t)c[i]&{bit_predicate['mask']})!={bit_predicate['value']})return;\n"
        if bit_predicate is not None
        else ""
    )
    bitmap_declaration = f"static const uint8_t {symbol}_ambiguous[8192] __attribute__((aligned(64)))={{{values}}};\n"
    pair_check = (
        " unsigned key=((unsigned)(uint8_t)a[i]<<8)|(unsigned)(uint8_t)b[i];\n"
        f" if ({symbol}_ambiguous[key>>3] & (1u<<(key&7))) {symbol}_replay(a,b,c,i);\n"
    )
    if sparse is not None:
        bitmap_declaration = ""
        pair_check = f" if ({sparse['predicate_c']}) {symbol}_replay(a,b,c,i);\n"
    code = f"""#include <stdint.h>
#include <stddef.h>
{bitmap_declaration}\
static __attribute__((noinline)) void {symbol}_replay(const int8_t *a,const int8_t *b,int8_t *c,size_t i) {{
 volatile float lhs=(float)a[i]*{source["lhs_scale"].hex()}f;
 volatile float rhs=(float)b[i]*{source["rhs_scale"].hex()}f;
 volatile float sum=lhs+rhs;
 {relu}
 volatile float scaled=sum*{proof["reciprocal"].hex()}f;
 int32_t result;
 if (scaled <= {float(low).hex()}f) result={low};
 else if (scaled >= 127.0f) result=127;
 else {{
  result=(int32_t)scaled;
  float delta=scaled-(float)result;
  float absolute=delta<0.0f?-delta:delta;
  if (absolute>0.5f || (absolute==0.5f && (result&1))) result+=scaled<0.0f?-1:1;
 }}
 c[i]=(int8_t)result;
}}
static inline __attribute__((always_inline)) void {symbol}_pair(const int8_t *a,const int8_t *b,int8_t *c,size_t i) {{
{byte_guard}\
{pair_check}\
}}
"""
    if source_guard is not None:
        selected = "a" if source_guard["axis"] == "lhs" else "b"
        any_hit = _source_word_hit_expression(source_guard["raw_byte_values"])
        return (
            code
            + f"""static __attribute__((noinline)) void {symbol}_check_word(
 const int8_t *a,const int8_t *b,int8_t *c) {{
 for(size_t lane=0;lane<8;lane++){symbol}_pair(a,b,c,lane);
}}
typedef uint64_t {symbol}_word __attribute__((may_alias));
void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 size_t i=0,packed_end=n&~(size_t)7;
 if(!((uintptr_t){selected}&7))for(;i<packed_end;i+=8) {{
  uint64_t raw=*(const {symbol}_word*)({selected}+i);
  if(!({any_hit}))continue;
  {symbol}_check_word(a+i,b+i,c+i);
 }}
 for(;i<n;i++){symbol}_pair(a,b,c,i);
}}
"""
        )
    if output_guard is not None:
        loop = ""
        if packed_prefix and bit_predicate is not None:
            mask = bit_predicate["mask"] * 0x0101010101010101
            value = bit_predicate["value"] * 0x0101010101010101
            loop = f"""static __attribute__((noinline)) void {symbol}_check_word(
 const int8_t *a,const int8_t *b,int8_t *c) {{
 for(size_t lane=0;lane<8;lane++){symbol}_pair(a,b,c,lane);
}}
typedef uint64_t {symbol}_word __attribute__((may_alias));
void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 size_t i=0,packed_end=n&~(size_t)7;
 if(!((uintptr_t)c&7))for(;i<packed_end;i+=8) {{
  uint64_t predicted=*(const {symbol}_word*)(c+i);
  uint64_t difference=(predicted&UINT64_C(0x{mask:016x}))^UINT64_C(0x{value:016x});
  if(!((difference-UINT64_C(0x0101010101010101))&~difference&UINT64_C(0x8080808080808080)))continue;
  {symbol}_check_word(a+i,b+i,c+i);
 }}
 for(;i<n;i++){symbol}_pair(a,b,c,i);
}}
"""
        else:
            loop = f"""void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 for(size_t i=0;i<n;i++){symbol}_pair(a,b,c,i);
}}
"""
        return code + loop
    guard = proof["packed_prefix_guard"] if packed_prefix else None
    bit_guard = proof["predictor_bit_guard"] if packed_prefix else None
    if guard is None and bit_guard is None:
        return (
            code
            + f"""void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 for(size_t i=0;i<n;i++) {symbol}_pair(a,b,c,i);
}}
"""
        )

    def broadcast(value):
        return value * 0x0101010101010101

    selected = "a" if guard is None or guard["axis"] == "lhs" else "b"
    prefix = ""
    if guard is not None:
        prefix = (
            f"  uint64_t raw=*(const {symbol}_word*)({selected}+i);\n  uint64_t outside={_prefix_expression(guard)};\n"
        )
    else:
        prefix = "  uint64_t outside=UINT64_C(0x8080808080808080);\n"
    alignment = f"!((uintptr_t){selected}&7)"
    if bit_guard is not None:
        alignment = f"({alignment}) && !((uintptr_t)c&7)"
        if bit_guard["mask"] == 1 and bit_guard["value"] == 1:
            predicate = "predicted<<7"
        elif bit_guard["mask"] == 1 and bit_guard["value"] == 0:
            predicate = "~predicted<<7"
        else:
            prefix += (
                f"  uint64_t difference=(predicted&UINT64_C(0x{broadcast(bit_guard['mask']):016x}))"
                f"^UINT64_C(0x{broadcast(bit_guard['value']):016x});\n"
            )
            predicate = "(difference-UINT64_C(0x0101010101010101))&~difference"
        prefix = f"  uint64_t predicted=*(const {symbol}_word*)(c+i);\n" + prefix
        prefix += f"  outside&=({predicate});\n"
    first_bits = ",".join(str((value & -value).bit_length() - 1 if value else 0) for value in range(256))
    return (
        code
        + f"""
_Static_assert(__BYTE_ORDER__==__ORDER_LITTLE_ENDIAN__,"little-endian word guard required");
typedef uint64_t {symbol}_word __attribute__((may_alias));
static const uint8_t {symbol}_first_bit[256]={{{first_bits}}};
void {symbol}(const int8_t *a,const int8_t *b,int8_t *c,size_t n) {{
 size_t i=0;
 if ({alignment}) for(;n-i>=8;i+=8) {{
{prefix}  // Gather one flag per byte, then inspect only flagged lanes.
  unsigned lanes=(unsigned)({_GATHER_EXPRESSION});
  while(lanes) {{
   size_t lane={symbol}_first_bit[lanes];lanes&=lanes-1;
   {symbol}_pair(a,b,c,i+lane);
  }}
 }}
 for(;i<n;i++) {symbol}_pair(a,b,c,i);
}}
"""
    )
