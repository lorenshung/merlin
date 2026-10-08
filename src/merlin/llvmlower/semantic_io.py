"""Preserve an exported tensor call boundary independently of tensor bufferization."""

from __future__ import annotations

import json
from pathlib import Path


def generate_observer(contract_path: Path, output: Path, out_specs, input_arrays, dtype_bytes):
    """Emit logical boundary storage, readback and pointer-based alias observations.

    Bufferization may copy views. The exported ABI restores their layout/ownership at the
    public boundary, using compiled values only. No oracle is admitted to this generator.
    """
    contract = json.loads(contract_path.read_text())
    entries = contract["outputs"]
    if len(entries) != len(out_specs):
        raise ValueError("semantic result count differs from compiled ABI")
    lines = ["/* Generated exported tensor boundary. */", "#include <stdlib.h>", "#define MERLIN_SEMANTIC_IO 1"]
    count = len(entries)
    max_rank = max(1, *(len(e["shape"]) for e in entries))
    lines += [
        f"static unsigned char *semantic_storage[{count}];",
        f"static size_t semantic_capacity[{count}];",
        f"static long semantic_offset[{count}];",
        f"static long semantic_stride[{count}][{max_rank}];",
        f"static long semantic_shape[{count}][{max_rank}];",
        "static long semantic_at(int i, size_t linear) {",
        "long at=semantic_offset[i];",
        "for(int a=MERLIN_OUTPUT_RANKS_SEMANTIC[i]-1;a>=0;a--) {",
        "long d=semantic_shape[i][a]; if(d) { at+=(linear % d)*semantic_stride[i][a]; linear/=d; } }",
        "return at; }",
    ]
    # Declare ranks before their first use.
    lines.insert(
        3,
        f"static const int MERLIN_OUTPUT_RANKS_SEMANTIC[{count}] = {{"
        + ",".join(str(len(e["shape"])) for e in entries)
        + "};",
    )
    for i, entry in enumerate(entries):
        if any(isinstance(v, int) and v < 0 for v in entry["stride"]) or entry["storage_offset"] < 0:
            raise ValueError("unsupported exported tensor boundary layout")
    lines += ["static void merlin_semantic_before(void) {"]
    for i in contract["input_indices"]:
        array, dt = input_arrays[i]
        nbytes = array.size * dtype_bytes[dt]
        lines += [
            f'htif_puts("PRE_BYTES {i} {nbytes}");',
            f"for(size_t b=0;b<{nbytes};b++) {{ htif_putc(' '); htif_putd(((unsigned char*)MERLIN_INPUT_PTR[{i}])[b]); }}",
            "htif_putc('\\n');",
        ]
    lines += ["}", "static void merlin_semantic_after(void) {"]
    # Allocate post arguments first; aliases then share these actual storages.
    post = contract["post_indices"]
    order = post + [i for i in range(count) if i not in post]
    for i in order:
        entry = entries[i]
        physical_shape, physical_dtype = out_specs[i]
        complex_factor = 2 if entry["dtype"].startswith("complex") else 1
        elem = dtype_bytes[physical_dtype] * complex_factor
        shape = entry["shape"]
        for axis, extent in enumerate(shape):
            expression = f"merlin_result_extent({i},{axis})" if extent < 0 else str(extent)
            lines += [f"semantic_shape[{i}][{axis}]={expression};"]
        for axis, stride in enumerate(entry["stride"]):
            if isinstance(stride, dict):
                axes = stride["shape_axes"]
                if any(type(a) is not int or a < 0 or a >= len(shape) for a in axes):
                    raise ValueError("invalid symbolic stride axis")
                expression = "*".join(f"semantic_shape[{i}][{a}]" for a in axes) or "1"
            else:
                expression = str(stride)
            lines += [f"semantic_stride[{i}][{axis}]={expression};"]
        lines += [f"semantic_offset[{i}]={entry['storage_offset']};"]
        aliases = entry.get("alias_outputs", [])
        if aliases:
            j = aliases[0]
            if j not in post:
                raise ValueError("boundary alias must reference a post-call input")
            lines += [
                f"semantic_storage[{i}]=semantic_storage[{j}];",
                f"semantic_capacity[{i}]=semantic_capacity[{j}];",
            ]
        else:
            lines += [f"size_t span_{i}=semantic_offset[{i}]+1;"]
            for axis in range(len(shape)):
                lines += [
                    f"if(semantic_shape[{i}][{axis}]) {{ size_t n=semantic_shape[{i}][{axis}]-1; size_t s=semantic_stride[{i}][{axis}]; if(s && n>(SIZE_MAX-span_{i})/s) abort(); span_{i}+=n*s; }}"
                ]
            lines += [
                f"if(span_{i}>SIZE_MAX/{elem}) abort();",
                f"semantic_capacity[{i}]=span_{i}*{elem};",
                f"semantic_storage[{i}]=(unsigned char*)calloc(span_{i},{elem});",
                f"if(!semantic_storage[{i}]) abort();",
            ]
        lines += [
            f"for(size_t n=0;n<MERLIN_OUTPUT_NBYTES[{i}]/{elem};n++)",
            f"{{ size_t at=semantic_at({i},n); if(at>=semantic_capacity[{i}]/{elem}) abort(); memcpy(semantic_storage[{i}]+at*{elem},(unsigned char*)MERLIN_OUTPUT_PTR[{i}]+n*{elem},{elem}); }}",
        ]
    for input_index, post_index in zip(contract["input_indices"], post):
        lines += [f"MERLIN_INPUT_PTR[{input_index}]=semantic_storage[{post_index}];"]
    for i, entry in enumerate(entries):
        _, dt = out_specs[i]
        elem = dtype_bytes[dt] * (2 if entry["dtype"].startswith("complex") else 1)
        # Independent packed transport buffers avoid corrupting another alias while gathering.
        lines += [
            f"unsigned char *packed_{i}=(unsigned char*)malloc(MERLIN_OUTPUT_NBYTES[{i}]+1);",
            f"if(!packed_{i}) abort();",
            f"for(size_t n=0;n<MERLIN_OUTPUT_NBYTES[{i}]/{elem};n++)",
            f"memcpy(packed_{i}+n*{elem},semantic_storage[{i}]+semantic_at({i},n)*{elem},{elem});",
            f"MERLIN_OUTPUT_PTR[{i}]=packed_{i};",
            f'htif_puts("OUT_META {i} {entry["storage_offset"]} {int(entry["requires_grad"])} {len(entry["shape"])}");',
        ]
        for axis in range(len(entry["shape"])):
            lines += [
                f"htif_putc(' '); htif_putd(semantic_shape[{i}][{axis}]);",
                f"htif_putc(' '); htif_putd(semantic_stride[{i}][{axis}]);",
            ]
        lines += ["htif_putc('\\n');", f'htif_puts("OUT_ALIAS {i}");']
        for j in post:
            lines += [f"if(semantic_storage[{i}]==semantic_storage[{j}]) {{ htif_putc(' '); htif_putd({j}); }}"]
        lines += ["htif_putc('\\n');"]
    lines += ["}"]
    output.write_text("\n".join(lines) + "\n")
