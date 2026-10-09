"""Validate common capsule declarations after their selected operation builder."""

from __future__ import annotations

import copy


def build_entry(entry: dict, binding, *, builders: dict, semantic_block) -> tuple[dict, str]:
    """Dispatch an abstract capsule entry to its op builder -> (capsule dict, interface MLIR)."""
    op = entry.get("op", "matmul")
    if op not in builders:
        raise ValueError(f"no corpus builder for op {op!r} (have {sorted(builders)})")
    cap, mlir = builders[op](entry, binding)
    if entry.get("input_palette") is not None:
        from .input_palette import realize, validate

        palette = validate(entry["input_palette"])
        for index, row in enumerate(cap.get("inputs") or []):
            realize(palette, name=row["name"], shape=row["shape"], dtype=row["dtype"], index=index)
        cap["input_palette"] = copy.deepcopy(palette)
    # AN EPILOGUE THE BUILDER DID NOT CARRY IS A COVERAGE LIE, so it is refused here rather than in each
    # builder. Only `matmul` and `conv2d` read the entry's `epilogue:`; every other builder writes its
    # own (often empty) list, so a stage declared on, say, a `movement` entry vanished from the capsule
    # -- while `_semantic_block` still credited that stage's FAMILY in `composed_families` off the same
    # entry. The capsule would then be counted as evidence for a family whose arithmetic no engine ever
    # performed, which is the exact failure the pooling epilogue was implemented to close. One check
    # here covers every present and future builder; per-matmul epilogues (resident_reuse) use their own
    # key and are unaffected.
    declared_epilogue = [str(x) for x in (entry.get("epilogue") or [])]
    if declared_epilogue:
        carried = [str(x) for x in ((cap.get("operation") or {}).get("attributes") or {}).get("epilogue", [])]
        dropped = [x for x in declared_epilogue if x not in carried]
        if dropped:
            raise ValueError(
                f"{entry.get('name', op)}: the {op!r} builder dropped epilogue stage(s) {dropped} "
                f"(carried: {carried}). The capsule would still be CREDITED for those stages' semantic "
                f"families, so it would count as evidence for arithmetic nothing computed"
            )
    # THE STIMULUS RANGE AN ENTRY DECLARES, carried here for the same reason: no builder reads it, so
    # an entry that asked for a signed stimulus produced a capsule on the non-negative default, and
    # the sign-sensitive stage it was written to test could not fail. Validated by the ABI's own
    # reader, so a malformed range is refused at generation and not at grading.
    if entry.get("stimulus_range") is not None:
        from merlin.runtime.commandbuffer import STIMULUS_RANGE_KEY, stimulus_range

        cap[STIMULUS_RANGE_KEY] = list(stimulus_range({"params": {STIMULUS_RANGE_KEY: list(entry["stimulus_range"])}}))
    # Stamped once here rather than in each builder, so every capsule a target emits carries the same
    # declaration and a new builder cannot silently forget it.
    if binding.inapplicable_tiers:
        cap["inapplicable_oracle_tiers"] = dict(binding.inapplicable_tiers)
    sem = semantic_block(entry, binding)
    if sem:
        cap["semantic"] = sem
    return cap, mlir
