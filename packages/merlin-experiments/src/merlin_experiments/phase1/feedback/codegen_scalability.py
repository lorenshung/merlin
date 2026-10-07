"""Public, fact-derived emitted-text growth observations for compiler authoring.

No model capture, validation shape, numerical answer or simulator is read here.
Text size is an observation, not executed work or a correctness theorem. A
runtime loop may correctly produce constant-sized (or smaller) code.
"""

from __future__ import annotations

import time
from pathlib import Path


def run(
    package: str | Path,
    *,
    target: str,
    contract: str | Path | None = None,
    timeout: int = 30,
    additional_forbidden: tuple[str, ...] = (),
) -> dict:
    """Observe the same public contraction at geometric multiples of its tile.

    The experiment ratios are fixed, but geometry and formats come from the
    selected target inputs. Neither a size threshold nor a compiler strategy
    follows from these measurements. Whole-model build and numerical gates
    remain separate, mandatory evidence.
    """
    from merlin.targetgen.capability_probes import tile_edge
    from merlin.targetgen.corpora import experiment_for
    from merlin.targetgen.corpus_spec import derive_binding
    from merlin.targetgen.lowering_coverage import probe_shape

    experiment = experiment_for(target)
    if experiment is None:
        raise ValueError("public scalability probe requires an explicitly selected target experiment")
    binding = derive_binding(experiment, {})
    tile = tile_edge(target, operand=binding.operand_dtype)
    samples = []
    baseline = None
    for scale in (1, 2, 4, 8):
        shape = (tile * scale,) * 3
        started = time.monotonic()
        outcome, detail, lines = probe_shape(
            package,
            target=target,
            m=shape[0],
            k=shape[1],
            n=shape[2],
            operand_mlir=binding.mlir_dtype(binding.operand_dtype),
            accum_mlir=binding.mlir_dtype(binding.accum_dtype),
            contract=contract,
            timeout=timeout,
            additional_forbidden=additional_forbidden,
        )
        elapsed = time.monotonic() - started
        if scale == 1 and outcome == "lowered" and lines > 0:
            baseline = lines
        sample = {
            "extent_scale": scale,
            "shape_mkn": list(shape),
            "contraction_volume_ratio": scale**3,
            "outcome": outcome,
            "artifact_nonempty_lines": lines,
            "line_ratio_vs_baseline": lines / baseline if baseline and outcome == "lowered" else None,
            "emit_pipeline_wall_s": elapsed,
        }
        if detail:
            sample["detail"] = detail
        samples.append(sample)
    return {
        "schema": "merlin.codegen_scalability_probe.v1",
        "scope": "public_emit_only",
        "tile_edge": tile,
        "samples": samples,
        "observations_complete": all(
            row["outcome"] == "lowered" and row["artifact_nonempty_lines"] > 0 for row in samples
        ),
        "note": (
            "Public fact-derived emit-only observations: not compilation, executed work, numerical "
            "equivalence or certification. Constant-sized or smaller looped code can be correct. "
            "Use the existing compiler/build and native numerical checks before claiming success."
        ),
    }
