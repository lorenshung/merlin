"""Shape EMISSION outcomes are observable without a golden.

The numerical generalization difftest needs a CPU reference for the operand format, so it cannot run on
an MX/fp8 datapath -- and those are exactly the targets whose shape coverage is most in doubt. Measured
while building this: all four multi-tile probes on an fp8 target were skipped for want of a golden and
the suite reported ``0 graded``, which reads as "nothing to report" rather than "could not look".

The probe directly records declines, empty command buffers and emitter errors per axis. Artifact
size is advisory: a larger problem can select a compact runtime loop instead of an unrolled path.
Neither nonempty emission nor size proves executed work or numerical correctness.

An older frozen submission emitted 418 lines at one tile and 5 at two M-tiles; that was a useful
diagnostic clue, but the size difference alone was not a general proof of noncoverage.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import lowering_coverage as LC

# --------------------------------------------------------------- the interface it probes with


def test_the_probe_interface_is_the_capsule_shape_with_different_extents():
    """Only the extents differ from a real corpus capsule -- otherwise this is a new op, not a probe."""
    mlir = LC.contraction_interface(64, 32, 32, target="t", operand_mlir="f8E4M3FN", accum_mlir="bf16")
    assert "tensor<64x32xf8E4M3FN>" in mlir, "A0 carries M x K"
    assert "tensor<32x32xf8E4M3FN>" in mlir, "W carries K x N"
    assert "merlin_iface.acc<bf16>" in mlir
    assert "tensor<64x32xbf16>" in mlir, "the commit carries M x N"
    assert 'merlin_iface.target = "t"' in mlir


def test_every_corner_is_a_multiple_of_the_tile_edge():
    """The corners are ratios, resolved against the target's DERIVED tile -- never absolute literals."""
    assert LC.CORNERS["tile"] == (1, 1, 1)
    assert LC.CORNERS["m_2tiles"] == (2, 1, 1)
    assert LC.CORNERS["k_2tiles"] == (1, 2, 1)
    assert LC.CORNERS["n_2tiles"] == (1, 1, 2)


def test_tail_corners_cover_subtile_and_each_independent_remainder_axis():
    assert LC.TAIL_CORNERS["sub_tile"] == (-1, -1, -1)
    assert LC.TAIL_CORNERS["m_tail"] == (1, 0, 0)
    assert LC.TAIL_CORNERS["k_tail"] == (0, 1, 0)
    assert LC.TAIL_CORNERS["n_tail"] == (0, 0, 1)


# --------------------------------------------------------------- emit outcomes and advisory size


def _sweep(monkeypatch, work_by_corner, declined=()):
    """Drive sweep() with a fake emit path so the invariant is tested, not a real backend."""
    monkeypatch.setattr(
        LC,
        "_binding",
        lambda t: type(
            "B",
            (),
            {
                "operand_dtype": "int8",
                "accum_dtype": "int32",
                "mlir_dtype": staticmethod(lambda tok: {"int8": "i8"}.get(tok, "i32")),
            },
        )(),
    )
    monkeypatch.setattr(LC, "tile_edge", lambda t: 32)

    def fake(
        package, *, target, m, k, n, operand_mlir, accum_mlir, contract=None, timeout=300, additional_forbidden=()
    ):
        corner = next(c for c, (fm, fk, fn) in LC.CORNERS.items() if (32 * fm, 32 * fk, 32 * fn) == (m, k, n))
        if corner in declined:
            return "declined", "no loop over this axis", 0
        return "lowered", None, work_by_corner[corner]

    monkeypatch.setattr(LC, "probe_shape", fake)
    return LC.sweep("pkg", target="t", tail_corners={})


def test_tail_failures_are_named_per_axis_without_a_golden(monkeypatch):
    monkeypatch.setattr(
        LC,
        "_binding",
        lambda t: type(
            "B",
            (),
            {
                "operand_dtype": "int8",
                "accum_dtype": "int32",
                "mlir_dtype": staticmethod(lambda tok: {"int8": "i8"}.get(tok, "i32")),
            },
        )(),
    )
    monkeypatch.setattr(LC, "tile_edge", lambda t: 32)

    def fake(
        package, *, target, m, k, n, operand_mlir, accum_mlir, contract=None, timeout=300, additional_forbidden=()
    ):
        if (m, k, n) == (32, 32, 33):
            return "declined", "no N-tail legalization", 0
        return "lowered", None, 100

    monkeypatch.setattr(LC, "probe_shape", fake)
    result = LC.sweep(
        "pkg",
        target="t",
        corners={"tile": (1, 1, 1)},
        tail_corners={"sub_tile": (-1, -1, -1), "m_tail": (1, 0, 0), "k_tail": (0, 1, 0), "n_tail": (0, 0, 1)},
    )

    assert result["tail_axes_uncovered"] == ["n"]
    assert result["tail_cases_uncovered"] == ["n_tail"]
    assert result["all_covered"] is False


def test_a_shrinking_artifact_is_reported_without_claiming_a_silent_refusal(monkeypatch):
    """The historical 418-to-5 size difference remains visible, but is not a verdict."""
    r = _sweep(monkeypatch, {"tile": 418, "m_2tiles": 5, "k_2tiles": 1187, "n_2tiles": 1205})
    by = {c["corner"]: c["outcome"] for c in r["corners"]}
    assert by["m_2tiles"] == by["k_2tiles"] == by["n_2tiles"] == "lowered"
    assert r["smaller_emitted_artifacts"] == ["m_2tiles"]
    assert r["multi_tile_axes_uncovered"] == []
    assert r["all_covered"] is True
    assert r["scope"] == "public_emit_only"
    assert "not executed work or numerical correctness" in r["note"]


def test_smaller_nonempty_multitile_artifact_is_advisory_not_a_coverage_failure(monkeypatch):
    """A larger shape may select a compact runtime loop instead of an unrolled tile path."""
    r = _sweep(monkeypatch, {"tile": 30, "m_2tiles": 20, "k_2tiles": 30, "n_2tiles": 30})
    assert {c["corner"]: c["outcome"] for c in r["corners"]}["m_2tiles"] == "lowered"
    assert r["all_covered"] is True
    assert r["smaller_emitted_artifacts"] == ["m_2tiles"]


def test_work_that_grows_on_every_axis_is_covered(monkeypatch):
    """The measured gemmini shape: 29 words at one tile, 37/37/40 at two."""
    r = _sweep(monkeypatch, {"tile": 29, "m_2tiles": 37, "k_2tiles": 37, "n_2tiles": 40})
    assert r["all_covered"] is True
    assert r["multi_tile_axes_uncovered"] == []
    assert r["n_collapsed"] == 0


def test_a_stated_decline_is_uncovered_but_not_a_collapse(monkeypatch):
    """Declining is the HONEST way to not cover a shape. Still uncovered; no longer silent."""
    r = _sweep(monkeypatch, {"tile": 418, "m_2tiles": 0, "k_2tiles": 900, "n_2tiles": 900}, declined=("m_2tiles",))
    by = {c["corner"]: c["outcome"] for c in r["corners"]}
    assert by["m_2tiles"] == "declined"
    assert r["n_declined"] == 1 and r["n_collapsed"] == 0
    assert r["multi_tile_axes_uncovered"] == ["m"], "honest, but still not covered"


def test_equal_work_is_not_flagged(monkeypatch):
    """A backend may emit a loop whose text does not grow with the trip count."""
    r = _sweep(monkeypatch, {"tile": 100, "m_2tiles": 100, "k_2tiles": 100, "n_2tiles": 100})
    assert r["all_covered"] is True
    assert r["n_collapsed"] == 0


def test_optional_observer_sees_exact_lowered_artifact_without_changing_emit_result(monkeypatch):
    artifact = 'module { llvm.func @fixture_entry() { llvm.return } }\n'
    seen = []
    monkeypatch.setattr(
        LC, "run_entrypoints", lambda *_args, **_kwargs: (None, {"commands": [{"kind": "fixture"}]}, artifact)
    )
    result = LC.probe_shape(
        "candidate", target="fixture", m=4, k=4, n=4, operand_mlir="i8", accum_mlir="i32",
        on_lowered=lambda text, work: seen.append((text, work.is_dir())),
    )
    assert result == ("lowered", None, 1)
    assert seen == [(artifact, True)]


def test_a_failing_baseline_refuses_to_attribute_anything_to_shape(monkeypatch):
    """With the one-tile baseline down, every multi-tile corner fails for reasons unrelated to shape.

    Reporting "M, K and N all uncovered" there would be a confident, specific, wrong answer -- which is
    worse than no answer. Measured: substituting a gradeable operand dtype moved the probe onto a
    different lowering path with a different tile edge and produced exactly that false reading.
    """
    r = _sweep(
        monkeypatch,
        {"tile": 0, "m_2tiles": 0, "k_2tiles": 0, "n_2tiles": 0},
        declined=("tile", "m_2tiles", "k_2tiles", "n_2tiles"),
    )
    assert r["baseline_tile_lowered"] is False
    assert r["multi_tile_axes_uncovered"] == []
    assert r["all_covered"] is False
    assert "fix the baseline" in r["unmeasured"]


# --------------------------------------------------------------- end to end, on the real submissions


@pytest.mark.parametrize(
    "target,pkg,expected_smaller",
    [
        ("atlas", "out/runs/atlas/capsule-bench/merlin_assisted/merlincirct_atlas_arm4_v1/submission", ["m_2tiles"]),
        ("gemmini", "out/runs/gemmini/capsule-bench/merlin_assisted/merlincirct_gemarm4_codex/submission", []),
    ],
)
def test_the_frozen_submissions_preserve_size_observations_as_advisory(target, pkg, expected_smaller, monkeypatch):
    """Retain the historical size observation without turning it into a coverage verdict."""
    from merlin.common.paths import repo_root

    p = repo_root() / pkg
    if not (p / "manifest.yaml").is_file():
        pytest.skip(f"frozen submission not present: {pkg}")
    # monkeypatch, NOT os.environ: pytest unwinds it after the test. Setting it directly leaked the
    # descriptor into every later test in the process -- and because this file sorts before
    # test_model_grade.py, the eight tests holding "a model capsule only passes if its layers actually
    # ran on the mesh" resolved a different target, failed closed, and reported `incomplete`. They
    # passed in isolation and failed in the full suite, which is the shape of a guard nobody trusts.
    monkeypatch.setenv(
        "MERLIN_TARGET_EXPERIMENT",
        str(repo_root() / f"merlin/experiments/capsule_bench/targets/{target}/target_experiment.yaml"),
    )
    cov = LC.sweep(p, target=target, contract=str(repo_root() / "merlin/contract"))
    assert cov["baseline_tile_lowered"] is True, "the one-tile baseline must lower for this to mean anything"
    multi_tile = {"m_2tiles", "k_2tiles", "n_2tiles"}
    assert sorted(set(cov["smaller_emitted_artifacts"]) & multi_tile) == expected_smaller
