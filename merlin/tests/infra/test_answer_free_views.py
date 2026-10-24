"""Frozen grants holding answer files are served from answer-free mirrors, not one overlay per file.

bwrap caps its whole argument list, ``--args`` files included, at 9,000. A sealed release corpus has
an answer file in every capsule directory, next to files the agent must read, so one ``/dev/null``
overlay per answer (three arguments each) stops the sandbox before the agent's first turn once a
corpus passes 3,000 capsules. These tests pin the bounded mount plan and that the hermetic coverage
proof still inspects the bytes actually served.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.targetgen.sandbox import bwrap as BW
from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface
from merlin.targetgen.sandbox.preflight import probe_sandbox

_SANDBOX = probe_sandbox()
_BWRAP_MAX_ARGS = 9000  # bubblewrap.c MAX_ARGS: counts argv plus every --args file


def _corpus(repo: Path, n: int) -> list[Path]:
    corpus = repo / "corpus/public"
    for index in range(n):
        case = corpus / f"case_{index:05d}"
        case.mkdir(parents=True)
        (case / "capsule.yaml").write_text(f"name: case_{index}\n")
        (case / "weights.safetensors.manifest.json").write_text("{}\n")
    return sorted(corpus.glob("*/weights.safetensors.manifest.json"))


def _frozen(ws: Path, answers: list[Path], repo: Path) -> list[AnswerSurface]:
    """Answer surfaces at their live names and at their frozen snapshot names, as the final pass sees them."""
    root = BW.bundle_snapshot_root(ws)
    return [AnswerSurface("weight", path, "file", "weight") for path in answers] + [
        AnswerSurface("frozen weight", BW._snapshot_path(root, path, repo), "file", "weight") for path in answers
    ]


def _masked(argv: list[str], surfaces: list[AnswerSurface], ws: Path) -> tuple[list[str], list[AnswerSurface]]:
    rewritten, checked = BW.answer_free_views(argv, surfaces, ws)
    return BW.apply_answer_masks(rewritten, checked), checked


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    repo, ws = tmp_path / "repo", tmp_path / "run/workspace"
    made = []

    def make(n: int) -> tuple[Path, Path, dict, list[Path]]:
        answers = _corpus(repo, n)
        bundle = {"allowed": [{"path": "corpus/public"}]}
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        ws.mkdir(parents=True)
        made.append(ws)
        return repo, ws, bundle, answers

    yield make
    for path in made:
        BW.remove_bundle_snapshot(path)


def test_corpus_over_the_bwrap_argument_cap_stays_bounded(snapshot):
    repo, ws, bundle, answers = snapshot(3100)  # 3,100 overlays alone would be 9,300 arguments
    argv = BW.base_argv(ws, bundle, repo=repo)
    surfaces = _frozen(ws, answers, repo)
    assert len(BW.apply_answer_masks(argv, surfaces)) > _BWRAP_MAX_ARGS  # the per-file plan bwrap refuses

    masked, checked = _masked(argv, surfaces, ws)

    assert len(masked) < len(argv) + 10
    assert BW._DEVNULL not in masked
    assert BW.coverage_gap(masked, checked) == []
    assert BW.coverage_gap(masked, surfaces) == []
    assert all(not BW.is_exposed(masked, path) for path in answers)
    assert BW.is_exposed(masked, answers[0].parent / "capsule.yaml")
    if _SANDBOX.usable:
        corpus = repo / "corpus/public"
        probe = f'find "{corpus}" -name "weights*" | wc -l; ls "{corpus}" | wc -l'
        done = subprocess.run(
            ["bash", "-c", BW.compose_command(masked, f" bash -c '{probe}'", ws)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["0", "3100"]
    assert answers[0].is_file()  # the host corpus and its snapshot keep every answer
    BW.remove_bundle_snapshot(ws)
    assert not (ws.parent / "bundle_views").exists()


def test_coverage_proof_inspects_the_served_mirror(snapshot):
    repo, ws, bundle, answers = snapshot(3)
    surfaces = _frozen(ws, answers, repo)
    rewritten, checked = BW.answer_free_views(BW.base_argv(ws, bundle, repo=repo), surfaces, ws)
    assert BW.coverage_gap(rewritten, checked) == []

    # An answer that reaches the mirror is exposed again, and the proof says so at the agent's name.
    view = next((ws.parent / "bundle_views").iterdir())
    leaked = view / answers[0].relative_to(repo / "corpus/public")
    leaked.parent.chmod(0o700)
    leaked.write_text("{}\n")
    assert answers[0] in {surface.path for surface in BW.coverage_gap(rewritten, checked)}
    masked = BW.apply_answer_masks(rewritten, checked)
    assert ["--ro-bind", BW._DEVNULL, str(answers[0])] in [masked[i : i + 3] for i in range(len(masked) - 2)]
    assert BW.coverage_gap(masked, checked) == []


def test_mount_point_answer_stays_in_the_mirror_and_is_overlaid(snapshot):
    """A later mount on an answer path needs that path in the mirror; it keeps its overlay."""
    repo, ws, _, answers = snapshot(2)
    denied = {"allowed": [{"path": "corpus/public"}], "denied": [{"path": str(answers[1])}]}
    argv = BW.base_argv(ws, denied, repo=repo)
    masked, checked = _masked(argv, _frozen(ws, answers, repo), ws)

    view = next((ws.parent / "bundle_views").iterdir())
    assert not (view / answers[0].relative_to(repo / "corpus/public")).exists()
    assert (view / answers[1].relative_to(repo / "corpus/public")).is_file()
    assert BW.coverage_gap(masked, checked) == []
    if _SANDBOX.usable:
        probe = f'test ! -e "{answers[0]}" && test ! -s "{answers[1]}" && test -s "{answers[0].parent}/capsule.yaml"'
        done = subprocess.run([*masked, "bash", "-c", probe], capture_output=True, text=True, timeout=60)
        assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("covered", ["corpus/public", "corpus/public/case_00000"])
def test_source_whose_projection_depends_on_its_path_is_never_mirrored(snapshot, covered):
    """A directory surface covering a grant, or one declaring a public subtree inside it, keeps the frozen bind."""
    repo, ws, bundle, answers = snapshot(2)
    root = BW.bundle_snapshot_root(ws)
    frozen = BW._snapshot_path(root, repo / covered, repo)
    grantable = ("contracts",) if covered != "corpus/public" else ()
    surfaces = [*_frozen(ws, answers, repo), AnswerSurface("support", frozen, "dir", "backend", grantable)]
    argv = BW.base_argv(ws, bundle, repo=repo)

    rewritten, checked = BW.answer_free_views(argv, surfaces, ws)

    assert rewritten == argv and checked == surfaces
    masked = BW.apply_answer_masks(rewritten, checked)
    assert BW.coverage_gap(masked, surfaces) == []
