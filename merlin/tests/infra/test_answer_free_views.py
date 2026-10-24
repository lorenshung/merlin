"""Frozen grants holding answer files are served from answer-free mirrors, not one overlay per file.

bwrap caps its whole argument list, ``--args`` files included, at 9,000. A sealed release corpus has
an answer file in every capsule directory, next to files the agent must read, so one ``/dev/null``
overlay per answer (three arguments each) stops the sandbox before the agent's first turn once a
corpus passes 3,000 capsules. These tests pin the bounded mount plan and that the hermetic coverage
proof still inspects the bytes actually served.
"""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
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


def _masked(argv: list[str], surfaces: list[AnswerSurface], ws: Path) -> list[str]:
    return BW.apply_answer_masks(BW.answer_free_views(argv, surfaces, ws), surfaces)


def _views(ws: Path) -> list[Path]:
    """The published views; their source records and staging trees sit beside them."""
    return sorted(path for path in (ws.parent / "bundle_views").iterdir() if path.is_dir() and path.name[0] != ".")


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

    masked = _masked(argv, surfaces, ws)

    assert len(masked) < len(argv) + 10
    assert BW._DEVNULL not in masked
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
    rewritten = BW.answer_free_views(BW.base_argv(ws, bundle, repo=repo), surfaces, ws)
    assert BW.coverage_gap(rewritten, surfaces) == []

    # An answer that reaches the mirror is exposed again, and the proof says so at the agent's name.
    [view] = _views(ws)
    leaked = view / answers[0].relative_to(repo / "corpus/public")
    leaked.parent.chmod(0o700)
    leaked.write_text("{}\n")
    assert answers[0] in {surface.path for surface in BW.coverage_gap(rewritten, surfaces)}
    masked = BW.apply_answer_masks(rewritten, surfaces)
    assert ["--ro-bind", BW._DEVNULL, str(answers[0])] in [masked[i : i + 3] for i in range(len(masked) - 2)]
    assert BW.coverage_gap(masked, surfaces) == []


def test_mount_point_answer_stays_in_the_mirror_and_is_overlaid(snapshot):
    """A later mount on an answer path needs that path in the mirror; it keeps its overlay."""
    repo, ws, _, answers = snapshot(2)
    denied = {"allowed": [{"path": "corpus/public"}], "denied": [{"path": str(answers[1])}]}
    argv = BW.base_argv(ws, denied, repo=repo)
    surfaces = _frozen(ws, answers, repo)
    masked = _masked(argv, surfaces, ws)

    [view] = _views(ws)
    assert not (view / answers[0].relative_to(repo / "corpus/public")).exists()
    assert (view / answers[1].relative_to(repo / "corpus/public")).is_file()
    assert BW.coverage_gap(masked, surfaces) == []
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

    rewritten = BW.answer_free_views(argv, surfaces, ws)

    assert rewritten == argv
    masked = BW.apply_answer_masks(rewritten, surfaces)
    assert BW.coverage_gap(masked, surfaces) == []


def test_proof_given_only_frozen_surfaces_reaches_a_view_at_any_destination(snapshot):
    """The view's recorded source lets an independent proof name the mirror at an unrelated alias."""
    repo, ws, bundle, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    frozen_corpus = BW._snapshot_path(BW.bundle_snapshot_root(ws), repo / "corpus/public", repo)
    alias = ws.parent / "alias"
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(frozen_corpus), str(alias)]
    rewritten = BW.answer_free_views(argv, surfaces, ws)
    assert str(frozen_corpus) not in rewritten  # both binds are served by the view
    [view] = _views(ws)

    leaked = view / answers[0].relative_to(repo / "corpus/public")
    leaked.parent.chmod(0o700)
    leaked.write_text("{}\n")
    gap = {surface.path for surface in BW.coverage_gap(rewritten, surfaces)}
    assert {answers[0], alias / answers[0].relative_to(repo / "corpus/public")} <= gap
    assert BW.coverage_gap(BW.apply_answer_masks(rewritten, surfaces), surfaces) == []


@pytest.mark.parametrize("above", ["storage", "run"])
def test_bind_above_view_storage_hides_it(snapshot, above):
    """A provider may alias the run directory; views of other commands keep answers no surface names."""
    repo, ws, _, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    denied = {"allowed": [{"path": "corpus/public"}], "denied": [{"path": str(answers[1])}]}
    BW.answer_free_views(BW.base_argv(ws, denied, repo=repo), surfaces, ws)  # keeps answers[1]
    [kept] = _views(ws)
    storage = ws.parent / "bundle_views"
    source = storage if above == "storage" else ws.parent
    bundle = {"allowed": [{"path": "corpus/public"}]}
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(source), "/views"]

    masked = BW.apply_answer_masks(BW.answer_free_views(argv, surfaces, ws), surfaces)

    alias = Path("/views") / storage.relative_to(source)
    stale = alias / kept.name / answers[1].relative_to(repo / "corpus/public")
    assert not BW.is_exposed(masked, stale)
    assert BW.coverage_gap(masked, surfaces) == []
    if _SANDBOX.usable:
        done = subprocess.run(
            [*masked, "bash", "-c", f'test -d "{alias}" && test -z "$(ls -A "{alias}")"'],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert done.returncode == 0, done.stderr


def test_bind_inside_view_storage_is_refused(snapshot):
    """Only a whole view has a recorded source; a path inside storage is served unnamed."""
    repo, ws, bundle, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    BW.answer_free_views(BW.base_argv(ws, bundle, repo=repo), surfaces, ws)
    [view] = _views(ws)
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(view / "case_00000"), "/views"]
    with pytest.raises(RuntimeError, match="inside answer-free view storage"):
        BW.answer_free_views(argv, surfaces, ws)


def test_whole_view_of_another_command_keeps_its_answers_masked(snapshot):
    """A whole view bound again elsewhere is recorded, so the answers it keeps are still overlaid."""
    repo, ws, _, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    denied = {"allowed": [{"path": "corpus/public"}], "denied": [{"path": str(answers[1])}]}
    BW.answer_free_views(BW.base_argv(ws, denied, repo=repo), surfaces, ws)  # keeps answers[1]
    [kept] = _views(ws)
    bundle = {"allowed": [{"path": "corpus/public"}]}
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(kept), "/old"]

    masked = BW.apply_answer_masks(BW.answer_free_views(argv, surfaces, ws), surfaces)

    old_answer = Path("/old") / answers[1].relative_to(repo / "corpus/public")
    assert ["--ro-bind", BW._DEVNULL, str(old_answer)] in [masked[i : i + 3] for i in range(len(masked) - 2)]
    assert BW.coverage_gap(masked, surfaces) == []


def test_concurrent_builders_of_one_view_both_succeed(snapshot):
    repo, ws, _, answers = snapshot(50)
    source = BW._snapshot_path(BW.bundle_snapshot_root(ws), repo / "corpus/public", repo)
    omitted = {answer.relative_to(repo / "corpus/public") for answer in answers}
    storage = ws.parent / "bundle_views"
    with ThreadPoolExecutor(8) as pool:
        built = set(pool.map(lambda _: BW._answer_free_view(source, omitted, storage, "salt"), range(16)))
    [view] = built
    assert [view] == _views(ws)
    assert not any(path.name.endswith(".pending") for path in storage.iterdir())
    assert sum(1 for _ in view.rglob("weights*")) == 0 and sum(1 for _ in view.rglob("capsule.yaml")) == 50


def test_surface_spelled_through_a_symlink_is_masked_through_its_twin(tmp_path, monkeypatch):
    """A run directory reached through a symlink: an answer named by its resolved path is not omitted
    from the view, so it must still be overlaid where the view serves it."""
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    repo, ws = tmp_path / "repo", tmp_path / "link/run/workspace"
    answers = _corpus(repo, 2)
    goldens = []
    for answer in answers:
        goldens.append(answer.parent / "golden.out")
        goldens[-1].write_text("SECRET\n")
    bundle = {"allowed": [{"path": "corpus/public"}]}
    BW.materialize_bundle_inputs(ws, bundle, repo=repo)
    ws.mkdir(parents=True)
    root = BW.bundle_snapshot_root(ws)
    surfaces = [
        *_frozen(ws, answers, repo),
        *(AnswerSurface("golden", BW._snapshot_path(root, path, repo).resolve(), "file", "golden") for path in goldens),
    ]
    argv = BW.base_argv(ws, bundle, repo=repo)

    masked = BW.apply_answer_masks(BW.answer_free_views(argv, surfaces, ws), surfaces)

    assert not any(BW.is_exposed(masked, path) for path in [*answers, *goldens])
    assert BW.coverage_gap(masked, surfaces) == []
    if _SANDBOX.usable:
        done = subprocess.run(
            [*masked, "bash", "-c", f'cat "{goldens[0]}"'], capture_output=True, text=True, timeout=60
        )
        assert "SECRET" not in done.stdout
    BW.remove_bundle_snapshot(ws)


@pytest.mark.parametrize("damage", ["source", "omitted", "garbage"])
def test_view_whose_record_does_not_reproduce_its_name_is_refused(snapshot, damage):
    """A record that does not rebuild the view's name names no twins, so binding that view is refused."""
    repo, ws, _, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    denied = {"allowed": [{"path": "corpus/public"}], "denied": [{"path": str(answers[1])}]}
    BW.answer_free_views(BW.base_argv(ws, denied, repo=repo), surfaces, ws)
    [kept] = _views(ws)
    record = kept.parent / f"{kept.name}.source"
    built = json.loads(record.read_text())
    if damage == "garbage":
        record.write_text("/nonexistent")
    else:
        built[damage] = "/nonexistent" if damage == "source" else []
        record.write_text(json.dumps(built))
    bundle = {"allowed": [{"path": "corpus/public"}]}
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(kept), "/old"]
    with pytest.raises(RuntimeError, match="inside answer-free view storage"):
        BW.answer_free_views(argv, surfaces, ws)


def test_run_directory_alias_keeps_the_plan_bounded(snapshot):
    """A provider aliasing the run directory serves the whole snapshot there; it is hidden, not overlaid."""
    repo, ws, bundle, answers = snapshot(3100)
    surfaces = _frozen(ws, answers, repo)
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(ws.parent), "/metadata-view"]

    masked = BW.apply_answer_masks(BW.answer_free_views(argv, surfaces, ws), surfaces)

    assert len(masked) < len(argv) + 10
    assert BW.coverage_gap(masked, surfaces) == []
    if _SANDBOX.usable:
        done = subprocess.run(
            [*masked, "bash", "-c", "find /metadata-view/bundle_inputs /metadata-view/bundle_views -type f | wc -l"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.split() == ["0"]


def test_denying_a_whole_source_denies_a_view_of_it(snapshot):
    """A directory surface that is a view's source covers the whole view wherever it is bound."""
    repo, ws, bundle, answers = snapshot(2)
    surfaces = _frozen(ws, answers, repo)
    BW.answer_free_views(BW.base_argv(ws, bundle, repo=repo), surfaces, ws)
    [view] = _views(ws)
    frozen_corpus = BW._snapshot_path(BW.bundle_snapshot_root(ws), repo / "corpus/public", repo)
    denied = [*surfaces, AnswerSurface("hidden corpus", frozen_corpus, "dir", "hidden")]
    argv = [*BW.base_argv(ws, bundle, repo=repo), "--ro-bind", str(view), "/old"]

    masked = BW.apply_answer_masks(BW.answer_free_views(argv, denied, ws), denied)

    assert not BW.is_exposed(masked, Path("/old") / "case_00000/capsule.yaml")
    assert BW.coverage_gap(masked, denied) == []
    if _SANDBOX.usable:
        done = subprocess.run([*masked, "bash", "-c", 'test -z "$(ls -A /old)"'], capture_output=True, timeout=60)
        assert done.returncode == 0, done.stderr


def test_later_bind_over_hidden_storage_still_takes_effect(snapshot, tmp_path):
    """Storage is hidden right after the bind that serves it, so a later bind at that path still wins."""
    repo, ws, bundle, answers = snapshot(2)
    public = tmp_path / "public"
    public.mkdir()
    (public / "readme").write_text("public\n")
    argv = [
        *BW.base_argv(ws, bundle, repo=repo),
        *("--ro-bind", str(ws.parent), "/alias"),
        *("--ro-bind", str(public), "/alias/bundle_views"),
    ]

    rewritten = BW.answer_free_views(argv, _frozen(ws, answers, repo), ws)

    assert BW.is_exposed(rewritten, Path("/alias/bundle_views/readme"))
    if _SANDBOX.usable:
        done = subprocess.run(
            [*rewritten, "cat", "/alias/bundle_views/readme"], capture_output=True, text=True, timeout=60
        )
        assert done.stdout == "public\n", done.stderr
