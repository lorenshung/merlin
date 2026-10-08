"""The gates that keep target coupling visible: the coupling scan and the lift detector.

These exist because the repo's cardinal rule ("derive, never hardcode") was being enforced by a check
with two blind spots that happened to cancel out. The whole-identifier name check matches ``gemmini``
but not ``gemmini_kernel`` or ``cycle_window_gemmini_region``, so vendor SYMBOL coupling was invisible;
and the coupling scan skips any file whose own path names a target, since self-reference is legitimate.
A module called ``<target>_<thing>.py`` therefore fell through both — which is how two fully general
modules (a fixed-format linker and boot builder, ``target`` a parameter throughout, derived ISA facts
end to end) sat in a vendor-named home for months with nothing asking whether they belonged there.

Tested against SYNTHETIC files rather than the live tree: a test that asserts today's counts turns into
a chore that gets bumped, and it would pass just as well if the scan stopped working.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_no_target_name.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_check_no_target_name", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ what "owned by a target" means
def test_a_path_naming_a_target_is_treated_as_that_targets_own_module(gate):
    assert gate._is_target_owned("merlin/python/merlin/runtime/backends/muon_link.py")
    assert not gate._is_target_owned("merlin/python/merlin/targetgen/fixed_format/link.py")


# ------------------------------------------------------------------ the coupling scan
def test_a_vendor_symbol_is_caught_even_though_it_is_not_a_bare_identifier(gate, tmp_path):
    """The leak the whole-identifier check cannot see, and the reason the second scan exists."""
    src = tmp_path / "generic.py"
    src.write_text('def emit():\n    return "METRIC cycle_window_gemmini_region 1"\n', encoding="utf-8")
    hits = gate._scan_coupling(src)
    assert [(k, t) for _, t, k, _ in hits] == [("symbol", "gemmini")]


def test_importing_a_target_module_from_a_generic_one_is_caught(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text("from merlin.runtime.backends import muon_capsule_runner\n", encoding="utf-8")
    assert [t for _, t, k, _ in gate._scan_coupling(src) if k == "import"] == ["muon"]


def test_a_docstring_mention_is_not_coupling(gate, tmp_path):
    """Prose describing a target is documentation. Counting it would flood the register with noise
    and train people to ignore it, which costs more than the few real hits it might also surface."""
    src = tmp_path / "generic.py"
    src.write_text('"""Originally written for gemmini; now generic."""\nX = 1\n', encoding="utf-8")
    assert gate._scan_coupling(src) == []


def test_an_inline_marker_suppresses_a_deliberate_mention(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text(f'NAMES = ("gemmini",)  # {gate.INLINE_MARKER} the set this gate hunts\n', encoding="utf-8")
    assert gate._scan_coupling(src) == []


# ------------------------------------------------------------------ where the marker may sit
def test_the_marker_is_honoured_beside_the_mention_not_only_at_the_constants_start(gate, tmp_path):
    """Python reports an implicitly concatenated string at the line its FIRST fragment starts on.

    A long ``description=(...)`` can carry the target name a dozen lines below that, so anchoring the
    marker to the reported line put the annotation nowhere near what it explains — and someone who
    placed it correctly, beside the mention, watched the gate keep failing with no hint why.
    """
    src = tmp_path / "generic.py"
    src.write_text(
        "DESCRIPTION = (\n"
        '    "first fragment, no target here "\n'
        "    # target-ok: cites a pin, not a routing fact\n"
        '    "second fragment mentioning saturn "\n'
        ")\n",
        encoding="utf-8",
    )
    assert gate._scan_file(src) == []


def test_an_unmarked_multi_line_constant_is_still_caught(gate, tmp_path):
    """The span rule must widen where a marker is ACCEPTED, never what gets scanned."""
    src = tmp_path / "generic.py"
    src.write_text(
        'DESCRIPTION = (\n    "first fragment, no target here "\n    "second fragment mentioning saturn "\n)\n',
        encoding="utf-8",
    )
    assert [name for _, name, _ in gate._scan_file(src)] == ["saturn"]


# ------------------------------------------------------------------ the lift detector
def test_the_lift_detector_reports_the_live_tree_without_enforcing(gate):
    """A hit is a question ("is this general?"), not a violation — some answers are legitimately no.

    Asserting the CONTENTS here would be asserting today's debt; what must hold is that the detector
    runs over the real tree, returns reportable strings, and never fails the build.
    """
    assert gate.main(["--coupling"]) == 0
    for line in gate.lift_candidates():
        assert line.endswith("audit for a lift")


# ------------------------------------------------------------------ the token check
# Identifiers and the text a program emits, case-insensitively. The literal check above reads only
# lower-case whole-identifier string constants, which is how `SATURN_BENCHES`, `class SaturnBench` and
# "a bare-metal Saturn ELF" sat in the core with the gate green.
def _token_hits(gate, path, names=None):
    return [(kind, name) for _ln, name, kind, _snip in gate._scan_tokens(path, names)]


def test_the_token_check_catches_a_target_inside_an_identifier(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text("FOO_SATURN_X = 1\nclass SaturnBench:\n    pass\n", encoding="utf-8")
    assert _token_hits(gate, src) == [("identifier", "saturn"), ("identifier", "saturn")]


def test_the_token_check_reads_program_text_case_insensitively(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text(
        'MSG = "a bare-metal Saturn ELF"\nENV = "MERLIN_MUON_CONFIG"\nREF = f"from radiance-kernels {MSG}"\n',
        encoding="utf-8",
    )
    assert _token_hits(gate, src) == [("string", "saturn"), ("string", "muon"), ("string", "radiance")]


def test_the_token_check_leaves_comments_and_docstrings_to_review(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text(
        '"""Measured on saturn: 26055 cycles."""\n'
        "# the gemmini mesh drains here\n"
        "def f():\n"
        '    """Atlas example."""\n'
        '    "a bare prose statement about Radiance"\n'
        "    return 1\n",
        encoding="utf-8",
    )
    assert gate._scan_tokens(src) == []


def test_the_token_check_reports_the_most_specific_target(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text("MX_GEMMINI_ROUTE = 1\n", encoding="utf-8")
    assert _token_hits(gate, src) == [("identifier", "mx_gemmini")]


def test_the_token_check_honours_the_marker_on_the_tokens_own_line(gate, tmp_path):
    src = tmp_path / "generic.py"
    src.write_text(f'ENV = "MERLIN_MUON_CONFIG"  {gate.INLINE_MARKER} the variable a gate hunts\n', encoding="utf-8")
    assert gate._scan_tokens(src) == []


def test_a_word_that_merely_contains_a_target_is_not_one(gate, tmp_path):
    """Whole WORDS, not substrings: `irradiance` and `saturnine` name no target."""
    src = tmp_path / "generic.py"
    src.write_text('irradiance = "saturnine"\n', encoding="utf-8")
    assert gate._scan_tokens(src) == []


def test_a_path_owns_the_targets_its_words_name(gate):
    assert gate._token_owned("merlin/contract/external/gsim/model_build/gsim_gemmini_cmd_encode.py", "gemmini")
    assert gate._token_owned("src/merlin/targetgen/mx_gemmini_route.py", "mx_gemmini")
    assert not gate._token_owned("src/merlin/targetgen/mx_oracle.py", "mx_gemmini")
    assert not gate._token_owned("src/merlin/kernels/bench_ceiling.py", "saturn")


def test_the_token_check_uses_the_derived_roster(gate, tmp_path):
    """No new hand-kept list: a target declared only in a registry is caught in an identifier."""
    roster = importlib.util.spec_from_file_location("_target_roster_t", GATE.parent / "_target_roster.py")
    module = importlib.util.module_from_spec(roster)
    roster.loader.exec_module(module)
    (tmp_path / "examples" / "dev").mkdir(parents=True)
    (tmp_path / "examples" / "dev" / "experiment.yaml").write_text("target: blk_hw\n", encoding="utf-8")
    names = frozenset(module.target_names(tmp_path))
    src = tmp_path / "generic.py"
    src.write_text("FOO_BLK_HW_X = 1\n", encoding="utf-8")
    assert _token_hits(gate, src, names) == [("identifier", "blk_hw")]
    assert gate.TARGET_NAMES == frozenset(
        (module.target_names(repo_root()) | gate.RESIDUAL_NAMES) - gate.REFERENCE_DEFAULTS
    )


def test_mutating_a_clean_core_module_with_a_target_identifier_is_caught(gate, tmp_path):
    """MUTATION: the live module this gate was tightened for is clean; one planted identifier is not."""
    rel = "src/merlin/kernels/bench_ceiling.py"
    live = repo_root() / rel
    assert gate.token_hits(rel) == []
    mutant = tmp_path / "bench_ceiling.py"
    mutant.write_text(live.read_text(encoding="utf-8") + "\nFOO_SATURN_X = 1\n", encoding="utf-8")
    hits = gate._scan_tokens(mutant)
    assert [(kind, name, snip) for _ln, name, kind, snip in hits] == [("identifier", "saturn", "FOO_SATURN_X")]


def test_a_target_identifier_planted_in_a_clean_module_fails_the_gate(gate, tmp_path, monkeypatch, capsys):
    """MUTATION, at the VERDICT: the gate's own ``main`` passes the clean module and fails it once an
    identifier naming a target is planted -- the spelling the literal check let through."""
    rel = "src/merlin/kernels/bench_ceiling.py"
    clean = (repo_root() / rel).read_text(encoding="utf-8")
    module = tmp_path / rel
    module.parent.mkdir(parents=True)
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    monkeypatch.setattr(gate, "_iter_targets", lambda staged: [Path(rel)])
    module.write_text(clean, encoding="utf-8")
    assert gate.main([]) == 0
    module.write_text(clean + "\nFOO_SATURN_X = 1\n", encoding="utf-8")
    capsys.readouterr()
    assert gate.main([]) == 1
    assert "FOO_SATURN_X" in capsys.readouterr().out
