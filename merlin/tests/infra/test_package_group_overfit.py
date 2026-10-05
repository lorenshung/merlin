"""The guardrail against a submitted package keying behavior on a whole-model group id, or carrying
a hardcoded per-model shape table (see the phase-2 whole-model plan's guardrail #5).

A candidate that special-cases "group 70 of THIS captured model" passes today's measurement and
answers nothing tomorrow -- a different model, or the same model recaptured with different group
numbers, gets none of the win. Tested against SYNTHETIC package files, never the live tree: a test
that asserts today's package is clean turns into a chore that gets bumped when it changes, and it
would pass just as well if the scan stopped working.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from merlin.common.paths import repo_root

GATE = repo_root() / "build_tools" / "scripts" / "check_package_group_overfit.py"


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("_check_package_group_overfit", GATE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- group-id keying


def test_a_comparison_against_a_group_named_value_is_refused(tmp_path, gate):
    _write(
        tmp_path,
        "backend.py",
        "def lower(group_id, x):\n    if group_id == 70:\n        return special(x)\n    return generic(x)\n",
    )
    violations = gate.scan_package(tmp_path)
    assert [v["kind"] for v in violations] == ["group_id_keying"]
    assert violations[0]["line"] == 2 and "70" in violations[0]["why"]


def test_an_int_keyed_table_is_refused_only_when_named_for_a_group(tmp_path, gate):
    _write(
        tmp_path,
        "overrides.py",
        'GROUP_OVERRIDES = {1: "conv_stem", 70: "mean", 71: "final"}\n\n'
        "def lower(group, x):\n    return GROUP_OVERRIDES.get(group)\n",
    )
    violations = gate.scan_package(tmp_path)
    assert [v["kind"] for v in violations] == ["group_id_keying"]
    assert violations[0]["line"] == 1


def test_an_int_keyed_table_named_for_something_else_passes(tmp_path, gate):
    """The exact same SHAPE of literal (several int keys) is fine when nothing nearby names a group --
    an opcode table, a bit-width table, a selector-to-role map are all legitimate."""
    _write(
        tmp_path,
        "opcodes.py",
        'OPCODE_BY_SELECTOR = {1: "MVIN", 5: "PRELOAD", 6: "COMPUTE"}\n\n'
        "def decode(selector):\n    return OPCODE_BY_SELECTOR.get(selector)\n",
    )
    assert gate.scan_package(tmp_path) == []


def test_a_match_case_against_a_group_named_subject_is_refused(tmp_path, gate):
    _write(
        tmp_path,
        "dispatch.py",
        "def lower(group_index, x):\n    match group_index:\n        case 70:\n            return special(x)\n"
        "    return generic(x)\n",
    )
    violations = gate.scan_package(tmp_path)
    assert [v["kind"] for v in violations] == ["group_id_keying"]


def test_a_comparison_between_two_plain_variables_passes(tmp_path, gate):
    _write(
        tmp_path,
        "generic.py",
        "def lower(shape, other):\n    if shape == other:\n        return True\n    return False\n",
    )
    assert gate.scan_package(tmp_path) == []


# --------------------------------------------------------------------------- hardcoded shape tables


def test_a_hardcoded_per_layer_shape_table_is_refused(tmp_path, gate):
    _write(
        tmp_path,
        "shapes.py",
        "LAYER_SHAPES = [\n"
        "    [1, 64, 112, 112],\n"
        "    [1, 256, 56, 56],\n"
        "    [1, 512, 28, 28],\n"
        "    [1, 1024, 14, 14],\n"
        "    [1, 2048, 7, 7],\n"
        "]\n",
    )
    violations = gate.scan_package(tmp_path)
    assert [v["kind"] for v in violations] == ["hardcoded_shape_table"]
    assert "5 literal" in violations[0]["why"]


def test_a_short_list_of_shapes_below_the_table_threshold_passes(tmp_path, gate):
    _write(tmp_path, "small.py", "PADDING = [(1, 1, 1), (0, 0, 0)]\n")
    assert gate.scan_package(tmp_path) == []


def test_mixed_length_tuples_are_not_counted_as_one_table(tmp_path, gate):
    """A table needs SEVERAL entries of the SAME shape length; four values that happen to be tuples of
    different lengths are not "a shape table", and this must not conflate them into one."""
    _write(
        tmp_path,
        "mixed.py",
        "ROWS = [\n    (1, 2, 3),\n    (1, 2, 3, 4),\n    (5, 6, 7, 8),\n    (9, 10, 11, 12),\n]\n",
    )
    assert gate.scan_package(tmp_path) == []


# --------------------------------------------------------------------------- exemptions and plumbing


def test_devtools_is_exempt_but_the_rest_of_the_package_is_not(tmp_path, gate):
    (tmp_path / "devtools").mkdir()
    _write(tmp_path / "devtools", "probe.py", "def check(group_id):\n    return group_id == 70\n")
    _write(tmp_path, "backend.py", "def lower(group_id, x):\n    return group_id == 70\n")
    violations = gate.scan_package(tmp_path)
    assert [v["path"] for v in violations] == [str(tmp_path / "backend.py")]


def test_an_inline_marker_excuses_one_line(tmp_path, gate):
    _write(
        tmp_path,
        "backend.py",
        "def lower(group_id, x):\n    if group_id == 70:  # overfit-ok: a documented, reviewed exception\n"
        "        return special(x)\n    return generic(x)\n",
    )
    assert gate.scan_package(tmp_path) == []


def test_the_cli_refuses_with_a_named_line_and_passes_a_clean_package(tmp_path, gate, capsys):
    _write(tmp_path, "backend.py", "def lower(group_id, x):\n    return group_id == 70\n")
    assert gate.main(["--package", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "backend.py:2" in out and "group_id_keying" in out

    clean = tmp_path / "clean"
    clean.mkdir()
    _write(clean, "backend.py", "def lower(entry, x):\n    return entry.get('op') == 'matmul'\n")
    assert gate.main(["--package", str(clean)]) == 0
    assert "[  ok]" in capsys.readouterr().out
