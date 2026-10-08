"""A capture whose dtype token alone decides its precision refuses a token it does not know.

WHY THIS EXISTS. ``_m2m_capture_worker`` looked the ``--dtype`` token up in its scheme table with a
``(None, None)`` default, so a token missing from the table -- ``fp8_e5m2`` is the one that was hit --
selected no quantization and no cast and captured the fp32 model under the requested name. The argument
check runs before the worker imports model2MLIR or torch, so it is exercised here without either.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import _m2m_capture_worker as W


def _argv(dtype, *extra):
    return ["--loader", "loader.py", "--out", "out", "--dtype", dtype, *extra]


def test_an_unknown_dtype_is_refused_before_capture(capsys):
    with pytest.raises(SystemExit) as exc:
        W.main(_argv("fp8_e5m2"))
    assert exc.value.code == 2
    assert "'fp8_e5m2' has no capture scheme" in capsys.readouterr().err


@pytest.mark.parametrize("extra", [("--scheme", "float8_weight_only_e4m3"), ("--recipe", "recipe.json")])
def test_an_explicit_scheme_or_recipe_still_owns_the_decision(extra, monkeypatch):
    # Past the argument check the worker imports model2MLIR; stopping there proves the check let it by.
    monkeypatch.setitem(__import__("sys").modules, "m2m", None)
    with pytest.raises(ImportError):
        W.main(_argv("fp8_e5m2", *extra))
