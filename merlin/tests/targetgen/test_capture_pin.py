"""Which model2MLIR made a capture: verified against its software pin, recorded, and enforced.

The integer softmax/GELU/layer norm lower exactly only through the integer shift and floor-division
decompositions a specific model2MLIR revision carries. A capture made from any other checkout -- another
commit, or the right commit with the decomposition files edited -- must say so, and a capture that asks
for the integer nonlinears must refuse to run on it at all.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from merlin.common import provenance as P
from merlin.integrations import model2mlir as M
from merlin.targetgen import capsule_source as CS


def test_the_capture_pin_is_a_full_sha_over_the_files_the_capture_reads():
    pin = P.load_pins(P.software_pins_path())[M.CAPTURE_PIN]
    assert len(pin.commit) == 40
    assert "m2m/ir/decompositions.py" in pin.requires_paths
    # Content, not HEAD: an uncommitted edit to a decomposition file must read as off-pin.
    assert pin.checks_content


def _off_pin_checkout(tmp_path: Path) -> Path:
    """A git checkout that is a model2MLIR in shape (``m2m/__init__.py``) and not the pinned revision."""
    repo = tmp_path / "model2MLIR"
    (repo / "m2m" / "ir").mkdir(parents=True)
    (repo / "m2m" / "__init__.py").write_text("")
    (repo / "m2m" / "ir" / "decompositions.py").write_text("# not the pinned decompositions\n")
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "off pin"], check=True, stdin=subprocess.DEVNULL)
    return repo


def test_an_off_pin_checkout_is_recorded_as_such(tmp_path):
    record = M.capture_pin(_off_pin_checkout(tmp_path))
    assert record["ok"] is False
    assert record["pin"] == M.CAPTURE_PIN
    assert any("pin declares" in d for d in record["verification"]["drift"])


def test_an_integer_nonlinear_capture_refuses_an_off_pin_checkout(tmp_path):
    source = CS.PytorchRefSource(m2m_dir=_off_pin_checkout(tmp_path), python=Path(sys.executable))
    loader = tmp_path / "loader.py"
    loader.write_text("def get_model_and_inputs():\n    raise AssertionError('the worker must not run')\n")
    with pytest.raises(CS.M2MUnavailable, match="software pin"):
        source.capture_loader(
            loader, "i8", workdir=tmp_path / "w", activation_contractions=True, integer_nonlinear=True
        )
