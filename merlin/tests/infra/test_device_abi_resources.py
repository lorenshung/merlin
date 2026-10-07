"""Device ABI lookup uses selected/bundled contracts, not the invocation CWD."""

import pytest

from merlin.common import paths
from merlin.llvmlower.device_shim import kernel_abi_for
from merlin.targetgen.contract import schemas


def test_device_abi_uses_shared_resource_resolution_outside_checkout(tmp_path, monkeypatch):
    selected = schemas.contract_dir()
    expected = schemas.render_backend_contract("test_device")["kernel_abi"]
    monkeypatch.delenv("MERLIN_CONTRACT_DIR", raising=False)
    monkeypatch.setattr(paths, "merlin_dir", lambda: tmp_path / "absent_checkout")
    monkeypatch.setattr(paths, "data_path", lambda *parts: selected)
    monkeypatch.chdir(tmp_path)
    actual = kernel_abi_for("test_device")
    assert actual is not None
    assert actual.symbol == expected["symbol"]
    assert actual.arg_order == expected["arg_order"]
    assert actual.pointee_layout == expected["pointee_layout"]


@pytest.mark.parametrize("contents", ["kernel_abi: {}\n", "not: [valid yaml\n"])
def test_device_abi_does_not_fallback_from_unreadable_selected_contract(tmp_path, monkeypatch, contents):
    (tmp_path / "mlir_oot_backend_contract.yaml").write_text(contents, encoding="utf-8")
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(tmp_path))
    assert kernel_abi_for("test_device") is None
