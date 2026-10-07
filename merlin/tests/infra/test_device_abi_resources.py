"""Device ABI lookup uses selected/bundled contracts, not the invocation CWD."""

import pytest

from merlin.common import paths
from merlin.llvmlower.device_shim import kernel_abi_for
from merlin.targetgen.contract import schemas


def test_resident_interface_uses_bundled_or_selected_contract_bytes(tmp_path, monkeypatch):
    from merlin.targetgen.contract.interface_emit import emit_interface_mlir
    from merlin.targetgen.contract.resident_interface_abi import bind_single_resident_matmul

    interface = emit_interface_mlir(
        {
            "abi_version": "0.1",
            "target": "test_device",
            "tensors": {
                "weights": {"shape": [19, 8], "dtype": "i8", "role": "weight"},
                "activation": {"shape": [4, 19], "dtype": "i8", "role": "input"},
            },
            "commands": [
                {
                    "opcode": "RES_PACK",
                    "operands": {"src": "weights", "dst": "resident"},
                    "attributes": {"layout": "packed_rhs"},
                },
                {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "activation", "rhs": "resident", "dst": "sum"}},
                {
                    "opcode": "COMMIT",
                    "operands": {"src": "sum", "dst": "result"},
                    "attributes": {"output_dtype": "i32", "epilogue": []},
                },
            ],
        }
    )
    monkeypatch.delenv("MERLIN_CONTRACT_DIR", raising=False)
    original = (schemas.contract_dir() / "mlir_oot_backend_contract.yaml").read_bytes()
    default = bind_single_resident_matmul(interface, target="test_device")
    assert default.kernel_symbol == "test_device_kernel"

    selected = original.replace(b'symbol: "{target}_kernel"', b'symbol: "selected_{target}_kernel"')
    assert selected != original
    contract = tmp_path / "mlir_oot_backend_contract.yaml"
    contract.write_bytes(selected)
    monkeypatch.setenv("MERLIN_CONTRACT_DIR", str(tmp_path))
    observed = bind_single_resident_matmul(interface, target="test_device")
    assert observed.kernel_symbol == "selected_test_device_kernel"
    assert observed.contract_sha256 != default.contract_sha256
    assert bind_single_resident_matmul(interface, target="test_device", abi_contract=original) == default

    contract.unlink()
    with pytest.raises(FileNotFoundError):
        bind_single_resident_matmul(interface, target="test_device")


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
