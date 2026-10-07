"""Exact complete-program roster checks for operator-private captures."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from merlin.compile.model_execution_inputs import file_sha256


def captured_programs(capture: Path, expected: Sequence[str]) -> list[tuple[str, Path]]:
    """Use every program in the attested root session, never an operator-picked slice."""
    # A complete single-network capture may also carry a version-1 execution
    # session contract (for example an image stream). Only a root session
    # receipt identifies the version-2 multi-program capture protocol.
    if tuple(expected) == ("model",) and not (capture / "session-receipt.json").exists():
        if (capture / "stages").exists():
            raise ValueError("single-network declaration received an unbound stage directory")
        if any(
            path.is_symlink() or not path.is_file()
            for path in (capture / "model.mlir", capture / "capture_receipt.json")
        ):
            raise ValueError("single-network declaration has no ordinary source model and receipt")
        contract_path = capture / "session_contract.yaml"
        if contract_path.exists():
            if contract_path.is_symlink() or not contract_path.is_file():
                raise ValueError("single-network execution contract is indirect or malformed")
            contract = yaml.safe_load(contract_path.read_bytes())
            if not isinstance(contract, Mapping) or contract.get("version") != 1:
                raise ValueError("single-network execution contract is not version 1")
        return [("model", capture)]
    contract_path = capture / "session_contract.yaml"
    receipt_path = capture / "session-receipt.json"
    if any(path.is_symlink() or not path.is_file() for path in (contract_path, receipt_path)):
        raise ValueError("complete multi-program capture has no ordinary root session contract/receipt")
    contract = yaml.safe_load(contract_path.read_bytes())
    receipt = json.loads(receipt_path.read_bytes())
    if not isinstance(contract, Mapping) or not isinstance(receipt, Mapping):
        raise ValueError("complete session contract/receipt is malformed")
    names = list(expected)
    programs = contract.get("programs")
    observed = receipt.get("programs")
    if (
        contract.get("version") != 2
        or contract.get("stages") != names
        or not isinstance(programs, list)
        or not isinstance(observed, list)
        or receipt.get("schema") != "merlin.model_session_capture.v1"
        or receipt.get("session_contract_sha256") != file_sha256(contract_path)
        or len(programs) != len(names)
        or len(observed) != len(names)
        or (len(names) > 1 and not contract.get("bindings"))
    ):
        raise ValueError("root session does not bind the declared complete program roster")
    stage_root = capture / "stages"
    if (
        stage_root.is_symlink()
        or not stage_root.is_dir()
        or sorted(p.name for p in stage_root.iterdir()) != sorted(names)
    ):
        raise ValueError("capture contains missing or extra session stage directories")
    result = []
    for name, entry, attested in zip(names, programs, observed, strict=True):
        stage = stage_root / name
        if (
            not isinstance(entry, Mapping)
            or entry.get("name") != name
            or entry.get("bundle") != f"stages/{name}"
            or not isinstance(attested, Mapping)
            or attested.get("name") != name
            or attested.get("ok") is not True
            or attested.get("opaque") != 0
            or attested.get("receipt_sha256") != file_sha256(stage / "capture_receipt.json")
            or stage.is_symlink()
            or not stage.is_dir()
        ):
            raise ValueError(f"session stage {name} is unverified, opaque, or not contract-bound")
        result.append((name, stage))
    return result
