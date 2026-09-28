"""Coarse checks for explicit capability observations and held-out boundaries."""

import tempfile
import unittest
from pathlib import Path

import yaml

from merlin.common.paths import resolve_grant
from merlin.targetgen.application_inventory import (
    application_demand_inventory,
    application_operation_inventory,
)
from merlin.targetgen.target_registry import TargetInfo, observed_contract


class ExplicitInputSelectionTests(unittest.TestCase):
    def test_grants_resolve_within_the_explicit_input_owner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "merlin/experiments/isa.h"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("legacy input")
            self.assertEqual(resolve_grant("experiments/isa.h", root=root), legacy)
            direct = root / "experiments/isa.h"
            direct.parent.mkdir()
            direct.write_text("canonical input")
            self.assertEqual(resolve_grant("experiments/isa.h", root=root), direct)

    def test_scoped_capabilities_do_not_change_provider_plugins_or_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contract = root / "contract.yaml"
            declared = {"name": "device", "compute_units": [], "plugin": {"backend": "owned:backend"}}
            contract.write_text(yaml.safe_dump(declared))
            info = TargetInfo(
                "device", "external", root, contract, root / "dialect.yaml", root / "facts.json", "simulator", root
            )
            observation = {"compute_units": [{"kind": "systolic"}], "plugin": {"backend": "forged:backend"}}
            with observed_contract("device", observation):
                observation["compute_units"].clear()
                loaded = info.load_contract()
                self.assertEqual(loaded["compute_units"], [{"kind": "systolic"}])
                loaded["compute_units"].clear()
                self.assertEqual(info.plugin()["backend"], "owned:backend")
                self.assertEqual(info.plugin()["path"], str(root))
                with observed_contract("device", {"compute_units": [{"kind": "vector"}]}):
                    self.assertEqual(info.load_contract()["compute_units"], [{"kind": "vector"}])
                    self.assertEqual(info.plugin()["backend"], "owned:backend")
                self.assertEqual(info.load_contract()["compute_units"], [{"kind": "systolic"}])
            self.assertEqual(info.load_contract(), declared)

    def test_validation_census_does_not_admit_validation_as_derivation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.mlir"
            path.write_text(
                "builtin.module { func.func @main(%a: f32, %b: f32) -> f32 { "
                "%x = arith.addf %a, %b : f32 func.return %x : f32 } }"
            )
            contract = {"compute_units": []}
            observed = application_operation_inventory(
                path, "device", capability_contract=contract, workload_id="resnet50"
            )
            self.assertEqual(observed["workload_identity"]["workload_role"], "validation")
            self.assertIn("evaluation_only", observed["purpose"])
            with self.assertRaises(ValueError):
                application_demand_inventory(
                    {"alias": path},
                    "device",
                    detailed=True,
                    capability_contract=contract,
                    application_metadata={"alias": observed["workload_identity"]},
                )


if __name__ == "__main__":
    unittest.main()
