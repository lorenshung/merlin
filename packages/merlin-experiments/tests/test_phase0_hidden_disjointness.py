"""A hidden capsule that repeats a public program is caught by the points, not by names."""

from __future__ import annotations

import yaml
from merlin_experiments.phase0.hidden_disjointness import capsule_point, check


def _capsule(name, lhs, shape, epilogue=()):
    return {
        "name": name,
        "inputs": [
            {"name": lhs, "role": "input", "shape": shape, "dtype": "i8"},
            {"name": "W" + lhs, "role": "weight", "shape": [shape[1], 16], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {"lhs": lhs, "weight": "W" + lhs, "out": "Y0", "epilogue": list(epilogue)},
        },
        "expected": {"modes": {}},
        "numeric_policy": {"compare": "exact_int", "dtype": "i32"},
    }


def _write(root, category, capsule):
    directory = root / category / capsule["name"]
    directory.mkdir(parents=True)
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule))
    return directory


def test_a_renamed_public_program_is_the_same_point():
    assert capsule_point(_capsule("a", "A0", [16, 32])) == capsule_point(_capsule("b", "Ah5", [16, 32]))
    assert capsule_point(_capsule("a", "A0", [16, 32])) != capsule_point(_capsule("a", "A0", [16, 48]))
    assert capsule_point(_capsule("a", "A0", [16, 32])) != capsule_point(_capsule("a", "A0", [16, 32], ["relu"]))


def test_the_run_check_counts_overlap_without_naming_it(tmp_path):
    written = [
        _write(tmp_path, "isa", _capsule("pub", "A0", [16, 32])),
        _write(tmp_path, "hidden", _capsule("held", "Ah", [31, 48])),
    ]
    report = check(written)
    assert report["status"] == "disjoint" and (report["hidden_capsules"], report["public_capsules"]) == (1, 1)
    written.append(_write(tmp_path, "hidden", _capsule("renamed", "Ax", [16, 32])))
    report = check(written)
    assert report["status"] == "overlap" and report["overlapping_hidden_capsules"] == 1
    assert "renamed" not in str(report) and "16" not in str(report)
