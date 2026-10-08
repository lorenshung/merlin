"""Full grading catches independent value, post-argument, layout and alias defects."""

import copy

import numpy as np
import pytest

from merlin.targetgen.core_aten_semantics import validate_full


def fixture():
    values = [1.0, 2.0, 3.0, 4.0]
    tensor = dict(
        kind="tensor", dtype="float32", shape=[4], stride=[1], storage_offset=0, requires_grad=False, values=values
    )
    arguments = dict(args=dict(kind="tuple", items=[tensor]), kwargs={})
    output = {**tensor, "shape": [2, 2], "stride": [2, 1], "values": [[1.0, 2.0], [3.0, 4.0]]}
    record = dict(
        input_indices=[0],
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[2, 2])],
        semantic_boundary=dict(post_indices=[1], post_abi=[dict(dtype="f32", shape=[4])]),
        case=dict(
            arguments=arguments,
            post_arguments=copy.deepcopy(arguments),
            expected=output,
            comparison="torch_close",
            comparison_parameters={"rtol": 1e-5, "atol": 1e-6},
            mutated_arguments=[],
            output_input_aliases=["result->args[0]"],
        ),
    )
    raw = [np.array(values, np.float32).tobytes()] * 2
    readback = dict(
        metadata=[
            dict(shape=[2, 2], stride=[2, 1], storage_offset=0, requires_grad=False),
            dict(shape=[4], stride=[1], storage_offset=0, requires_grad=False),
        ],
        aliases=[[1], [1]],
        pre_bytes={"0": raw[1].hex()},
    )
    return record, raw, readback


def test_full_view_and_post_arguments():
    result = validate_full(*fixture())
    assert result["status"] == "pass"
    assert result["semantic_scope"] == "full"
    assert result["observation"]["output_input_aliases"] == ["result->args[0]"]


@pytest.mark.parametrize("defect", ["alias", "stride", "post_values", "output_values", "missing_snapshot"])
def test_independent_semantic_defects(defect):
    record, raw, readback = fixture()
    if defect == "alias":
        readback["aliases"][0] = []
    elif defect == "stride":
        readback["metadata"][0]["stride"] = [1, 2]
    elif defect == "post_values":
        raw[1] = np.zeros(4, np.float32).tobytes()
    elif defect == "output_values":
        raw[0] = np.zeros(4, np.float32).tobytes()
    else:
        readback["pre_bytes"] = {}
    assert validate_full(record, raw, readback)["status"] != "pass"


def test_mutation_is_measured_from_snapshot_not_expected_paths():
    record, raw, readback = fixture()
    record["case"]["mutated_arguments"] = ["args[0]"]
    assert validate_full(record, raw, readback)["status"] == "mismatch"
    readback["pre_bytes"]["0"] = np.zeros(4, np.float32).tobytes().hex()
    assert validate_full(record, raw, readback)["status"] == "pass"


def test_absent_boundary_cannot_claim_full_semantics():
    record, raw, readback = fixture()
    del record["semantic_boundary"]
    assert validate_full(record, raw, readback) is None
