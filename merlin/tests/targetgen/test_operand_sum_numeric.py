from merlin.targetgen.operand_sum_numeric import audit_i8_operand_sum

_FACET = {
    "operand_sum": {
        "operands": 2,
        "operand_dtype": "i8",
        "operand_rounding": "half_even",
        "operand_saturates": True,
        "scale_dtype": "f32",
    },
    "readouts": [{"selector": "i8", "applies": ["acc_scale", "relu"]}],
    "scale": {"dtype": "f32", "granularities": ["tensor"]},
}


def test_full_i8_domain_accepts_the_selected_gain_but_refuses_a_larger_gain():
    selected = audit_i8_operand_sum(
        lhs_scale=1.0626662443588681,
        rhs_scale=0.31430679816210955,
        bound_lsb=2,
        relu=True,
        facet=_FACET,
    )
    assert selected["status"] == "within_bound"
    assert selected["pairs_checked"] == 65536 and selected["max_error_lsb"] == 1
    assert selected["n_over_bound"] == 0 and len(selected["witness_sha256"]) == 64

    refused = audit_i8_operand_sum(lhs_scale=10, rhs_scale=5.1, bound_lsb=2, relu=True, facet=_FACET)
    assert refused["status"] == "exceeds_bound"
    assert refused["n_over_bound"] > 0 and refused["max_error_lsb"] == 5


def test_missing_numeric_fact_cannot_be_treated_as_success():
    assert audit_i8_operand_sum(lhs_scale=1, rhs_scale=1, bound_lsb=2, relu=True, facet={})["status"] == "unknown"
