"""Retain exact self-NaN classification after proved component arithmetic.

IEEE ord(x,x) and uno(x,x) only classify the source word; they do not perform
floating arithmetic or choose a NaN payload. Keep the original predicates and
SSA dependencies, including their selected readout masks. This permission needs
an existing exact data-arithmetic witness; conversion alone cannot obtain it.
Other source comparisons and math remain subject to complete coverage checks.
"""


def qualify_nan_readout(module, arithmetic_proofs):
    from xdsl.dialects.builtin import UnitAttr

    if not any(proof.get("source_data_arithmetic") is True for proof in arithmetic_proofs):
        return []
    qualified = []
    for ordinal, op in enumerate(module.walk()):
        if op.name != "arith.cmpf" or op.operands[0] is not op.operands[1] or op.predicate.value.data not in (7, 14):
            continue
        op.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        qualified.append(
            dict(
                source_operation_ordinal=ordinal,
                predicate="ord" if op.predicate.value.data == 7 else "uno",
                proof="identical_source_word_nan_classification_without_arithmetic_or_payload_selection",
                dependencies="original_source_SSA_and_readout_selection_retained",
                prerequisite="exact_component_source_data_arithmetic",
            )
        )
    return qualified
