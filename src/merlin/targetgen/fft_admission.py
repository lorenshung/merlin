"""Complete-operation screen for pre-PT2E constant-weight FFT candidates."""

from __future__ import annotations

from pathlib import Path

from merlin.targetgen.core_aten_spec_int8 import QUANTIZE, declaration_location, host_signature
from merlin.targetgen.software_spec import admit_operation


def screen_fft_component_dtype(dtype, rank, selection):
    """Bind unsupported component precision to the authored quantizer signature."""
    import yaml

    if dtype in {"torch.float32", "torch.complex64"}:
        return []
    component = {"torch.float64": "f64", "torch.complex128": "f64"}.get(dtype, dtype)
    signature = host_signature(QUANTIZE, rank, {"zero_point": 0, "quant_min": -128, "quant_max": 127})
    signature["operand_dtype"] = signature["ordered_operand_dtypes"][0] = component
    reasons = []
    for authority, document in (("software", selection["spec"]), ("host", selection["host"])):
        decision = admit_operation(document, QUANTIZE, signature, "host")
        if decision["status"] == "admitted":
            continue
        if not decision.get("declaration"):
            identity = selection["identities"]["software_spec" if authority == "software" else "host_capabilities"]
            root = yaml.compose(Path(identity["path"]).read_text())
            key = next(k for k, _ in root.value if k.value == "operations")
            reasons.append(
                f"FFT component {component} quantization {authority}: {identity['path']}:"
                f"{key.start_mark.line + 1}: {decision['reason']}"
            )
            continue
        location = declaration_location(selection, authority, decision["declaration"])
        root = yaml.compose(Path(location["path"]).read_text())
        operations = next(v for k, v in root.value if k.value == "operations")
        if authority == "software":
            declaration = next(v for k, v in operations.value if k.value == decision["declaration"])
            precision = next(v for k, v in declaration.value if k.value == "ordered_operand_dtypes")
            line = precision.start_mark.line + 1
        else:
            line = location["restriction_line"]
        reasons.append(
            f"FFT component {component} quantization {authority}: {location['path']}:{line}: {decision['reason']}"
        )
    return reasons


def screen_fft_candidates(candidates, selection):
    """Screen each prospective chain and every auxiliary ATen operation.

    A missing declaration is bound to the closed operations roster, rather
    than inventing a source location for an operation that does not exist.
    This is an admission diagnostic, never an emitted-code qualification.
    """
    import yaml

    def dtype(value):
        return {
            "torch.float32": "f32",
            "torch.float64": "f64",
            "torch.complex64": "complex<f32>",
            "torch.complex128": "complex<f64>",
        }.get(value, value)

    sites = []
    for candidate in candidates:
        steps, reasons = [], []

        def screen(name, signature, placement="host", role="auxiliary"):
            row = {"op": name, "role": role, "signature": signature, "placement": placement}
            for authority, document in (("software", selection["spec"]), ("host", selection["host"])):
                if placement == "accelerator" and authority == "host":
                    continue
                decision = admit_operation(document, name, signature, placement)
                if decision.get("declaration"):
                    location = declaration_location(selection, authority, decision["declaration"])
                else:
                    identity = selection["identities"][
                        "software_spec" if authority == "software" else "host_capabilities"
                    ]
                    root = yaml.compose(Path(identity["path"]).read_text())
                    key, _ = next((k, v) for k, v in root.value if k.value == "operations")
                    location = {
                        **identity,
                        "line": key.start_mark.line + 1,
                        "restriction_line": key.start_mark.line + 1,
                        "scope": "closed operation roster; requested op has no declaration",
                    }
                row[authority] = {**decision, "source": location}
                if decision["status"] != "admitted":
                    reasons.append(
                        f"{name} rank {signature['rank']} {authority}: "
                        f"{location['path']}:{location['restriction_line']}: {decision['reason']}"
                    )
            steps.append(row)

        for operation in candidate["operations"]:
            rank = operation["rank"]
            if operation["op"] in ("aten.matmul.default", "aten.mm.default"):
                constants = operation.get("constant_operands", [False] * len(operation["operand_ranks"]))
                for operand_rank, constant in zip(operation["operand_ranks"], constants, strict=True):
                    if constant:
                        continue  # Existing recipe freezes weight codes; no runtime quantizer is emitted.
                    screen(
                        QUANTIZE,
                        host_signature(QUANTIZE, operand_rank, {"zero_point": 0, "quant_min": -128, "quant_max": 127}),
                        role="quantize",
                    )
                from merlin.targetgen.transfer_contracts import screen_transfer_contract

                for key, source, destination, transfer_dtype in (
                    ("operand_load", "host", "accelerator", "int8"),
                    ("accumulator_readout", "accelerator", "host", "i32"),
                ):
                    decision = screen_transfer_contract(
                        selection["spec"],
                        source_placement=source,
                        destination_placement=destination,
                        operand_dtype=transfer_dtype,
                        result_dtype=transfer_dtype,
                        operand_layout="row_major_contiguous",
                        result_layout="row_major_contiguous",
                    )
                    identity = selection["identities"]["software_spec"]
                    root = yaml.compose(Path(identity["path"]).read_text())
                    transfers = next(v for k, v in root.value if k.value == "transfer_contracts")
                    declaration = next(v for k, v in transfers.value if k.value == key)
                    location = {**identity, "line": declaration.start_mark.line + 1}
                    steps.append(
                        {"op": f"transfer_contracts.{key}", "role": key, "software": {**decision, "source": location}}
                    )
                    if decision["status"] != "admitted" or key not in decision["matching_declarations"]:
                        reasons.append(f"{location['path']}:{location['line']}: {key}: {decision['reason']}")
                screen(
                    "contraction",
                    {
                        "family": "contraction",
                        "ordered_operand_dtypes": ["i8", "i8", "i32"],
                        "ordered_result_dtypes": ["i32"],
                        "operand_dtype": "i8",
                        "accum_dtype": "i32",
                        "readout_dtype": "i32",
                        "rank": rank,
                        "layout": "row_major_contiguous",
                        "tails": "zero_pad_valid_window",
                        "broadcasting": "none",
                        "aliasing": "disjoint_inputs_outputs",
                    },
                    "accelerator",
                    "integer contraction",
                )
                screen(
                    "aten._to_copy.default",
                    {
                        "family": "elementwise_map",
                        "rank": rank,
                        "operand_dtype": "i32",
                        "ordered_operand_dtypes": ["i32", "f32"],
                        "ordered_result_dtypes": ["f32"],
                    },
                    role="readout cast",
                )
                screen(
                    "aten.mul.Tensor",
                    {
                        "family": "elementwise_map",
                        "rank": rank,
                        "operand_dtype": "f32",
                        "ordered_operand_dtypes": ["f32", "f32", "f32"],
                        "ordered_result_dtypes": ["f32"],
                    },
                    role="readout scale",
                )
            else:
                operands = [dtype(value) for value in operation["operand_dtypes"]]
                result = dtype(operation["result_dtype"])
                screen(
                    operation["op"],
                    {
                        "rank": rank,
                        "ordered_operand_dtypes": [*operands, result],
                        "ordered_result_dtypes": [result],
                        "operand_dtype": operands[0] if operands else result,
                    },
                )
        sites.append(
            {
                "source_node": candidate["source_node"],
                "source_op": candidate["source_op"],
                "steps": steps,
                "reasons": reasons,
                "status": "refused" if reasons else "admitted",
                "scope": "prospective candidate; source graph unchanged; no operations emitted",
            }
        )
    return sites
