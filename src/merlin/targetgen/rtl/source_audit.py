"""Compare authored direct RTL audit questions with actual extracted artifacts."""

import argparse
import json
from pathlib import Path

import yaml

from .circt_introspect import _module_port_sig
from .source_selection import digest, load_selection


def build_source_audit(selection: dict, facts: dict, declaration: dict) -> dict:
    """Audit exact source slices, preserving disagreements instead of writing facts.

    The expected values are human-reviewed audit inputs. Observed source slices
    and extracted values are generated outputs; neither supplies hardware support
    or numerical conformance merely by agreeing.
    """
    checks = declaration.get("audit_checks") or []
    results, seen = [], set()
    text = Path(selection["sources"]["core_hw"]["path"]).read_text()
    for check in checks:
        identity = check["id"]
        if identity in seen:
            raise ValueError("duplicate RTL audit check")
        seen.add(identity)
        if check["kind"] != "hw_port_type":
            raise ValueError("unsupported direct RTL audit question")
        module, port = check["module"], check["port"]
        signature = _module_port_sig(text, module)
        observed, snippet = None, None
        if signature:
            for member in signature.split(","):
                lhs, separator, typ = member.strip().rpartition(" : ")
                if separator and lhs.split()[-1].lstrip("%") == port:
                    observed, snippet = typ, member.strip()
        projection = check.get("extraction") or {}
        records = (facts.get("facts") or {}).get(projection.get("collection")) or []
        rows = [row for row in records if row.get("name") == projection.get("name")]
        extracted = rows[0].get(projection.get("field")) if len(rows) == 1 else None
        if extracted is None and projection.get("field") == "elem_bits" and len(rows) == 1:
            from merlin.common.quant_formats import get as quant_format

            try:
                extracted = quant_format(rows[0].get("dtype")).element_bits
            except (KeyError, ValueError, TypeError):
                pass
        source_status = (
            "verified" if observed == check.get("expected_type") else "unknown" if observed is None else "mismatch"
        )
        comparable = check.get("comparison", "same_quantity") == "same_quantity"
        bits = int(observed[1:]) if observed and observed.startswith("i") and observed[1:].isdigit() else None
        extraction_status = (
            "different_quantity"
            if not comparable
            else "unknown"
            if bits is None or extracted is None
            else "agrees"
            if bits == extracted
            else "mismatch"
        )
        results.append(
            {
                "id": identity,
                "source_status": source_status,
                "source": {
                    **selection["sources"]["core_hw"],
                    "module": module,
                    "port": port,
                    "observed_type": observed,
                    "snippet": snippet,
                },
                "authored_expected_type": check.get("expected_type"),
                "extraction": {**projection, "value": extracted, "status": extraction_status},
                "gap": check.get("gap"),
                "qualification": check.get("qualification"),
            }
        )
    return {
        "schema": "merlin.rtl_source_audit.v1",
        "target": selection["target"],
        "source_selection_sha256": selection.get("selection_sha256"),
        "status": "verified" if results and all(row["source_status"] == "verified" for row in results) else "unknown",
        "checks": results,
        "historical_hierarchy": selection.get("diagnostics"),
        "qualification": "manual audit checked against exact current RTL; not operation/numerical certification",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-bundle", "facts", "hardware-spec", "output"):
        parser.add_argument(f"--{name}", required=True)
    args = parser.parse_args(argv)
    declaration = yaml.safe_load(Path(args.hardware_spec).read_bytes())
    selected = load_selection(args.source_bundle, target=declaration["target"])
    facts = json.loads(Path(args.facts).read_bytes())
    if facts.get("inputs", {}).get("source_bundle_sha256") != selected["selection_sha256"]:
        raise ValueError("audit facts do not bind selected source bundle")
    report = build_source_audit(selected, facts, declaration)
    report["facts_sha256"] = digest(args.facts)
    report["hardware_spec_sha256"] = digest(args.hardware_spec)
    output = Path(args.output)
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
