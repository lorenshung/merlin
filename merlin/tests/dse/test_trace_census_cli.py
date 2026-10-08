"""``merlin experiment census``: the command a trace reader or a target provider hands its records to.

Each case runs the CLI on records written to disk and compares the printed document with the library
call on the same records, so the command can neither reshape a result nor swallow a refusal.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from merlin.perf import census_cli as CLI
from merlin.perf.address_locality import address_locality
from merlin.perf.execution_boundaries import (
    BoundaryInstruction,
    FunctionExtent,
    boundary_features,
    summarize_boundaries,
)

DIGESTS = {"program_sha256": "1" * 64, "census_sha256": "2" * 64, "provider_sha256": "3" * 64}


def _run(capsys, *argv: str) -> tuple[int, dict | None, str]:
    code = CLI.main(list(argv))
    captured = capsys.readouterr()
    return code, (json.loads(captured.out) if code == 0 else None), captured.err


def test_locality_prints_the_library_census(tmp_path, capsys):
    trace = [0, 64, 0, 128, 64, 0, 192]
    path = tmp_path / "addresses.json"
    path.write_text(json.dumps(trace))
    code, document, _ = _run(
        capsys, "locality", str(path), "--granule", "64", "--max-requests", "16", "--capacity", "1", "--capacity", "2"
    )
    expected = address_locality(trace, granule=64, max_requests=16, capacities=(1, 2))
    assert code == 0 and document["schema"] == "address_locality_v1"
    assert {k: v for k, v in document.items() if k != "schema"} == json.loads(json.dumps(asdict(expected)))
    assert document["first_touches"] == 4 and document["requests"] == len(trace)


@pytest.mark.parametrize(
    "payload, argv",
    [
        ([0, 64, 128], ("--granule", "64", "--max-requests", "2")),  # longer than the stated budget
        ([0, -64], ("--granule", "64", "--max-requests", "8")),  # an address the census cannot place
        ({"addresses": [0]}, ("--granule", "64", "--max-requests", "8")),  # not a list
    ],
)
def test_locality_refuses_rather_than_truncating(tmp_path, capsys, payload, argv):
    path = tmp_path / "addresses.json"
    path.write_text(json.dumps(payload))
    code, document, err = _run(capsys, "locality", str(path), *argv)
    assert code == CLI.REFUSED and document is None and err.startswith("refused:")


def _records(executions=(1, 1, 3, 3, 3, 1)):
    functions = [FunctionExtent("entry", 100, 108, 100, 16), FunctionExtent("child", 108, 112, 108, 8)]
    rows = [
        (100, "none", None, 0, 8),
        (102, "direct", "child", 0, 0),
        (104, "none", None, 8, 0),
        (106, "none", None, 0, 0),
        (108, "none", None, 4, 4),
        (110, "none", None, 4, 0),
    ]
    instructions = [
        BoundaryInstruction(address, 2, count, call, callee, read, write)
        for (address, call, callee, read, write), count in zip(rows, executions)
    ]
    return functions, instructions


def _write_records(path, functions, instructions):
    path.write_text(
        json.dumps(
            {**DIGESTS, "functions": [vars(f) for f in functions], "instructions": [vars(i) for i in instructions]}
        )
    )


def test_boundaries_print_the_library_summary_and_features(tmp_path, capsys):
    functions, instructions = _records()
    path = tmp_path / "records.json"
    _write_records(path, functions, instructions)
    code, document, _ = _run(capsys, "boundaries", str(path), "--entry", "entry")
    summary = summarize_boundaries(functions, instructions, **DIGESTS)
    assert code == 0 and document["summary"] == json.loads(json.dumps(summary))
    assert document["features"] == boundary_features(summary, entry_function="entry")


def test_a_derived_domain_admits_its_own_training_and_refuses_an_unknown(tmp_path, capsys):
    functions, instructions = _records()
    records = tmp_path / "records.json"
    _write_records(records, functions, instructions)
    _, document, _ = _run(capsys, "boundaries", str(records))
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(document["summary"]))
    domain_id = "4" * 64
    pointer = "/boundaries/direct_calls_per_entry"
    code, domain, _ = _run(
        capsys, "boundary-domain", str(summary), "--entry", "entry", "--pointer", pointer, "--domain-sha256", domain_id
    )
    assert code == 0 and domain["bounds"] == [[pointer, 1.0, 1.0]]
    domain_path = tmp_path / "domain.json"
    domain_path.write_text(json.dumps(domain))
    admit = ("--entry", "entry", "--admit", str(domain_path), "--domain-sha256", domain_id)
    _, admitted, _ = _run(capsys, "boundaries", str(records), *admit)
    assert admitted["admission"]["status"] == "supported_boundary_subdomain"
    assert admitted["admission"]["ranking_approved"] is False

    # an execution count the provider could not establish stays UNKNOWN, and so does the admission
    _write_records(records, *_records(executions=(1, None, 3, 3, 3, 1)))
    _, unknown, _ = _run(capsys, "boundaries", str(records), *admit)
    assert unknown["features"][pointer] is None and unknown["admission"]["status"] == "unknown"


def test_a_malformed_provider_record_is_refused(tmp_path, capsys):
    functions, instructions = _records()
    path = tmp_path / "records.json"
    document = {**DIGESTS, "functions": [vars(f) for f in functions], "instructions": [vars(i) for i in instructions]}
    del document["instructions"][0]["width_bytes"]
    path.write_text(json.dumps(document))
    code, _, err = _run(capsys, "boundaries", str(path))
    assert code == CLI.REFUSED and "declared fields" in err
    del document["census_sha256"]
    path.write_text(json.dumps(document))
    assert _run(capsys, "boundaries", str(path))[0] == CLI.REFUSED


def test_the_census_is_a_merlin_experiment_subcommand_with_no_script_of_its_own(tmp_path, capsys):
    """One experiment front door, and the locality census it serves reads the file ``inspect --trace``
    records (a JSON list of requested addresses in commit order)."""
    import tomllib

    from merlin.common.paths import repo_root

    cli = pytest.importorskip("merlin_experiments.cli")
    trace = [0x80001010, 0x80001010, 0x80002000, 0x80002004]
    path = tmp_path / "requested_addresses.json"
    path.write_text(json.dumps(trace) + "\n")
    assert cli.main(["census", "locality", str(path), "--granule", "16", "--max-requests", "4"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document == {
        "schema": "address_locality_v1",
        **json.loads(json.dumps(asdict(address_locality(trace, granule=16, max_requests=4)))),
    }
    scripts = tomllib.loads((repo_root() / "pyproject.toml").read_text())["project"]["scripts"]
    assert not [name for name, entry in scripts.items() if entry.startswith("merlin.perf.census_cli")]
