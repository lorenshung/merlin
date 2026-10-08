"""Independent normal generation at declared aggregate capacity boundaries.

Synthetic facts exercise the derivation and real writer/goldens/screens. They
grant neither RTL admission nor complete compiler or scenario coverage.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import generation, resource_boundaries
from merlin_experiments.phase0.resource_boundaries import BoundaryUnavailable, derive
from merlin_experiments.phase0.sweeps import resolve_extent
from test_component_generation import write

from merlin.targetgen import golden_store as GS

pytest_plugins = ["test_component_generation"]


def boundary_spec():
    return {
        "derive": "resident_allocation_boundary",
        "store": "scratch",
        "capacity_fact": ["memories", 0, "bytes"],
        "reservation_facts": [["reservations", "command_bytes"]],
        "quantum": "tile",
        "tail_offsets": [-1, 1],
        "allocations": [
            {"name": "lhs", "shape": ["M", "K"], "dtype": "operand", "round_up": [1]},
            {"name": "rhs", "shape": ["K", "N"], "dtype": "operand"},
            {"name": "output", "shape": ["M", "N"], "dtype": "i32"},
        ],
    }


def declare(options, tile, capacity_rows, *, reserved_rows=2):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
    contract["capabilities"]["mesh"] = {"rows": tile, "cols": tile}
    contract["compute_units"][0]["accumulate"] = [{"in": "int8", "weight": "int8", "acc": "i32"}]
    write(options["capability_contract"], contract)
    raw = {
        "facts": {
            "arrays": [{"name": "mesh", "rows": tile, "cols": tile}],
            "memories": [
                {
                    "name": "scratch",
                    "bytes": capacity_rows * tile,
                    "depth": capacity_rows,
                    "row_elems": tile,
                    "elem_bits": 8,
                }
            ],
            "reservations": {"command_bytes": reserved_rows * tile},
        }
    }
    write(options["rtl_facts"], raw)
    software = yaml.safe_load(options["software_spec"].read_bytes())
    software["component_performance"]["hardware"] = {
        "contract_sha256": hashlib.sha256((json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()).hexdigest(),
        "raw_facts_sha256": hashlib.sha256(options["rtl_facts"].read_bytes()).hexdigest(),
    }
    software["component_performance"]["objectives"] = []
    write(options["software_spec"], software)
    template = yaml.safe_load(options["performance_template"].read_bytes())
    sweep = template["sweeps"][0]
    sweep["id"] = "aggregate"
    sweep["name"] = "aggregate_{M}_{K}_{N}_{K_label}"
    sweep["axes"] = {"M": ["tile-1"], "N": ["tile+1"], "K": boundary_spec()}
    sweep["fit_axes"] = ["K"]
    sweep["base"]["op"] = "matmul"
    sweep["base"]["performance"]["family"] = "aggregate"
    write(options["performance_template"], template)
    return raw


@pytest.mark.parametrize(("tile", "capacity"), [(4, 86), (6, 150)])
def test_normal_writer_generates_joint_aligned_and_tail_boundaries(independent, tile, capacity):
    declare(independent, tile, capacity)
    written = generation.generate_target("fixture", **independent)
    assert 3 <= len(written) <= 7
    seen = {}
    for path in written:
        cap = yaml.safe_load((path / "capsule.yaml").read_bytes())
        assert cap["software_screen"]["status"] == "admitted"
        record = cap["performance"]["emitter"]["derived_axes"]["K"]["derivation"]
        assert record["status"] == "derived" and record["capacity_rows"] == capacity
        assert record["quantum"] == tile and record["reserved_rows"] == 2
        lhs = next(row for row in cap["inputs"] if row["role"] == "input")
        rhs = next(row for row in cap["inputs"] if row["role"] == "weight")
        m, k = lhs["shape"]
        depth, n = rhs["shape"]
        assert m == tile - 1 and n == tile + 1 and depth == k
        point = next(row for row in record["points"] if row["extent"] == k)
        assert point["total_rows"] == point["reserved_rows"] + sum(row["rows"] for row in point["allocations"])
        assert point["fits"] == (point["total_rows"] <= capacity)
        assert point["inequality"] == {
            "lhs": point["total_rows"],
            "rhs": capacity,
            "relation": "<=" if point["fits"] else ">",
        }
        # Do not confuse every buffer fitting separately with a simultaneous fit.
        if "first_overflow" in point["roles"]:
            assert all(row["rows"] <= capacity for row in point["allocations"])
            assert point["total_rows"] > capacity
        golden = GS.load_golden(path)
        assert len(golden["outputs"]["Y0"]) == m
        assert all(len(row) == n for row in golden["outputs"]["Y0"])
        assert golden["golden_source"] == "merlin_tensor_int"
        assert "target execution" in golden["qualification"]
        assert f"tensor<{m}x{k}x" in (path / "capsule.interface.mlir").read_text()
        seen[k] = point
    roles = {role: point for point in seen.values() for role in point["roles"]}
    assert roles["below"]["fits"] and roles["last_fitting"]["fits"]
    assert not roles["first_overflow"]["fits"]
    assert roles["first_overflow"]["extent"] - roles["last_fitting"]["extent"] == tile
    assert any(k % tile for k in seen)
    assert {
        "last_fitting_tail_minus_1",
        "last_fitting_tail_plus_1",
        "first_overflow_tail_minus_1",
        "first_overflow_tail_plus_1",
    } <= roles.keys()
    manifest = yaml.safe_load((independent["output_root"] / "MANIFEST.yaml").read_bytes())
    identity = manifest["performance_generation"]["fixture"]["component_generation"]
    source = next(row for row in identity["generator_sources"] if row["path"] == resource_boundaries.__file__)
    assert source["sha256"] == hashlib.sha256(Path(resource_boundaries.__file__).read_bytes()).hexdigest()
    assert record["raw_facts_sha256"] == hashlib.sha256(independent["rtl_facts"].read_bytes()).hexdigest()


def selected(raw):
    return SimpleNamespace(target="fixture", refreshed_facts=raw, raw_facts_sha256="synthetic")


def derive_from(raw, spec=None):
    return derive(
        spec or boundary_spec(),
        owner="test",
        axis="K",
        target="fixture",
        tile=4,
        dtype="int8",
        fixed={"M": [3], "N": [5]},
        evidence=selected(raw),
        resolve_extent=resolve_extent,
    )


@pytest.mark.parametrize("missing", ["capacity", "reservation", "row_width"])
def test_normal_generator_records_missing_selected_facts(independent, missing):
    raw = declare(independent, 4, 86)
    if missing == "capacity":
        raw["facts"]["memories"][0].pop("bytes")
    elif missing == "reservation":
        raw["facts"].pop("reservations")
    else:
        raw["facts"]["memories"][0].pop("elem_bits")
    write(independent["rtl_facts"], raw)
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    software["component_performance"]["hardware"]["raw_facts_sha256"] = hashlib.sha256(
        independent["rtl_facts"].read_bytes()
    ).hexdigest()
    write(independent["software_spec"], software)
    assert generation.generate_target("fixture", **independent) == []
    report = yaml.safe_load((independent["output_root"] / "MANIFEST.yaml").read_bytes())
    skips = report["performance_generation"]["fixture"]["skipped_inapplicable"]
    assert len(skips) == 1
    record = skips[0]["axis_derivation"]["record"]
    assert record["status"] == "unknown" and record["missing"]
    assert record["declaration"]["reservation_facts"] == [["reservations", "command_bytes"]]


def test_too_small_boundary_does_not_fabricate_below_point(independent):
    raw = declare(independent, 4, 35)
    values, _, record = derive_from(raw)
    assert "below" in record["missing_scenarios"] and 0 not in values
    last = next(p for p in record["points"] if "last_fitting" in p["roles"])
    assert last["total_rows"] < last["capacity_rows"]  # last fitting is not necessarily exactly full
    raw["facts"]["memories"][0]["bytes"] = 4
    with pytest.raises(BoundaryUnavailable) as failure:
        derive_from(raw)
    assert failure.value.record["missing_scenarios"] == ["below", "last_fitting"]
    assert not failure.value.record["first_overflow"]["fits"]


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.update(capacity_fact=["memories", True, "bytes"]),
        lambda s: s["allocations"][0].update(copies=True),
        lambda s: s.update(quantum="tile-4"),
        lambda s: s.update(tail_offsets=[True]),
        lambda s: s["allocations"][0].update(round_up=[True]),
        lambda s: s.update(executable_formula="external code"),
    ],
)
def test_malformed_declarations_stop_normal_generation(independent, change):
    declare(independent, 4, 86)
    template = yaml.safe_load(independent["performance_template"].read_bytes())
    change(template["sweeps"][0]["axes"]["K"])
    write(independent["performance_template"], template)
    with pytest.raises(ValueError):
        generation.generate_target("fixture", **independent)


def test_selected_store_fact_identity_and_reservation_copies(independent):
    raw = declare(independent, 4, 86)
    baseline, _, _ = derive_from(raw)
    spec = boundary_spec()
    spec["allocations"][0]["copies"] = 2
    doubled, _, record = derive_from(raw, spec)
    assert max(doubled) < max(baseline)
    assert record["physical_store"]["row_bytes"] == 4
    wrong = deepcopy(raw)
    wrong["facts"]["memories"].append({**wrong["facts"]["memories"][0], "name": "other"})
    spec["capacity_fact"] = ["memories", 1, "bytes"]
    with pytest.raises(ValueError, match="actual byte capacity"):
        derive_from(wrong, spec)
    wrong["facts"]["memories"][1]["name"] = "scratch"
    with pytest.raises(BoundaryUnavailable, match="ambiguous"):
        derive_from(wrong)
    raw["facts"]["reservations"]["command_bytes"] = 1
    with pytest.raises(BoundaryUnavailable, match="exact physical row"):
        derive_from(raw)


def test_equal_tail_footprints_keep_distinct_extents_and_roles(independent):
    raw = declare(independent, 4, 86)
    spec = boundary_spec()
    spec["allocations"][1]["round_up"] = [0]
    values, labels, record = derive_from(raw, spec)
    last = next(p for p in record["points"] if "last_fitting" in p["roles"])
    tail = next(p for p in record["points"] if "last_fitting_tail_minus_1" in p["roles"])
    assert last["extent"] != tail["extent"] and last["total_rows"] == tail["total_rows"]
    assert last["extent"] in values and tail["extent"] in values
    assert labels[last["extent"]] != labels[tail["extent"]]


def test_distinct_physical_stores_are_never_summed(independent):
    raw = declare(independent, 4, 86)
    raw["facts"]["memories"].append({"name": "result", "bytes": 16 * 25, "depth": 25, "row_elems": 4, "elem_bits": 32})
    # Huge unused second store cannot change this declared scratch footprint.
    before, _, _ = derive_from(raw)
    raw["facts"]["memories"][1]["bytes"] *= 100
    after, _, _ = derive_from(raw)
    assert after == before
    raw["facts"]["memories"][1]["bytes"] = 16 * 25
    spec = boundary_spec()
    spec.update(
        store="result",
        capacity_fact=["memories", 1, "bytes"],
        reservation_facts=[],
        allocations=[{"name": "result", "shape": ["M", "K"], "dtype": "i32"}],
    )
    _, _, record = derive_from(raw, spec)
    assert record["capacity_rows"] == 25 and record["physical_store"]["row_bytes"] == 16
    assert all([row["name"] for row in p["allocations"]] == ["result"] for p in record["points"])
    spec["allocations"][0]["store"] = "scratch"
    with pytest.raises(ValueError, match="allocation must declare"):
        derive_from(raw, spec)


def test_explicit_evidence_is_required_and_fixed_geometry_cannot_be_borrowed(independent):
    raw = declare(independent, 4, 86)
    kwargs = dict(
        owner="test",
        axis="K",
        target="fixture",
        tile=4,
        dtype="int8",
        fixed={"M": [3], "N": [5]},
        resolve_extent=resolve_extent,
    )
    with pytest.raises(BoundaryUnavailable, match="explicit refreshed"):
        derive(boundary_spec(), evidence=None, **kwargs)
    kwargs["fixed"]["M"] = [3, 5]
    with pytest.raises(ValueError, match="single positive fixed axis"):
        derive(boundary_spec(), evidence=selected(raw), **kwargs)


def test_quantum_one_records_unavailable_tails_without_losing_boundary(independent):
    raw = declare(independent, 1, 86)
    spec = boundary_spec()
    spec["allocations"] = [{"name": "operand", "shape": ["K"], "dtype": "operand"}]
    kwargs = dict(
        owner="test",
        axis="K",
        target="fixture",
        tile=1,
        dtype="int8",
        fixed={},
        evidence=selected(raw),
        resolve_extent=resolve_extent,
    )
    values, labels, record = derive(spec, **kwargs)
    assert values == [83, 84, 85]
    assert set(labels.values()) == {"below", "last_fitting", "first_overflow"}
    assert len(record["unavailable_tails"]) == 4
    assert len(record["missing_scenarios"]) == 4
    spec["tail_offsets"] = []
    empty_values, _, empty = derive(spec, **kwargs)
    assert empty_values == values and empty["tail_status"] == "not_declared"
    assert empty["missing_scenarios"] == empty["unavailable_tails"] == []


def test_existing_memory_regime_axis_and_fitting_validation_remain_available(independent):
    from merlin_experiments.phase0.profiles import _validate_declared_fit_axes
    from merlin_experiments.phase0.sweeps import _resolve_derived_axis

    raw = declare(independent, 4, 86)
    spec = {"derive": "memory_regime_reduction_depth", "points_per_regime": 2, "regimes": ["fits_single", "spills"]}
    sweep = {"axes": {"K": spec}, "fit_axes": ["K"]}
    _validate_declared_fit_axes(sweep, owner="test")
    values, labels, record = _resolve_derived_axis(
        spec,
        owner="test",
        axis="K",
        target="fixture",
        tile=4,
        dtype="int8",
        fixed={"M": [3], "N": [5]},
        evidence=selected(raw),
    )
    assert values and set(labels.values()) == {"fits_single", "spills"}
    assert record["capacity_rows"] == 86 and record["m_extent"] == 3 and record["n_extent"] == 5
    spec["points_per_regime"] = 1
    with pytest.raises(ValueError, match="fewer than two points"):
        _validate_declared_fit_axes(sweep, owner="test")
