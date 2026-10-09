"""Joint-domain arithmetic and actual native joins never issue physical roles."""
import json
import shutil
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_analytical as analytical
from merlin_experiments.phase2 import component_measurement_qualification as measurement
from merlin_experiments.phase2.component_applicability import (
    ComponentApplicabilityObservation,
    applicability_observation,
    qualify_required_applicability_strata,
)
from merlin_experiments.phase2.component_measurement_protocol import ComponentScreenSample
from merlin_experiments.phase2.contracts import StageGateError, sha256_file

from merlin.common import invocation_record
from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_applicability import (
    ComponentApplicabilityCoordinates as Coordinates,
)
from merlin.perf.component_applicability import (
    ComponentApplicabilityDomain as Domain,
)
from merlin.perf.component_applicability import (
    ComponentApplicabilityStratum as Stratum,
)
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
    complete_component_cost,
)
from merlin.perf.fast_estimate_validation import Observation


def point(**changes):
    return replace(Coordinates(
        *(sha(value) for value in ("hardware", "timer", "accuracy", "inputs", "unit")),
        (("moved", 16),), (("storage", 64),), (("storage", 16),), (("storage", 16),),
        (4,), (1,), (0,), False, 0, "fresh", 0, 1,
    ), **changes)


def domain():
    return Domain((
        Stratum("resident", (point(), point(payload_bytes=(("moved", 32),))), ("a", "b", "c")),
        Stratum("streaming", (point(streaming=True, working_set_bytes=(("storage", 128),),
                                   tile_counts=(8,), reuse_state="within_invocation", reuse_count=1),),
                ("d", "e", "f")),
    ), ("fit_a", "fit_b"))


@pytest.mark.parametrize("changes", [
    {name: sha("other")} for name in ("hardware_sha256", "timer_sha256", "accuracy_sha256",
                                      "input_policy_sha256", "unit_meaning_sha256")
] + [
    {"payload_bytes": (("moved", 24),)}, {"capacity_bytes": (("storage", 128),)},
    {"live_bytes": (("storage", 8),)}, {"working_set_bytes": (("storage", 32),)},
    {"tile_shape": (8,)}, {"tile_counts": (2,)}, {"tails": (1,)}, {"streaming": True},
    {"dependency_depth": 1}, {"reuse_state": "cross_invocation", "reuse_count": 2}, {"chain_length": 2},
])
def test_every_unobserved_joint_coordinate_remains_unknown(changes):
    assert domain().lookup(point(**changes))["status"] == "UNKNOWN"


def test_no_cartesian_extrapolation_or_declaration_authority():
    declared = domain()
    assert declared.lookup(point())["stratum"] == "resident"
    assert declared.lookup(None)["status"] == "UNKNOWN"
    # Both payload and streaming/reuse are individually observed, but not jointly.
    combination = replace(declared.strata[1].cells[0], payload_bytes=(("moved", 32),))
    assert declared.lookup(combination)["status"] == "UNKNOWN"
    values = {field.name: None for field in fields(measurement.IndependentMeasurementQualification)}
    values.update(applicability_domain=declared, applicability_sha256=declared.sha256)
    with pytest.raises(StageGateError, match="live evaluated controls"):
        measurement.IndependentMeasurementQualification(**values).verify()


@pytest.mark.parametrize("changes", [
    {"live_bytes": (("storage", 65),)}, {"working_set_bytes": (("storage", 65),)},
    {"tails": (4,)}, {"tile_counts": (True,)}, {"reuse_count": 1}, {"chain_length": 0},
])
def test_impossible_resource_shape_and_reuse_declarations_refuse(changes):
    with pytest.raises(ValueError, match="applicability"):
        point(**changes).verify()


def test_ambiguous_or_leaking_domains_refuse():
    declared = domain()
    with pytest.raises(ValueError, match="ambiguous"):
        replace(declared, strata=declared.strata + (replace(declared.strata[0], id="duplicate"),)).verify()
    with pytest.raises(ValueError, match="calibration membership"):
        replace(declared, calibration_groups=("a",)).verify()
    with pytest.raises(ValueError, match="different HW"):
        invalid = replace(declared.strata[0], cells=(point(), point(timer_sha256=sha("different"))))
        replace(declared, strata=(invalid,)).verify()


def calibrated_rows(declared):
    rows = []
    for stratum in declared.strata:
        for cell in stratum.cells:
            for group in stratum.held_groups:
                for n in range(1, 11):
                    observed = Observation(sha([group, cell.sha256, n]), sha([group, cell.sha256]), group,
                                           declared.sha256, {"/calibrated_lo": n, "/calibrated_hi": n}, n,
                                           (sha([group, n, "diagnostic"]),))
                    rows.extend((stratum.id, cell.sha256, regime, observed) for regime in ("cold", "warm"))
    return rows


def test_each_declared_stratum_and_regime_requires_held_cell_group_coverage():
    declared = domain()
    rows = calibrated_rows(declared)
    reports = qualify_required_applicability_strata(declared, rows, calibration_sha256=sha("calibration"))
    assert set(reports) == {(row.id, regime) for row in declared.strata for regime in ("cold", "warm")}
    assert all(report["exposable"] for report in reports.values())
    for filter_out in (
        lambda row: row[0] == "streaming" and row[2] == "warm",
        lambda row: row[0] == "resident" and row[1] == point().sha256,
        lambda row: row[0] == "resident" and row[1] == point().sha256 and row[3].group == "a",
    ):
        with pytest.raises(StageGateError, match="omitted"):
            qualify_required_applicability_strata(declared, [row for row in rows if not filter_out(row)],
                                                  calibration_sha256=sha("calibration"))


def test_bad_required_stratum_cannot_hide_in_good_aggregate_feedback():
    declared = domain()
    rows = calibrated_rows(declared)
    corrupt = [(ident, cell, regime, replace(row, cycles=100 - row.cycles)
                if ident == "streaming" and regime == "warm" else row)
               for ident, cell, regime, row in rows]
    with pytest.raises(StageGateError, match="ranking/error/coverage"):
        qualify_required_applicability_strata(declared, corrupt, calibration_sha256=sha("calibration"))
    for bad in (replace(rows[0][3], group="fit_a"), replace(rows[0][3], domain=sha("foreign"))):
        with pytest.raises(StageGateError, match="leaks or selects"):
            qualify_required_applicability_strata(declared, [(*rows[0][:3], bad), *rows[1:]],
                                                  calibration_sha256=sha("calibration"))


def test_semantic_strata_cannot_split_variants_out_of_their_source_family(tmp_path):
    # Diagnostic objects exercise a refusal before any normal control/role calls.
    variant = SimpleNamespace(sha256=sha("variant"), verify=lambda: None)
    member = SimpleNamespace(source_sha256=sha("member"), family="different_family")
    sample = ComponentScreenSample(sha([variant.sha256, member.source_sha256]), member, variant, "a", {},
                                    tmp_path / "held", ("normal",), "resident")
    context = SimpleNamespace(prepare_screen_samples=lambda _root: (sample,))
    admission = SimpleNamespace(corpus=SimpleNamespace(capsules=(member,)))
    with pytest.raises(StageGateError, match="leak groups"):
        measurement._samples(context, tmp_path, admission, (variant,), domain())


def calibration(target):
    return {
        "schema": "phase2_host_analytical_calibration_v1", "target_sha256": target,
        "evidence_sha256s": [sha(value) for value in ("composition", "compute", "movement", "encoding")],
        "composition": {"operator": "sum", "eta": 0, "provenance_sha256": sha("composition")},
        "accelerator_compute_roles": ["execute"], "risk_score": 0,
        "features": [{"id": kind, "pointer": "/count", "resource": kind, "kind": kind,
                      "cycles_per_unit": {"lo": 1, "hi": 1}, "provenance_sha256": sha(kind),
                      "physical_bytes_per_unit": 4 if kind != "compute" else None,
                      "commands_per_unit": 1 if kind != "compute" else None,
                      "transitions_per_unit": 1 if kind == "encoding" else None}
                     for kind in ("compute", "movement", "encoding")],
    }


def test_complete_cost_preserves_unknown_outside_exact_joint_domain():
    declared, selected = domain(), point()
    scope = ComponentCostScope(selected.timer_sha256, selected.accuracy_sha256, selected.input_policy_sha256)
    region = ComponentCostRegion("complete", COMPLETE_STAGES, ("compute",), {"count": 10})
    features = ComponentFeatureObservation(*(sha(value) for value in ("compiler", "member", "corpus", "target")),
                                           scope.sha256, declared.sha256, sha("elf"), sha("deps"), sha("inputs"),
                                           (sha("evidence"),), (region,), (region,), "PASS", legality_status="PASS")
    totals, _report = complete_component_cost(features, calibration(features.target_sha256), scope=scope,
                                              qualified_domains=(declared.sha256,), applicability_domain=declared,
                                              applicability_coordinates=selected)
    assert totals["cold"].resolved and totals["cold"].lo == 10
    totals, report = complete_component_cost(features, calibration(features.target_sha256), scope=scope,
                                             qualified_domains=(declared.sha256,), applicability_domain=declared,
                                             applicability_coordinates=point(tile_counts=(2,)))
    assert not totals["cold"].resolved and not totals["warm"].resolved
    assert report["regimes"]["cold"]["regions"][0]["cycles"]["resolved"]
    with pytest.raises(StageGateError, match="independently qualified semantic applicability"):
        analytical._applicability(SimpleNamespace(independent_runtime=SimpleNamespace(qualification=SimpleNamespace())),
                                   features, Path("unused"))


def write_product(path, document):
    path.write_text(json.dumps(document))


def test_actual_native_derivation_join_cannot_admit_an_unobserved_product(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native diagnostic compiler unavailable")
    root, source = tmp_path / "private", tmp_path / "control.c"
    root.mkdir()
    source.write_text("int main(void) { return 0; }\n")
    elf = root / "control.elf"
    result = invocation_record.run(
        [str(Path(compiler).resolve()), str(source), "-o", str(elf)], directory=root,
        stage="native_diagnostic_link", inputs=(source,), outputs=(elf,), capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    declared = domain()
    features = SimpleNamespace(domain_sha256=declared.sha256, compiler_sha256=sha("compiler"),
                               member_sha256=sha("member"), executable_sha256=sha256_file(elf),
                               inputs_sha256=sha("inputs"), dependencies_sha256=sha("dependencies"))
    document = {"schema": "merlin.component_applicability_observation.v1", "domain_sha256": declared.sha256,
                "coordinates": point().to_dict(), **{name: value for name, value in vars(features).items()
                                                     if name != "domain_sha256"}}
    product = root / "diagnostic.json"
    with invocation_record.observe_call(root, stage="native_diagnostic_observation", function=write_product,
                                         arguments={"path": str(product)}, inputs=(elf,), outputs=(product,)) as call:
        write_product(product, document)
        call.returned()
    # Deliberately unissued diagnostic context only tests actual file/invocation joins.
    context = SimpleNamespace(verify_component_applicability=lambda *_args: None)
    runtime = SimpleNamespace(qualification=SimpleNamespace(context=context))
    observation = ComponentApplicabilityObservation(point(), product, sha256_file(product),
                                                     tuple((root / "invocations").glob("*/invocation.json")))
    lookup, _pins = applicability_observation(observation=observation, features=features, domain=declared,
                                              runtime=runtime, owner=root)
    assert lookup["status"] == "IN_DOMAIN"
    foreign = root / "unobserved.json"
    foreign.write_bytes(product.read_bytes())
    with pytest.raises(StageGateError, match="exact executed-ELF derivation"):
        applicability_observation(observation=replace(observation, product=foreign), features=features,
                                   domain=declared, runtime=runtime, owner=root)
    with invocation_record.observe_call(
        root, stage="unmatched_diagnostic_observation", function=write_product,
        arguments={"path": str(foreign)}, inputs=(source,), outputs=(foreign,),
    ) as call:
        write_product(foreign, document)
        call.returned()
    unmatched = replace(observation, product=foreign,
                        invocations=tuple((root / "invocations").glob("*/invocation.json")))
    with pytest.raises(StageGateError, match="exact executed-ELF derivation"):
        applicability_observation(observation=unmatched, features=features, domain=declared,
                                   runtime=runtime, owner=root)
