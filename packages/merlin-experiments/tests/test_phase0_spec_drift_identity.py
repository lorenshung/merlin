"""Coverage drift's legacy readers remain inside the supplied source identity."""

import pytest
from merlin_experiments.phase0 import spec_drift

from merlin.targetgen import target_registry
from merlin.targetgen.rtl import facts


@pytest.mark.parametrize("supplied", [{"facts": {"arrays": [{"name": "mesh", "rows": 3}]}}, None])
@pytest.mark.parametrize("raise_inside", [False, True])
def test_supplied_views_never_extract_and_restore_nested_context(monkeypatch, supplied, raise_inside):
    contract = {"name": "fixture", "compute_units": []}
    outside = {"facts": {"outside": True}}
    monkeypatch.setattr(facts, "ensure_facts", lambda *a, **k: pytest.fail("drift attempted live extraction"))
    monkeypatch.setattr(target_registry, "resolve", lambda *a, **k: pytest.fail("drift attempted live registry"))
    seen = []

    def derive(**kwargs):
        from merlin.targetgen.target_registry import load_contract

        assert facts.load_facts("fixture") == (supplied or {})
        assert load_contract("fixture") == contract
        # Detached values are not authority to mutate the selected input.
        facts.load_facts("fixture")["mutated"] = True
        assert facts.load_facts("fixture") == (supplied or {})
        seen.append(kwargs["raw_facts"])
        if raise_inside:
            raise ValueError("owned failure")
        return {}

    monkeypatch.setattr(spec_drift.D, "fact_capabilities", derive)
    monkeypatch.setattr(spec_drift.D, "resolve_spec", lambda spec, capability: (spec, {}))
    monkeypatch.setattr(spec_drift.D, "spec_fact_drift", lambda *a, **k: {})
    with facts.observed_facts("fixture", outside), target_registry.observed_contract("fixture", {"name": "outer"}):
        options = dict(
            target="fixture",
            software_spec={},
            contract=contract,
            raw_facts=supplied,
            readout_facets=(),
            quantization_candidates=(),
            taxonomy={},
        )
        if raise_inside:
            with pytest.raises(ValueError, match="owned failure"):
                spec_drift.from_views(**options)
        else:
            assert spec_drift.from_views(**options) == {"prohibited_instruction_roles": []}
        assert facts.load_facts("fixture") == outside
        assert target_registry.load_contract("fixture") == {"name": "outer"}
    assert seen == [supplied]
    with pytest.raises(pytest.fail.Exception, match="live extraction"):
        facts.load_facts("fixture")
