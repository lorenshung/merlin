"""Model component certification must retain the ordinary RTL admission boundary."""

from contextlib import contextmanager

import pytest

from merlin.targetgen import package_certification, rtl_simulator_execution


@pytest.mark.parametrize("pin_state", ["stable", "invalid", "replaced"])
def test_certification_checks_selected_product_inside_shared_slot(monkeypatch, pin_state):
    events = []

    @contextmanager
    def slot(*, timeout):
        events.append("admitted")
        assert timeout == 30
        yield 17
        events.append("released")

    def verify(backend, engine, target):
        assert events[0] == "admitted"
        assert (engine, target) == ("verilator", "example")
        if pin_state == "invalid":
            raise ValueError("selected product changed")
        identity = "replacement" if pin_state == "replaced" and "executed" in events else "selected"
        events.append("verified")
        return {"execution_identity": identity}

    def execute(cb, llvm, **options):
        assert events == ["admitted", "verified"]
        assert 0 < options["timeout"] <= 17
        assert options["inputs"] == {"Activation": [[3]]}
        events.append("executed")
        return {
            "outputs": {"Result": [[6]]},
            "oracle": {"kind": "elaborated_rtl", "derived_from_rtl": True},
            "timing": {"sim_active_s": 2},
        }

    monkeypatch.setattr(rtl_simulator_execution, "verilator_slot", slot)
    monkeypatch.setattr(rtl_simulator_execution, "verify_selected", verify)
    monkeypatch.setattr(package_certification._runtime.oot_compile, "run_on_oracle", execute)

    def certify():
        return package_certification._run_certificate_oracle(
            object(), {}, "LLVM", simulator="verilator", target="example", timeout=30, inputs={"Activation": [[3]]}
        )

    if pin_state == "stable":
        result = certify()
        assert result["oracle"]["provenance"]["execution_identity"] == "selected"
        published = package_certification._certificate_oracle_record(result, "verilator", True)
        assert published["provenance"] == result["oracle"]["provenance"]
        assert published["derived_from_rtl"] is True
        assert published["timing"]["sim_active_s"] == 2
        assert published["result"] == "pass"
        assert events == ["admitted", "verified", "executed", "verified", "released"]
    else:
        with pytest.raises((ValueError, RuntimeError), match="changed"):
            certify()
        assert ("executed" in events) is (pin_state == "replaced")
