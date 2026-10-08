"""The no-FSM rule names this target's hardware-loop instructions, derived from its own facts.

A sealed Phase 0 release prohibited ``loop_descriptor`` and resolved it to ZERO instructions: the
endpoint declaration that binds the role to this target's RTL names was read from the wrong place and
came back empty, so every later phase enforced a rule that forbade nothing. The count is not written
here; the expected set is read from the target's own derived name table -- every instruction its RTL
spells as a hardware loop (the ``LOOP_`` family: the weight-stationary matmul loop and the convolution
loop with their configuration words) -- and the role binding must cover exactly that set, through
both the core derivation and the Phase 0 taxonomy that is sealed into a release.

WHERE THE NAME TABLE COMES FROM. The facts are extracted from RTL, generated and gitignored, so CI has
none. The derived table is therefore pinned as data (``tests/data/rtl_facts/gemmini_decode_names_facts.json``,
with the extraction's digest and the hardware pins it came from) and read through the facts accessor
the binding itself uses, so the binding is checked everywhere the vendored support exists. A host that
extracts live facts also checks that its own derivation still names the pinned table. The only skips are
the genuinely absent ones: no vendored support (an installed distribution), and no live facts.
"""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir

pytestmark = pytest.mark.target("gemmini")

TARGET = "gemmini"
ROLE = "loop_descriptor"
#: This target's own spelling of its hardware-loop family, as its RTL decode table names it.
LOOP_FAMILY = "LOOP_"
PINNED_FACTS = merlin_dir() / "tests" / "data" / "rtl_facts" / "gemmini_decode_names_facts.json"


def _vendored_support():
    from merlin.targetgen import target_registry

    root = target_registry.in_repo_support().get(TARGET)
    if root is None:
        pytest.skip(f"no vendored {TARGET} support in this installation (only a checkout carries one)")
    return root


def _names() -> dict[str, str]:
    from merlin.kernels.decode.rocc import funct_table_for

    return {str(k): str(v) for k, v in (funct_table_for(TARGET).get("names") or {}).items()}


@pytest.fixture
def names(monkeypatch) -> dict[str, str]:
    """The pinned name table, read through the facts accessor with the vendored support selected."""
    assert PINNED_FACTS.is_file(), f"the pinned name table is missing: {PINNED_FACTS}"
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(_vendored_support()))
    monkeypatch.setenv("MERLIN_RTL_FACTS", str(PINNED_FACTS))
    table = _names()
    assert table, "the pinned facts carry no instruction name table; this test would prove nothing"
    return table


def test_the_role_covers_exactly_the_targets_hardware_loop_family(names):
    from merlin.kernels.endpoints import endpoints_for

    endpoints = endpoints_for(TARGET)
    assert endpoints, "no compute endpoint binds this target's roles"
    bound = {name for name in names.values() for endpoint in endpoints if ROLE in endpoint.roles_of(name)}
    family = {name for name in names.values() if name.startswith(LOOP_FAMILY)}
    assert family, "the derived name table has no hardware-loop instructions; this test would prove nothing"
    assert bound == family, (sorted(bound - family), sorted(family - bound))


def test_the_phase0_policy_resolves_the_role_to_that_family(names):
    roles = pytest.importorskip("merlin_experiments.phase0.instruction_roles")
    taxonomy = roles.derive_role_taxonomy(TARGET)
    assert taxonomy["status"] == "derived", taxonomy.get("reason")
    policy = roles.resolve_policy([ROLE], taxonomy)
    assert policy["status"] == roles.RESOLVED and policy["vacuous_roles"] == []
    prohibited = {row["name"] for row in policy["prohibited_instructions"][ROLE]}
    assert prohibited == {name for name in names.values() if name.startswith(LOOP_FAMILY)}
    assert roles.enforcement_problems(policy, [ROLE]) == []


def test_this_hosts_derivation_still_names_the_pinned_table(monkeypatch):
    """Where live facts exist, the pinned table is not stale. A host without them skips, by name."""
    import json

    from merlin.common.paths import ExternalPathUnset
    from merlin.targetgen.rtl import facts as F

    monkeypatch.setenv("MERLIN_TARGET_PATH", str(_vendored_support()))
    monkeypatch.delenv("MERLIN_RTL_FACTS", raising=False)
    try:
        path = F.find_facts(TARGET)
    except (FileNotFoundError, ExternalPathUnset) as exc:
        pytest.skip(f"no live {TARGET} RTL facts on this host: {exc}")
    if path is None:
        pytest.skip(f"no live {TARGET} RTL facts on this host")
    live = _names()
    if not live:
        pytest.skip(f"this host's {TARGET} facts ({path}) carry no instruction name table")
    pinned = json.loads(PINNED_FACTS.read_text(encoding="utf-8"))["facts"]["interfaces"]
    (table,) = [entry for entry in pinned if entry["name"] == "funct_decode_table"]
    assert live == table["names"], "re-pin tests/data/rtl_facts from this host's derivation, with its digest"
