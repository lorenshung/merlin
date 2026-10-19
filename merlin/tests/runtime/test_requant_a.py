"""Complete monotone-boundary proofs, exact ties, overflow and contract parity."""

import importlib.util
import os
import random
from fractions import Fraction as F
from pathlib import Path

import pytest

from merlin.runtime.requant_a import apply, exact, plan, validate


def test_identity_and_ties():
    p = plan(F(1))
    assert (p["constants"]["M"], p["constants"]["B"], p["constants"]["S"]) == ("1", "0", 0)
    for f, c in [(F(1, 2), F(0)), (F(3, 8), F(1, 2)), (F(1), F(1, 2))]:
        p = plan(f, c)
        assert p["algorithm"] == "threshold_table"
        assert len(p["thresholds"]) <= 255
        assert all(apply(x, p) == exact(x, f, c, 0, -128, 127) for x in range(-2048, 2049))


def test_random_full_int32_domains_and_adversarial_boundaries():
    rng = random.Random(7234)
    ratios = [F(1, 1 << 149), F(1 << 127), F(1, 3), F(17, 31), F(0xFFFFFF, 1 << 24)]
    ratios += [F(rng.randrange(1, 1 << 24), rng.randrange(1, 1 << 24)) for _ in range(60)]
    for f in ratios:
        c = F(rng.randrange(-500, 500), rng.randrange(1, 40))
        zp = rng.randrange(255)
        p = plan(f, c, zp, 0, 255, bool(rng.randrange(2)))
        assert validate(p) == p["proof"]
        from merlin.runtime.requant_a import transitions

        ts = transitions(f, c, zp, 0, 255, p["relu"], -(1 << 31), (1 << 31) - 1)
        points = {-(1 << 31), (1 << 31) - 1, 0}
        points.update(v for t in ts for v in (t - 1, t) if -(1 << 31) <= v < 1 << 31)
        points.update(rng.randrange(-(1 << 31), 1 << 31) for _ in range(50))
        for x in points:
            assert apply(x, p) == exact(x, f, c, zp, 0, 255, p["relu"])


def test_exhaustive_small_domain():
    for f in [F(1, 2), F(3, 8), F(5, 7), F(1), F(2), F(17, 31)]:
        for c in [F(0), F(-1, 2), F(7, 3)]:
            p = plan(f, c, 123, 1, 255, False, -1024, 1024)
            assert all(apply(x, p) == exact(x, f, c, 123, 1, 255) for x in range(-1024, 1025))


@pytest.mark.parametrize("key,value", [("M", str(1 << 63)), ("B", str(1 << 63)), ("S", 63), ("L", str(-(1 << 63) - 1))])
def test_imported_invalid_constants_are_refused(key, value):
    p = plan(F(17, 31), F(2, 3), 123, 1, 255)
    p["constants"][key] = value
    with pytest.raises(ValueError):
        validate(p)


def test_hash_table_and_rounding_corruption_refused():
    p = plan(F(1, 2))
    p["thresholds"][0] = str(int(p["thresholds"][0]) + 1)
    with pytest.raises(ValueError):
        validate(p)
    p = plan(F(17, 31))
    p["proof"]["checked_points_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate(p)
    p = plan(F(1))
    p["rounding"] = "half_up"
    with pytest.raises(ValueError):
        validate(p)


def test_provider_contract_round_trip():
    selected = os.environ.get("MERLIN_REQUANT_A_PROVIDER")
    if not selected:
        pytest.skip("no external requantization provider selected")
    path = Path(selected)
    if not path.is_file():
        pytest.skip("external provider prototype unavailable")
    spec = importlib.util.spec_from_file_location("provider_requant_a", path)
    provider = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provider)
    for f, c in [(F(17, 31), F(2, 3)), (F(1, 2), F(0)), (F(1, 1 << 149), F(-1, 7)), (F(1 << 127), F(0))]:
        local = plan(f, c, 123, 0, 255)
        remote = provider.plan(f, c, 123, 0, 255)
        assert provider.validate(local) == local["proof"]
        assert validate(remote) == remote["proof"]
        assert local == remote


def test_imported_provider_manifest_refuses_tampered_constants():
    from copy import deepcopy

    from merlin.runtime.requant_a import validate_manifest

    record = plan(F(17, 31))
    document = {"kernels": [{"meta": {"host_requantization_proofs": [record, deepcopy(record)]}}]}
    assert validate_manifest(document) == 1
    document["kernels"][0]["meta"]["host_requantization_proofs"][1]["constants"]["M"] = "0"
    with pytest.raises(ValueError):
        validate_manifest(document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r.pop("proof"),
        lambda r: r["domain"].update(min="-02147483648"),
        lambda r: r["constants"].update(S=0.5),
        lambda r: r.update(relu=0),
        lambda r: r["constants"].update(out_clamp=1),
    ],
)
def test_invalid_certificate_types_are_refused(mutation):
    r = plan(F(17, 31))
    mutation(r)
    with pytest.raises(ValueError):
        validate(r)
