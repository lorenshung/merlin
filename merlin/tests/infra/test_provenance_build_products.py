"""Build-product and loss declarations cannot silently substitute other bytes."""

from pathlib import Path

import pytest
import yaml

from merlin.common import provenance as P
from merlin.common.provenance_lost import parse_lost


def _loss(**changes):
    return {
        "digest": "a" * 64,
        "what": "a previously attributed compiler",
        "lost_on": "2026-01-01",
        "why_unrecoverable": "original bytes were overwritten",
        "still_verifies": ["independently retained output"],
        **changes,
    }


@pytest.mark.parametrize("relative", ["../private", "/private", "x/../private", "x//y", "./x", "", "x\\y", 12])
def test_product_paths_must_be_ordinary_checkout_relative_paths(tmp_path, relative):
    registry = tmp_path / "pins.yaml"
    registry.write_text(yaml.safe_dump({"pins": {"p": {"commit": "b" * 40, "build_products": {relative: "a" * 64}}}}))
    with pytest.raises(P.PinsError, match="repo-relative path"):
        P.load_pins(registry)


@pytest.mark.parametrize("parent_link", [False, True])
def test_indirect_product_never_reads_external_bytes(tmp_path, monkeypatch, parent_link):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (private / "binary").write_bytes(b"not an authorized build product")
    if parent_link:
        (checkout / "build").symlink_to(private, target_is_directory=True)
        relative = "build/binary"
    else:
        (checkout / "binary").symlink_to(private / "binary")
        relative = "binary"
    pin = P.Pin(name="p", commit="b" * 40, build_products=((relative, "a" * 64),))
    reads = []
    monkeypatch.setattr(P, "file_digest", lambda path: reads.append(path) or "a" * 64)
    drift, notes = P._build_product_findings(pin, checkout)
    assert reads == [] and notes == []
    assert "symlink component" in drift[0]


@pytest.mark.parametrize("document", [[], {"lost_artifacts": []}, {"lost_artifacts": None}])
def test_malformed_loss_section_cannot_disappear(document):
    with pytest.raises(P.PinsError, match="mapping"):
        parse_lost(document, Path("registry.yaml"))


@pytest.mark.parametrize(
    "changes",
    [
        {"still_verifies": "not a list"},
        {"still_verifies": [None]},
        {"do_not": "not a list"},
        {"what": ["not a string"]},
        {"lost_on": 20260101},
        {"bytes_declared": True},
        {"bytes_declared": -1},
        {"bytes_declared": "13"},
    ],
)
def test_malformed_loss_account_refuses(changes):
    with pytest.raises(P.PinsError):
        parse_lost({"lost_artifacts": {"gone": _loss(**changes)}}, Path("registry.yaml"))


def test_duplicate_lost_digest_and_partial_digest_refuse():
    for records in [{"first": _loss(), "second": _loss()}, {"gone": _loss(digest="abc")}]:
        with pytest.raises(P.PinsError):
            parse_lost({"lost_artifacts": records}, Path("registry.yaml"))


def test_loss_section_is_optional_but_does_not_rename_live_artifacts():
    assert parse_lost({"pins": {}}, Path("registry.yaml")) == {}
    record = parse_lost({"lost_artifacts": {"gone": _loss(bytes_declared=0)}}, Path("registry.yaml"))["gone"]
    assert record.state == P.UNRECOVERABLE and record.bytes_declared == 0
    assert record.to_dict()["digest"] == "a" * 64
