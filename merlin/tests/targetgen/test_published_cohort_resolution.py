"""Coverage reads the immutable cohort build behind this module's own publish link, never a foreign link."""

from __future__ import annotations

import os

import pytest

from merlin.targetgen.contract import materialize as M


def test_the_publish_link_resolves_to_its_build(tmp_path):
    build = tmp_path / ".fixture.build.4242.abcd1234"
    build.mkdir()
    link = tmp_path / "fixture"
    os.symlink(build.name, link)
    assert M.resolve_published_cohort(link) == build
    plain = tmp_path / "plain"
    plain.mkdir()
    assert M.resolve_published_cohort(plain) == plain


@pytest.mark.parametrize(
    "target",
    [
        "elsewhere",  # a sibling that is not a build directory
        ".other.build.4242.abcd1234",  # another target's build
        "../outside/.fixture.build.4242.abcd1234",  # leaves the cache namespace
    ],
)
def test_a_foreign_link_is_refused(tmp_path, target):
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / ".other.build.4242.abcd1234").mkdir()
    (tmp_path.parent / "outside" / ".fixture.build.4242.abcd1234").mkdir(parents=True, exist_ok=True)
    link = tmp_path / "fixture"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="did not publish"):
        M.resolve_published_cohort(link)
