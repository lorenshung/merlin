"""The device plane an elementwise group states, and where its feature axis comes from.

A rank-4 activation has no self-evident axis order, and picking the wrong one is not a near miss:
it states the right number of elements in the wrong order, so every count-based check still passes
and the program is numerically wrong. These pin that the axis is DERIVED from the capture's own
framework operators and that the derivation fails closed when it cannot be made.
"""

from __future__ import annotations

import pytest

from merlin.xdsl_dialects.lowering import group_demand as GD


def test_a_rank_two_tensor_is_already_the_device_plane():
    assert GD._activation_plane([3136, 256], object()) == (3136, 256)


def test_a_rank_four_activation_folds_every_axis_but_the_features(monkeypatch):
    monkeypatch.setattr(GD, "feature_axis", lambda op, rank: 1)
    assert GD._activation_plane([1, 256, 56, 56], object()) == (3136, 256)
    assert GD._activation_plane([1, 2048, 7, 7], object()) == (49, 2048)


def test_the_axis_moves_the_plane(monkeypatch):
    """The axis is load-bearing: a different one is a different shape, not a rounding difference."""
    monkeypatch.setattr(GD, "feature_axis", lambda op, rank: 3)
    assert GD._activation_plane([1, 256, 56, 56], object()) == (14336, 56)


def test_a_rank_the_vocabulary_cannot_state_is_refused():
    with pytest.raises(GD.NoCapsuleForm, match="rank 1"):
        GD._activation_plane([7], object())


class _Attr:
    def __init__(self, data):
        self.data = data


class _Op:
    """The smallest thing `feature_axis` reads: a provenance tag and a walkable parent chain."""

    def __init__(self, tag="", children=()):
        # The literal name a capture writes, NOT `GD._PROVENANCE_ATTRIBUTE` -- a fake that keys
        # itself off the name under test would follow it wherever it moved and prove nothing.
        self.attributes = {"prov.aten": _Attr(tag)} if tag else {}
        self._children = list(children)

    def parent_op(self):
        return None

    def walk(self):
        yield self
        for child in self._children:
            yield from child.walk()


def test_the_axis_comes_from_the_captures_own_operators():
    module = _Op(children=[_Op("aten.conv2d.default"), _Op("aten.add_.Tensor")])
    assert GD.feature_axis(module, 4) == 1


def test_an_operator_indifferent_to_the_order_does_not_state_one():
    """`aten.add` is elementwise; its signature fixes nothing, so it must not be consulted."""
    module = _Op(children=[_Op("aten.add_.Tensor")])
    with pytest.raises(GD.NoCapsuleForm, match="records no framework operator"):
        GD.feature_axis(module, 4)


def test_operators_that_disagree_are_refused_rather_than_resolved(monkeypatch):
    monkeypatch.setitem(GD._ATEN_FEATURE_AXIS, "aten.max_pool2d.default", {4: 3})
    module = _Op(children=[_Op("aten.conv2d.default"), _Op("aten.max_pool2d.default")])
    with pytest.raises(GD.NoCapsuleForm, match="disagree on the feature axis"):
        GD.feature_axis(module, 4)


def test_an_unrecorded_provenance_attribute_states_nothing(monkeypatch):
    monkeypatch.setattr(GD, "_PROVENANCE_ATTRIBUTE", "prov.nothing")
    module = _Op(children=[_Op("aten.conv2d.default")])
    with pytest.raises(GD.NoCapsuleForm, match="records no framework operator"):
        GD.feature_axis(module, 4)


def test_an_operator_is_consulted_only_about_the_rank_its_signature_mentions():
    """A convolution's signature says where a rank-4 activation keeps its features and nothing
    about a rank-3 one; consulting it anyway would state a transformer's layout from a CNN's."""
    module = _Op(children=[_Op("aten.conv2d.default")])
    assert GD.feature_axis(module, 4) == 1
    with pytest.raises(GD.NoCapsuleForm, match="rank-3"):
        GD.feature_axis(module, 3)


def test_a_rank_three_activation_keeps_its_features_last_when_a_batched_matmul_says_so():
    module = _Op(children=[_Op("aten.bmm.default")])
    assert GD.feature_axis(module, 3) == 2
    assert GD._activation_plane([2, 49, 512], module) == (98, 512)
