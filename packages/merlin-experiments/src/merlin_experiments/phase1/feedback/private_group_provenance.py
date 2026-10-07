"""Exact source-to-emission identity join for operator-private whole-model builds.

Group indices describe one compiler stage and may change during preparation. The
capture frontend's typed region and contributing-node IDs are the join authority;
missing or ambiguous IDs are never replaced by matching counts, shapes or names.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def eligible_source_identity(root: Any) -> tuple[str, tuple[str, ...]]:
    """Read one complete captured root identity, independent of group numbering."""
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    attributes = getattr(root, "attributes", {})
    region = attributes.get("prov.region_id")
    nodes = attributes.get("prov.source_node_ids")
    if not isinstance(region, StringAttr) or not region.data or not isinstance(nodes, ArrayAttr) or not nodes.data:
        raise ValueError("eligible source root lacks complete provenance identity")
    if any(not isinstance(node, StringAttr) or not node.data for node in nodes.data):
        raise ValueError("eligible source root has malformed provenance nodes")
    identities = tuple(sorted(node.data for node in nodes.data))
    if len(identities) != len(set(identities)):
        raise ValueError("eligible source root repeats a provenance node")
    return region.data, identities


def join_routed_source_groups(assigned: Any, source: Mapping[str, Any]) -> dict[int, int]:
    """Join emitted calls to eligible roots by producer-declared region identity.

    A region ID must be unique within one captured program; a repeated ID has no
    unambiguous source owner, regardless of whether this model happened to route it.
    Preparation may renumber groups, so those integers are never join keys.
    """
    expected = source.get("eligible_group_provenance")
    eligible = source.get("eligible_groups")
    if not isinstance(expected, list) or not isinstance(eligible, list) or not isinstance(assigned, list):
        raise ValueError("routed group source roster is incomplete")
    if any(type(group) is not int or group < 0 for group in eligible) or eligible != sorted(set(eligible)):
        raise ValueError("eligible source group indices are ambiguous")

    def identity(row: Any) -> tuple[str, tuple[str, ...]]:
        if not isinstance(row, Mapping):
            raise ValueError("routed group has no provenance record")
        region, nodes = row.get("source_region_id"), row.get("source_node_ids")
        if (
            type(region) is not str
            or not region
            or not isinstance(nodes, list)
            or not nodes
            or any(type(node) is not str or not node for node in nodes)
            or nodes != sorted(set(nodes))
        ):
            raise ValueError("routed group has missing, malformed, or duplicate source provenance")
        return region, tuple(nodes)

    source_by_region: dict[str, tuple[tuple[str, ...], int]] = {}
    source_groups = set()
    for row in expected:
        region, nodes = identity(row)
        group = row.get("source_group")
        if type(group) is not int or group < 0 or group in source_groups or region in source_by_region:
            raise ValueError("eligible source group provenance is ambiguous")
        source_groups.add(group)
        source_by_region[region] = nodes, group
    if len(expected) != len(eligible) or sorted(source_groups) != eligible:
        raise ValueError("eligible source group provenance differs from source eligibility")

    joined: dict[int, int] = {}
    emitted_regions = set()
    for row in assigned:
        region, nodes = identity(row)
        group = row.get("group")
        if type(group) is not int or group < 0 or group in joined or region in emitted_regions:
            raise ValueError("routed group provenance or emitted group index is ambiguous")
        emitted_regions.add(region)
        original = source_by_region.get(region)
        if original is None or original[0] != nodes:
            raise ValueError("routed group provenance differs from independently eligible source")
        joined[group] = original[1]
    if len(joined) != len(source_by_region):
        raise ValueError("routed groups omit independently eligible source computation")
    return joined


def routed_kernel_symbols(emitted: Mapping[str, Any]) -> set[str]:
    """Validate unique compiled kernels, independently of the source call roster.

    The trusted emitter may reuse one monomorphic kernel for several source calls.
    Source identities remain one-to-one; sharing never permits a missing call or a
    different dimension/type signature to borrow the same linked symbol.
    """
    rows, signatures = emitted.get("routed"), emitted.get("signatures")
    if not isinstance(rows, list) or not rows or not isinstance(signatures, Mapping):
        raise ValueError("routed kernel signature roster is incomplete")
    by_symbol = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("routed kernel record is malformed")
        symbol = row.get("symbol")
        if (
            type(symbol) is not str
            or not symbol
            or not symbol.isascii()
            or not (symbol[0].isalpha() or symbol[0] in "_.$")
            or any(not (char.isalnum() or char in "_.$") for char in symbol)
        ):
            raise ValueError("routed kernel symbol is missing or malformed")
        parallel, reduction, dtypes = row.get("parallel"), row.get("reduction"), row.get("dtypes")
        if (
            not isinstance(parallel, list)
            or not parallel
            or not isinstance(reduction, list)
            or any(type(size) is not int or size < 0 for size in parallel + reduction)
            or not isinstance(dtypes, list)
            or not dtypes
            or any(type(dtype) is not str or not dtype for dtype in dtypes)
        ):
            raise ValueError("routed kernel has malformed dimension/type signature")
        signature = (tuple(parallel), tuple(reduction), tuple(dtypes))
        if symbol in by_symbol and by_symbol[symbol] != signature:
            raise ValueError("reused kernel symbol has inconsistent dimension/type signature")
        extents = signatures.get(symbol)
        if (
            not isinstance(extents, list)
            or any(type(size) is not int for size in extents)
            or extents != parallel + reduction
        ):
            raise ValueError("routed kernel extents differ from compiled signature")
        by_symbol[symbol] = signature
    if set(signatures) != set(by_symbol):
        raise ValueError("compiled kernel signature roster differs from routed symbols")
    return set(by_symbol)
