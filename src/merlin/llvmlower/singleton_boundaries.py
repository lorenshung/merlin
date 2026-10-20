"""Retain closed singleton identities before exposing scalar IEEE boundaries.

Splitting a tuple snapshots its initializers as inputs. Prove index fills and
integer monoid identities first, while their closed seed relation is explicit.
Unknown or nonidentity computations retain their original scalar operations.
"""


def normalize_singleton_boundaries(module):
    from merlin.llvmlower.singleton_reduction_boundaries import expose_singleton_reduction_boundaries
    from merlin.llvmlower.singleton_reductions import fold_singleton_reductions
    from merlin.llvmlower.singleton_selectors import fold_singleton_selectors
    from merlin.llvmlower.singleton_tuple_regions import expose_singleton_tuple_regions

    changed = fold_singleton_reductions(module) if any(op.name == "linalg.reduce" for op in module.walk()) else 0
    changed += fold_singleton_selectors(module)
    changed += expose_singleton_reduction_boundaries(module)
    changed += expose_singleton_tuple_regions(module)
    return changed
