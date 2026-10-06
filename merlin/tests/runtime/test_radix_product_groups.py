import pytest

from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


@pytest.mark.parametrize("bits,digits,k", [(7, 3, 512), (4, 2, 65), (2, 4, 17), (1, 1, 1)])
def test_complete_pairs_and_exact_bounds(bits, digits, k):
    p = plan_radix_product_groups(radix_bits=bits, digits=digits, reduction_length=k)
    assert sorted(pair for g in p.groups for pair in g.pairs) == [(i, j) for i in range(digits) for j in range(digits)]
    assert all((i + j) * bits == g.exponent for g in p.groups for i, j in g.pairs)
    assert p.weighted_absolute_bound == k * ((1 << (bits * digits)) - 1) ** 2
    assert all(g.accumulator_bound == len(g.pairs) * k * ((1 << bits) - 1) ** 2 for g in p.groups)


@pytest.mark.parametrize(
    "args",
    [
        dict(radix_bits=7, digits=3, reduction_length=512, accumulator_bits=16),
        dict(radix_bits=7, digits=4, reduction_length=512),
        dict(radix_bits=0, digits=3, reduction_length=1),
        dict(radix_bits=True, digits=3, reduction_length=1),
        dict(radix_bits=7, digits=3, reduction_length=512, reconstruction_precision_bits=32),
    ],
)
def test_refuse_overflow_rounding_or_invalid_domains(args):
    with pytest.raises(ValueError):
        plan_radix_product_groups(**args)
