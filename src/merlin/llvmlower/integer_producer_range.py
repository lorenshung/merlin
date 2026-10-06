"""Conservative integer sum-of-products domains supplied by a typed producer.

The caller must bind these facts to the actual producer, output storage and
consumer. This arithmetic certificate does not establish that source binding.
It admits exact signed-i32 accumulation only, with every prefix in range.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class IntegerSumProductsRange:
    terms: int
    lhs_min: int
    lhs_max: int
    rhs_min: int
    rhs_max: int
    initial_min: int = 0
    initial_max: int = 0

    def interval(self):
        values = (
            self.terms,
            self.lhs_min,
            self.lhs_max,
            self.rhs_min,
            self.rhs_max,
            self.initial_min,
            self.initial_max,
        )
        if any(type(value) is not int for value in values) or self.terms < 0:
            raise ValueError("integer bounds and nonnegative term count required")
        for low, high in (
            (self.lhs_min, self.lhs_max),
            (self.rhs_min, self.rhs_max),
            (self.initial_min, self.initial_max),
        ):
            if not -(1 << 31) <= low <= high < (1 << 31):
                raise ValueError("ordered signed-i32 source bounds required")
        products = [a * b for a in (self.lhs_min, self.lhs_max) for b in (self.rhs_min, self.rhs_max)]
        low = self.initial_min + self.terms * min(products)
        high = self.initial_max + self.terms * max(products)
        prefix_low = min(self.initial_min, low)
        prefix_high = max(self.initial_max, high)
        if prefix_low < -(1 << 31) or prefix_high >= (1 << 31):
            raise ValueError("possible signed-i32 accumulation overflow")
        if self.terms and (min(products) < -(1 << 31) or max(products) >= (1 << 31)):
            raise ValueError("possible signed-i32 product overflow")
        return low, high

    def require_contained(self, low, high):
        actual_low, actual_high = self.interval()
        if actual_low < low or actual_high > high:
            raise ValueError("readout domain does not cover producer interval")
        return actual_low, actual_high
