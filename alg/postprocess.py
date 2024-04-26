from numbers import Real

import numpy as np

from data_io.fixed import to_fixed, unscale, SCALE

"""
Truncation and Folding Mechanisms
"""
"""From: https://github.com/IBM/differential-privacy-library"""


class TruncationAndFolding:
    def __init__(self, lower, upper):
        self.lower, self.upper = self._check_bounds(lower, upper)
        self.lower_f, self.upper_f = self.lower * SCALE, self.upper * SCALE

    @classmethod
    def _check_bounds(cls, lower, upper):
        """Performs a check on the bounds provided for the mechanism."""
        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return lower, upper

    def _truncate(self, value):
        if value > self.upper:
            return self.upper
        if value < self.lower:
            return self.lower

        return value

    def _fold(self, value):
        while value < self.lower_f or value > self.upper_f:
            if value < self.lower_f:
                value = (2 * self.lower_f - value)
            else:
                value = (2 * self.upper_f - value)
        return value

    def fold(self, values):
        fixed_vals = to_fixed(values)
        for x in np.nditer(fixed_vals, op_flags=['readwrite']):
            x[...] = self._fold(x[...])
        return unscale(fixed_vals)

    def truncate(self, values):
        for x in np.nditer(values, op_flags=['readwrite']):
            x[...] = self._truncate(x[...])
        return values
