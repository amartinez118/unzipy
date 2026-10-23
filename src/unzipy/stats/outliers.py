"""Outlier detection."""

from __future__ import annotations

import numpy as np
from scipy import stats


def grubbs_outliers(a, alpha=0.05):
    """Flag outliers with the iterative two-sided Grubbs test.

    The test assumes approximately normal data. Constant data have no outliers.

    Args:
        a (array-like): One-dimensional sample.
        alpha (float): Significance level.

    Returns:
        numpy.ndarray: Boolean mask, True for outliers.
    """
    a = np.asarray(a, dtype=float)
    mask = np.zeros(a.size, dtype=bool)
    while True:
        keep = np.flatnonzero(~mask & np.isfinite(a))
        n = keep.size
        if n < 3:
            return mask
        x = a[keep]
        if np.ptp(x) == 0:  # no spread: nothing can be an outlier
            return mask
        dev = np.abs(x - x.mean())
        i = np.argmax(dev)
        t = stats.t.ppf(alpha / (2 * n), n - 2)  # negative
        g_crit = (n - 1) / np.sqrt(n) * np.sqrt(t * t / (n - 2 + t * t))
        if dev[i] / x.std(ddof=1) <= g_crit:
            return mask
        mask[keep[i]] = True


def iqr_outliers(a, k=3.0):
    """Flag values beyond ``k`` interquartile ranges of the quartiles.

    Args:
        a (array-like): One-dimensional sample.
        k (float): 1.5 for minor and 3 for major outliers.

    Returns:
        numpy.ndarray: Boolean mask, True for outliers.
    """
    a = np.asarray(a, dtype=float)
    q1, q3 = np.nanpercentile(a, [25, 75])
    return (a < q1 - k * (q3 - q1)) | (a > q3 + k * (q3 - q1))
