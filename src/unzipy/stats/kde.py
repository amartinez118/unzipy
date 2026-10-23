"""Gaussian kernel density estimation with optimal bandwidth."""

from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde


def optimal_bandwidth(data, candidates=None, chunk=2000, max_n=3000):
    """Return the bandwidth minimising the Shimazaki-Shinomoto cost.

    Minimises ``C(h) = N^-2 sum_ij [phi_{sqrt(2) h}(xi - xj) - 2 phi_h(xi -
    xj)] + 2 phi_h(0) / N`` over a grid of candidates (J. Comput. Neurosci. 29,
    171 (2010)). The cost is O(N^2) per candidate, evaluated in chunks; larger
    samples are subsampled and the bandwidth is rescaled with the ``N**(-1/5)``
    law.

    Args:
        data (array-like): One-dimensional sample.
        candidates (array-like | None): Bandwidths to try; a logarithmic grid
            around the Silverman scale if None.
        chunk (int): Rows evaluated at a time.
        max_n (int): Largest sample used directly.

    Returns:
        float: Bandwidth (same units as ``data``).

    Raises:
        ValueError: If there are fewer than 3 samples or no spread.
    """
    x = np.asarray(data, dtype=float).ravel()
    n = x.size
    if n < 3 or np.ptp(x) == 0:
        raise ValueError(
            "need at least 3 samples with non-zero spread to select a "
            "bandwidth"
        )
    if (
        n > max_n
    ):  # Note: O(N^2) cost, subsample and rescale by the h ~ N^-1/5 law
        sub = np.random.default_rng(0).choice(x, max_n, replace=False)
        return (
            optimal_bandwidth(sub, candidates, chunk, max_n)
            * (max_n / n) ** 0.2
        )
    if candidates is None:
        s = x.std(ddof=1)
        candidates = s * np.logspace(-2, 0.5, 40) * n ** (-0.2) * 3
    candidates = np.asarray(candidates, dtype=float)
    cost = np.zeros(candidates.size)
    for s in range(0, n, chunk):
        d2 = (x[s : s + chunk, None] - x[None, :]) ** 2
        for k, h in enumerate(candidates):
            g2 = np.exp(-d2 / (4 * h * h)) / (2 * h * np.sqrt(np.pi))
            g1 = np.exp(-d2 / (2 * h * h)) / (h * np.sqrt(2 * np.pi))
            cost[k] += np.sum(g2 - 2 * g1) / n**2
    cost += 2.0 / (n * np.sqrt(2 * np.pi) * candidates)
    return candidates[np.argmin(cost)]


def kde_pdf(data, grid, bandwidth=None):
    """Evaluate a Gaussian kernel density estimate.

    Args:
        data (array-like): One-dimensional sample.
        grid (array-like): Points where the density is evaluated.
        bandwidth (float | None): Absolute bandwidth; optimised with
            :func:`optimal_bandwidth` if None.

    Returns:
        numpy.ndarray: Density on ``grid``.

    Raises:
        ValueError: If the sample has no spread.
    """
    x = np.asarray(data, dtype=float).ravel()
    if np.ptp(x) == 0:
        raise ValueError(
            "data have zero spread: the density is a delta function"
        )
    h = optimal_bandwidth(x) if bandwidth is None else bandwidth
    kde = gaussian_kde(x, bw_method=h / x.std(ddof=1))
    return kde(np.asarray(grid, dtype=float))
