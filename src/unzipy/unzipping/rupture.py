"""Rupture-force distributions, survival probabilities and rates.

For unfolding at constant loading rate ``r`` (pN/s), with rupture force density
``p(F)``, the survival probability of the folded state is ``S(F) = 1 - int p``
and the rate ``k(F) = -r (dS/dF) / S``. For refolding the force decreases and
``S(F) = int_{-inf}^F p``, giving ``k(F) = +r (dS/dF) / S`` (Bell-Evans
framework; Evans and Ritchie, Biophys. J. 72, 1541 (1997)).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..stats.kde import kde_pdf, optimal_bandwidth


@dataclass
class RuptureStats:
    """Summary statistics of rupture forces.

    Attributes:
        mean (float): Mean force (pN).
        std (float): Standard deviation (pN).
        variance (float): Variance (pN^2).
        variance_err (float): Uncertainty of the variance, ``var sqrt(2/n)``.
        mode (float): Maximum of the kernel density estimate (pN).
        n (int): Number of forces.
    """

    mean: float
    std: float
    variance: float
    variance_err: float  # var * sqrt(2 / n)
    mode: float  # KDE maximum (the mode of a continuous sample is not defined)
    n: int


def rupture_statistics(forces, grid=None, bandwidth=None):
    """Return the statistics of a sample of rupture forces.

    Args:
        forces (array-like): Rupture forces (pN).
        grid (array-like | None): Grid for the mode; 400 points over the data
            range if None.
        bandwidth (float | None): KDE bandwidth; optimised if None.

    Returns:
        RuptureStats: Mean, standard deviation, variance, KDE mode and size.

    Raises:
        ValueError: If the sample has no spread.
    """
    f = np.asarray(forces, dtype=float)
    grid = np.linspace(f.min(), f.max(), 400) if grid is None else grid
    pdf = kde_pdf(f, grid, bandwidth)
    return RuptureStats(
        f.mean(),
        f.std(ddof=1),
        f.var(ddof=1),
        f.var(ddof=1) * np.sqrt(2 / f.size),
        float(grid[np.argmax(pdf)]),
        f.size,
    )


@dataclass
class RateCurves:
    """Density, survival probability and rate as functions of force.

    Attributes:
        force (numpy.ndarray): Force grid (pN).
        pdf (numpy.ndarray): Mean bootstrap density.
        pdf_err (numpy.ndarray): Bootstrap standard deviation of the density.
        survival (numpy.ndarray): Mean survival probability.
        survival_err (numpy.ndarray): Standard deviation of the survival.
        rate (numpy.ndarray): Rate ``k(F)`` (1/s); NaN where the survival is
            below ``s_min``.
        rate_err (numpy.ndarray): Standard deviation of the rate.
    """

    force: np.ndarray
    pdf: np.ndarray
    pdf_err: np.ndarray
    survival: np.ndarray
    survival_err: np.ndarray
    rate: (
        np.ndarray
    )  # k(F) (1/s); NaN where the survival probability is below ``s_min``
    rate_err: np.ndarray


def rupture_rates(
    forces,
    loading_rate,
    unfolding=True,
    grid=None,
    bandwidth=None,
    n_boot=200,
    s_min=0.02,
    rng=None,
):
    """Estimate density, survival and rate from rupture forces.

    The KDE bandwidth is optimised once on the data and reused for every
    bootstrap resample.

    Args:
        forces (array-like): Rupture forces (pN).
        loading_rate (float): Loading rate (pN/s).
        unfolding (bool): True for unfolding (force increasing), False for
            refolding (force decreasing).
        grid (array-like | None): Force grid; 300 points around the data if
            None.
        bandwidth (float | None): KDE bandwidth; optimised if None.
        n_boot (int): Bootstrap resamples.
        s_min (float): Survival below which the rate is not reported.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        RateCurves: Density, survival and rate with bootstrap errors.
    """
    f = np.asarray(forces, dtype=float)
    h = optimal_bandwidth(f) if bandwidth is None else bandwidth
    if grid is None:
        grid = np.linspace(f.min() - 2 * h, f.max() + 2 * h, 300)
    gen = np.random.default_rng(rng)
    # Cost is n_boot * (N * grid): subsample or bin the data for N > ~1e5.
    pdfs = np.array(
        [
            kde_pdf(f[gen.integers(0, f.size, f.size)], grid, h)
            for _ in range(n_boot)
        ]
    )
    cdf = cumulative_trapezoid(pdfs, grid, axis=1, initial=0.0)
    surv = 1.0 - cdf if unfolding else cdf
    sign = -1.0 if unfolding else 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = sign * loading_rate * np.gradient(surv, grid, axis=1) / surv
    rate = np.where(surv > s_min, rate, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore", RuntimeWarning
        )  # all-NaN columns in the tails
        m, s = np.nanmean, lambda a, axis: np.nanstd(a, axis=axis, ddof=1)
        return RateCurves(
            grid,
            m(pdfs, 0),
            s(pdfs, 0),
            m(surv, 0),
            s(surv, 0),
            m(rate, 0),
            s(rate, 0),
        )
