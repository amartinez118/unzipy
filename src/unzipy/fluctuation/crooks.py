"""Crooks fluctuation theorem: ``ln[P_F(W) / P_R(-W)] = W - dG`` (kT)."""

from __future__ import annotations

import numpy as np

from ..stats.bootstrap import bootstrap_pdf


def crooks_logratio(w_f, w_r, bins=None, n_boot=500, rng=None):
    """Return the log-ratio of forward and sign-flipped reverse work.

    Both densities are estimated on one shared grid spanning the overlap of the
    two supports, with bootstrap errors.

    Args:
        w_f (array-like): Forward work values (kT).
        w_r (array-like): Reverse work values (kT); they are sign-flipped to
            compare ``P_F(W)`` with ``P_R(-W)``.
        bins (int | array-like | None): Number of bins or bin edges. The Rice
            rule is used if None.
        n_boot (int): Bootstrap resamples.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Work ``w``, the
        log-ratio and its propagated bootstrap error, for the bins where both
        densities are positive.

    Raises:
        ValueError: If the two distributions do not overlap.
    """
    w_f = np.asarray(w_f, dtype=float)
    w_r = -np.asarray(w_r, dtype=float)
    lo, hi = max(w_f.min(), w_r.min()), min(w_f.max(), w_r.max())
    if lo >= hi:
        raise ValueError(
            "forward and reverse work distributions do not overlap"
        )
    if bins is None:
        bins = int(np.ceil(2 * min(w_f.size, w_r.size) ** (1 / 3)))
    edges = (
        np.linspace(lo, hi, bins + 1)
        if np.ndim(bins) == 0
        else np.asarray(bins)
    )
    gen = np.random.default_rng(rng)
    w, pf, ef = bootstrap_pdf(w_f, n_boot, edges, rng=gen)
    _, pr, er = bootstrap_pdf(w_r, n_boot, edges, rng=gen)
    ok = (pf > 0) & (pr > 0)
    return (
        w[ok],
        np.log(pf[ok] / pr[ok]),
        np.hypot(ef[ok] / pf[ok], er[ok] / pr[ok]),
    )


def crooks_crossing(w, logratio, err=None):
    """Fit ``logratio = s (W - dG)`` and return the slope and ``dG``.

    The theorem predicts ``s = 1``; the crossing point estimates ``dG``.

    Args:
        w (array-like): Work values (kT).
        logratio (array-like): Log-ratio from :func:`crooks_logratio`.
        err (array-like | None): Errors used as weights ``1 / err``.

    Returns:
        tuple[float, float]: Slope and crossing point ``dG`` (kT).
    """
    sigma = None if err is None else np.asarray(err)
    slope, intercept = np.polyfit(
        w, logratio, 1, w=None if sigma is None else 1.0 / sigma
    )
    return slope, -intercept / slope
