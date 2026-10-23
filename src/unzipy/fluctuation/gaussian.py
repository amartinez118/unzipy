"""Gaussian-approximation relations between work and free energy."""

from __future__ import annotations

import numpy as np


def effective_beta(var_g, var_w, err_var_g=0.0, err_var_w=0.0):
    """Return the effective inverse temperature of a Gaussian model.

    ``beta_eff = var_w / (var_w + var_g)`` (in units of ``beta``) relates the
    width of the work distributions to the variance of the free energy across
    molecules. Errors are propagated with ``d beta/d var_g = -var_w / (var_w +
    var_g)^2`` and ``d beta/d var_w = var_g / (var_w + var_g)^2``.

    Args:
        var_g (float): Variance of the free energy across molecules (kT^2).
        var_w (float): Variance of the work distributions (kT^2).
        err_var_g (float): Uncertainty of ``var_g``.
        err_var_w (float): Uncertainty of ``var_w``.

    Returns:
        tuple[float, float, float]: ``beta_eff``, its uncertainty and the
        first-order expansion ``(var_w/var_g)(1 - var_w/var_g)``.

    Raises:
        ValueError: If a variance is not positive.
    """
    if var_g <= 0 or var_w <= 0:
        raise ValueError("variances must be positive")
    s = var_w + var_g
    beta = var_w / s
    err = np.hypot(var_w / s**2 * err_var_g, var_g / s**2 * err_var_w)
    ratio = var_w / var_g
    return beta, err, ratio * (1.0 - ratio)
