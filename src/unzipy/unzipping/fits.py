"""Fits of force-extension data to polymer models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..constants import KT25
from ..polymer import fjc_extension, wlc_extension


@dataclass
class FitResult:
    """Result of a least-squares fit.

    Attributes:
        params (dict): Fitted parameters by name.
        errors (dict): Standard errors by name.
        residual_rms (float): Root-mean-square residual (nm).
    """

    params: dict
    errors: dict
    residual_rms: float


def _fit(model, names, p0, f, x, bounds=(-np.inf, np.inf)):
    """Fit a model by non-linear least squares.

    Args:
        model (callable): ``model(params, f)`` returning the extension.
        names (list[str]): Parameter names.
        p0 (list[float]): Initial values.
        f (numpy.ndarray): Forces (pN).
        x (numpy.ndarray): Extensions (nm).
        bounds (tuple): Lower and upper bounds.

    Returns:
        FitResult: Parameters, errors and residual.

    Raises:
        ValueError: If there are not more data points than parameters.
    """
    if x.size <= len(p0):
        raise ValueError(
            f"need more than {len(p0)} data points for "
            f"{len(p0)} free parameters"
        )
    res = least_squares(
        lambda p: model(p, f) - x,
        p0,
        method="trf",
        bounds=bounds,
        x_scale="jac",
    )
    dof = max(x.size - len(p0), 1)
    s2 = np.sum(res.fun**2) / dof
    cov = np.linalg.pinv(res.jac.T @ res.jac) * s2
    return FitResult(
        dict(zip(names, res.x, strict=True)),
        dict(zip(names, np.sqrt(np.diag(cov)), strict=True)),
        float(np.sqrt(np.mean(res.fun**2))),
    )


def fit_ssdna_wlc(
    f, x, contour, k_trap=0.077, kT=KT25, p0=(0.0, 0.7), fit_contour=False
):
    """Fit a WLC model to single-strand force-extension data.

    The model is ``x = shift + L l_WLC(f; P) + f / k_trap``.

    Args:
        f (array-like): Force (pN).
        x (array-like): Extension (nm).
        contour (float): Contour length ``L`` (nm).
        k_trap (float): Trap stiffness (pN/nm).
        kT (float): Thermal energy (pN nm).
        p0 (tuple[float, float]): Initial ``shift`` (nm) and persistence length
            ``P`` (nm).
        fit_contour (bool): Also fit the contour length.

    Returns:
        FitResult: Fitted ``shift`` and ``P`` (and ``L``).
    """
    f, x = np.asarray(f, float), np.asarray(x, float)
    if fit_contour:
        return _fit(
            lambda p, f: p[0] + p[2] * wlc_extension(f, p[1], kT) + f / k_trap,
            ["shift", "P", "L"],
            [*p0, contour],
            f,
            x,
            ([-np.inf, 1e-3, 1e-3], np.inf),
        )
    return _fit(
        lambda p, f: p[0] + contour * wlc_extension(f, p[1], kT) + f / k_trap,
        ["shift", "P"],
        list(p0),
        f,
        x,
        ([-np.inf, 1e-3], np.inf),
    )


def fit_ssdna_fjc(f, x, contour, k_trap=0.077, kT=KT25, p0=(0.0, 1.5)):
    """Fit an FJC model to single-strand force-extension data.

    The model is ``x = shift + FJC(f; b, contour) + f / k_trap``.

    Args:
        f (array-like): Force (pN).
        x (array-like): Extension (nm).
        contour (float): Contour length (nm).
        k_trap (float): Trap stiffness (pN/nm).
        kT (float): Thermal energy (pN nm).
        p0 (tuple[float, float]): Initial ``shift`` (nm) and Kuhn length ``b``
            (nm).

    Returns:
        FitResult: Fitted ``shift`` and ``b``.
    """
    f, x = np.asarray(f, float), np.asarray(x, float)
    return _fit(
        lambda p, f: p[0] + fjc_extension(f, p[1], contour, kT) + f / k_trap,
        ["shift", "b"],
        list(p0),
        f,
        x,
        ([-np.inf, 1e-3], np.inf),
    )
