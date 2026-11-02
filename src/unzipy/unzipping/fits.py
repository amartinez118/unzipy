"""Fits of force-extension data to single-strand polymer models.

Convenience wrappers around the general fitters of :mod:`unzipy.fitting`:
the strand is in series with an optical trap of known stiffness.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import KT25
from ..fitting import Model, fit, fjc, trap, wlc


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


def _strand_fit(element, label, f, x, contour, k_trap, kT, shift0, p0, fit_L):
    """Fit one strand element in series with a trap.

    Args:
        element (callable): Element factory (``wlc`` or ``fjc``).
        label (str): Name of the strand length parameter in the result
            (``"P"`` or ``"b"``).
        f (array-like): Force (pN).
        x (array-like): Extension (nm).
        contour (float): Contour length (nm).
        k_trap (float): Trap stiffness (pN/nm).
        kT (float): Thermal energy (pN nm).
        shift0 (float): Initial shift (nm).
        p0 (float): Initial persistence or Kuhn length (nm).
        fit_L (bool): Also fit the contour length.

    Returns:
        FitResult: Parameters, errors and residual.
    """
    model = Model(element("s"), trap("t"), kT=kT)
    name = "p" if label == "P" else "b"
    fixed = {"t.k0": k_trap}
    start = {"shift": shift0, f"s.{name}": p0, "s.L": contour}
    if not fit_L:
        fixed["s.L"] = contour
    report = fit(model, f, x, start=start, fixed=fixed)
    params = {
        "shift": report.values["shift"],
        label: report.values[f"s.{name}"],
    }
    errors = {
        "shift": report.errors["shift"],
        label: report.errors[f"s.{name}"],
    }
    if fit_L:
        params["L"], errors["L"] = report.values["s.L"], report.errors["s.L"]
    return FitResult(params, errors, report.rms)


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

    Raises:
        ValueError: If there are not more data points than parameters.
    """
    return _strand_fit(
        wlc, "P", f, x, contour, k_trap, kT, p0[0], p0[1], fit_contour
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

    Raises:
        ValueError: If there are not more data points than parameters.
    """
    return _strand_fit(
        fjc, "b", f, x, contour, k_trap, kT, p0[0], p0[1], False
    )
