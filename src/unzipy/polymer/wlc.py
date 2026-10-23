"""Worm-like chain (WLC) and extensible WLC.

The WLC uses the Marko-Siggia interpolation, optionally with the Bouchiat
polynomial correction (Bouchiat et al., Biophys. J. 76, 409 (1999)).

Conventions: ``l = x / Lc`` is the relative extension, ``p`` the persistence
length (nm), ``f`` the force (pN); energies are in kT.
"""

from __future__ import annotations

import numpy as np

from ..constants import KT25

# Bouchiat et al., Biophys. J. 76, 409 (1999): coefficients of the
# polynomial correction (l^2 .. l^7).
_BOUCHIAT = np.array(
    [-0.5164228, -2.737418, 16.07497, -38.87607, 39.49944, -14.17718]
)
_POWERS = np.arange(2, 8)


def _check_l(l: np.ndarray) -> None:
    """Check that relative extensions lie in ``[0, 1)``.

    Args:
        l (numpy.ndarray): Relative extensions.

    Raises:
        ValueError: If any value is outside ``[0, 1)``.
    """
    if np.any((l < 0) | (l >= 1)):
        raise ValueError("relative extension must satisfy 0 <= x/Lc < 1")


def _bouchiat(l: np.ndarray) -> np.ndarray:
    """Evaluate the Bouchiat polynomial correction.

    Args:
        l (numpy.ndarray): Relative extensions.

    Returns:
        numpy.ndarray: Correction to ``f p / kT``.
    """
    return sum(a * l**k for a, k in zip(_BOUCHIAT, _POWERS, strict=True))


def wlc_force(l, p, kT=KT25, bouchiat=False):
    """Return the force of a WLC at a given relative extension.

    Args:
        l (array-like): Relative extension ``x / Lc``, in ``[0, 1)``.
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).
        bouchiat (bool): Add the Bouchiat polynomial correction.

    Returns:
        numpy.ndarray: Force in pN.

    Raises:
        ValueError: If ``l`` is outside ``[0, 1)``.
    """
    l = np.asarray(l, dtype=float)
    _check_l(l)
    u = 1.0 - l
    shape = (
        l * (2.0 - l) / (4.0 * u * u) + l
    )  # = 1/(4u^2) - 1/4 + l, stable for small l
    if bouchiat:
        shape = shape + _bouchiat(l)
    return kT / p * shape


def wlc_extension(f, p, kT=KT25, bouchiat=False):
    """Return the relative extension of a WLC at a given force.

    This is the inverse of :func:`wlc_force`. Without the Bouchiat correction
    the Marko-Siggia equation is a cubic in ``u = 1 - x / Lc`` with a single
    positive root, obtained in closed form (Cardano or trigonometric branch),
    started from the asymptote at very high force and polished with Newton
    steps. With the correction the Newton iteration starts from the plain
    solution and falls back to bisection where it does not converge.

    Args:
        f (array-like): Force (pN), non-negative.
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).
        bouchiat (bool): Include the Bouchiat correction.

    Returns:
        numpy.ndarray: Relative extension ``x / Lc`` in ``[0, 1)``.

    Raises:
        ValueError: If any force is negative.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < 0):
        raise ValueError("negative forces are not supported")
    fr = f * p / kT
    if bouchiat:
        return _invert_bouchiat(f, fr, p, kT)

    # u^3 - a u^2 - 1/4 = 0 with a = 3/4 - fr. Depressed cubic
    # t^3 + P t + Q = 0 with u = t + a/3.
    a = 0.75 - fr
    disc = a**3 / 108.0 + 1.0 / 64.0
    q = -(2.0 * a**3 / 27.0 + 0.25)
    with np.errstate(invalid="ignore"):
        s = np.sqrt(np.maximum(disc, 0.0))
        t_one = np.cbrt(-q / 2 + s) + np.cbrt(-q / 2 - s)
        absa = np.abs(a)
        arg = np.clip(
            -27.0 * q / (2.0 * np.maximum(absa, 1e-300) ** 3), -1.0, 1.0
        )
        t_three = 2.0 * absa / 3.0 * np.cos(np.arccos(arg) / 3.0)
    u = np.where(disc >= 0, t_one, t_three) + a / 3.0
    # the cubic loses precision (cancellation) at very high force: start
    # from the asymptote there
    u = np.where(fr > 50.0, 0.5 / np.sqrt(np.maximum(fr, 1e-300)), u)
    u = np.clip(u, 1e-16, 1.0)
    for _ in range(3):  # Newton polish on h(u) = (1-u^2)/(4u^2) + (1-u) - fr
        h = (1.0 - u * u) / (4.0 * u * u) + (1.0 - u) - fr
        u = np.clip(u + h / (0.5 / u**3 + 1.0), 1e-16, 1.0)
    return np.where(f == 0, 0.0, 1.0 - u)


def _invert_bouchiat(f, fr, p, kT):
    """Invert the corrected force-extension relation.

    Newton iteration from the plain Marko-Siggia solution (the polynomial is a
    small correction); points that fail to converge use bisection.

    Args:
        f (numpy.ndarray): Force (pN).
        fr (numpy.ndarray): Scaled force ``f p / kT``.
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Relative extension.
    """
    l = np.clip(wlc_extension(f, p, kT), 0.0, 1.0 - 1e-12)
    for _ in range(10):
        u = 1.0 - l
        g = _shape(l, True) - fr
        dg = 0.5 / u**3 + 1.0 + _dbouchiat(l)
        l = np.clip(l - g / dg, 0.0, 1.0 - 1e-15)
    resid = np.abs(_shape(l, True) - fr) <= 1e-10 * np.maximum(fr, 1.0)
    if not np.all(resid):
        l = np.where(resid, l, _invert_bisect(lambda x: _shape(x, True), fr))
    return np.where(fr == 0, 0.0, l)


def _shape(l, bouchiat):
    """Return the scaled force ``f p / kT`` of the WLC.

    Args:
        l (numpy.ndarray): Relative extension.
        bouchiat (bool): Include the Bouchiat correction.

    Returns:
        numpy.ndarray: Scaled force.
    """
    u = 1.0 - l
    s = l * (2.0 - l) / (4.0 * u * u) + l
    return s + _bouchiat(l) if bouchiat else s


def _invert_bisect(shape, target, iters=80):
    """Invert a monotonic scaled-force function by bisection.

    Args:
        shape (callable): Increasing function of the relative extension.
        target (array-like): Scaled forces to invert.
        iters (int): Number of bisection steps.

    Returns:
        numpy.ndarray: Relative extensions.
    """
    target = np.asarray(target, dtype=float)
    lo = np.zeros_like(target)
    hi = np.full_like(target, 1.0 - 1e-15)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        below = shape(mid) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return np.where(target == 0, 0.0, 0.5 * (lo + hi))


def _dbouchiat(l):
    """Return the derivative of the Bouchiat polynomial.

    Args:
        l (numpy.ndarray): Relative extensions.

    Returns:
        numpy.ndarray: ``d(correction) / dl``.
    """
    return sum(
        k * a * l ** (k - 1) for a, k in zip(_BOUCHIAT, _POWERS, strict=True)
    )


def _ibouchiat(l):
    """Return the integral of the Bouchiat polynomial from 0 to ``l``.

    Args:
        l (numpy.ndarray): Relative extensions.

    Returns:
        numpy.ndarray: Integral of the correction.
    """
    return sum(
        a * l ** (k + 1) / (k + 1)
        for a, k in zip(_BOUCHIAT, _POWERS, strict=True)
    )


def wlc_compliance(f, p, kT=KT25, bouchiat=False):
    """Return ``dl/df`` of a WLC at a given force.

    Multiply by the contour length to obtain ``dx/df``.

    Args:
        f (array-like): Force (pN).
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).
        bouchiat (bool): Include the Bouchiat correction.

    Returns:
        numpy.ndarray: ``dl/df`` in 1/pN.
    """
    l = wlc_extension(f, p, kT, bouchiat)
    slope = 1.0 + 0.5 / (1.0 - l) ** 3
    if bouchiat:
        slope = slope + _dbouchiat(l)
    return p / kT / slope


def wlc_energy(f, Lc, p, kT=KT25, bouchiat=False):
    """Return the elastic energy stored in a WLC at a given force.

    This is ``integral of f dx`` from zero extension, divided by ``kT``.

    Args:
        f (array-like): Force (pN).
        Lc (float): Contour length (nm).
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).
        bouchiat (bool): Include the Bouchiat correction.

    Returns:
        numpy.ndarray: Energy in kT.
    """
    l = wlc_extension(f, p, kT, bouchiat)
    u = 0.25 * (1.0 / (1.0 - l) - 1.0 - l + 2.0 * l * l)
    if bouchiat:
        u = u + _ibouchiat(l)
    return Lc / p * u


def wlc_gibbs(f, Lc, p, kT=KT25, bouchiat=False):
    """Return the force-ensemble free energy of a WLC.

    ``G = U - f x / kT``, equal to ``-integral of x df`` from zero force
    divided by ``kT``.

    Args:
        f (array-like): Force (pN).
        Lc (float): Contour length (nm).
        p (float): Persistence length (nm).
        kT (float): Thermal energy (pN nm).
        bouchiat (bool): Include the Bouchiat correction.

    Returns:
        numpy.ndarray: Free energy in kT (negative).
    """
    l = wlc_extension(f, p, kT, bouchiat)
    return wlc_energy(f, Lc, p, kT, bouchiat) - f * Lc * l / kT


def ewlc_extension(f, p, S, kT=KT25):
    """Return the relative extension of the additive extensible WLC.

    ``x / Lc = l_WLC(f) + f / S``.

    Args:
        f (array-like): Force (pN).
        p (float): Persistence length (nm).
        S (float): Stretch modulus (pN).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Relative extension (not bounded by 1).
    """
    return wlc_extension(f, p, kT) + np.asarray(f, dtype=float) / S


def ewlc_energy(f, Lc, p, S, kT=KT25):
    """Return the elastic energy of the additive extensible WLC.

    Args:
        f (array-like): Force (pN).
        Lc (float): Contour length (nm).
        p (float): Persistence length (nm).
        S (float): Stretch modulus (pN).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Energy in kT.
    """
    f = np.asarray(f, dtype=float)
    return wlc_energy(f, Lc, p, kT) + Lc * f * f / (2.0 * S * kT)


def ewlc_scaled_extension(f, p, Y, kT=KT25):
    """Return the relative extension of the scaled extensible WLC.

    The contour length is replaced by ``Lc (1 + f / Y)`` (Severino et al.,
    J. Stat. Mech. (2019) 124001), so that ``x / Lc = (1 + f / Y) l_WLC(f)``.
    :func:`ewlc_extension` is the additive variant.

    Args:
        f (array-like): Force (pN).
        p (float): Persistence length (nm).
        Y (float): Young modulus (pN).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Relative extension.
    """
    f = np.asarray(f, dtype=float)
    return (1.0 + f / Y) * wlc_extension(f, p, kT)


def ewlc_scaled_compliance(f, p, Y, kT=KT25):
    """Return ``d(x / Lc) / df`` of the scaled extensible WLC.

    Args:
        f (array-like): Force (pN).
        p (float): Persistence length (nm).
        Y (float): Young modulus (pN).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Compliance per unit contour length in 1/pN.
    """
    f = np.asarray(f, dtype=float)
    return wlc_extension(f, p, kT) / Y + (1.0 + f / Y) * wlc_compliance(
        f, p, kT
    )


def released_nucleotides(f, x, p, d0, kT=KT25):
    """Return the number of nucleotides that explain an extension.

    Solves ``x = n d0 l_WLC(f)`` for ``n`` in closed form.

    Args:
        f (array-like): Force (pN).
        x (array-like): Extension of the single strand (nm).
        p (float): Persistence length (nm).
        d0 (float): Contour length per nucleotide (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Number of nucleotides.
    """
    return np.asarray(x, dtype=float) / (d0 * wlc_extension(f, p, kT))
