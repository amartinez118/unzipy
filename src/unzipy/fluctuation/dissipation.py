"""Mean dissipated work of a two-state system at constant loading rate.

With dimensionless rate ``r`` and force sensitivity (fragility) ``mu``, the
unfolding rate is ``k(t) ~ exp(mu t) cosh t`` and

    W(r, mu) = int dy sech^2(y) int_{x>y} dx
               exp[-(1/r) int_y^x exp(mu t) cosh t dt].

The exponent has the closed form ``[fact(x) - fact(y)] / (r (1 - mu^2))`` with
``fact(x) = exp(mu x) (mu cosh x - sinh x)``. The total dissipation of a
forward plus reverse cycle is ``W(r, mu) + W(r, -mu)``. This is eq. A9 of
Monge, Manosas, Ritort, Phys. Rev. E 98, 032146 (2018), where ``mu = (x_F -
x_U) / (x_F + x_U)`` is the fragility and ``r = x_m rate / (4 kT k_c)`` the
dimensionless loading rate (see :func:`dimensionless_rate`).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad

from ..constants import KT25


def _antiderivative(t, mu):
    """Return the antiderivative of ``exp(mu t) cosh t``.

    Args:
        t (numpy.ndarray): Integration variable.
        mu (float): Fragility, ``|mu| != 1``.

    Returns:
        numpy.ndarray: ``C(t)``.
    """
    return 0.5 * (
        np.exp((mu + 1) * t) / (mu + 1) + np.exp((mu - 1) * t) / (mu - 1)
    )


def dissipated_work(r, mu, bound=30.0):
    """Return the mean dissipated work of one unfolding ramp.

    Args:
        r (float): Dimensionless loading rate.
        mu (float): Fragility, ``|mu| != 1``.
        bound (float): Integration domain is ``[-bound, bound]``.

    Returns:
        float: Dissipated work in kT.

    Raises:
        ValueError: If ``|mu| = 1``, where the closed form is singular.
    """
    if abs(abs(mu) - 1.0) < 1e-9:
        raise ValueError("|mu| = 1 is a singular point of the closed form")

    def inner(y):
        # the integrand decays on the scale r / (exp(mu y) cosh y): split there
        # so that quad resolves it
        scale = r / (np.exp(mu * y) * np.cosh(y))
        pts = [y + k * scale for k in (1, 10) if y + k * scale < bound]
        with warnings.catch_warnings():
            warnings.simplefilter(
                "ignore", IntegrationWarning
            )  # tiny integrals, epsabs=0
            val, _ = quad(
                lambda x: np.exp(
                    -(_antiderivative(x, mu) - _antiderivative(y, mu)) / r
                ),
                y,
                bound,
                points=pts or None,
                limit=200,
                epsabs=0,
                epsrel=1e-7,
            )
        return val / np.cosh(y) ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        val, _ = quad(inner, -bound, bound, limit=200, epsabs=0, epsrel=1e-7)
    return val


def dissipated_work_cycle(r, mu, bound=30.0):
    """Return the dissipated work of a forward plus reverse cycle.

    Args:
        r (float): Dimensionless loading rate.
        mu (float): Fragility.
        bound (float): Integration domain is ``[-bound, bound]``.

    Returns:
        float: ``W(r, mu) + W(r, -mu)`` in kT.
    """
    return dissipated_work(r, mu, bound) + dissipated_work(r, -mu, bound)


def fragility(x_f, x_u):
    """Return the molecular fragility ``(x_F - x_U) / (x_F + x_U)``.

    Args:
        x_f (float): Distance from the folded state to the transition state.
        x_u (float): Distance from the unfolded state to the transition state.

    Returns:
        float: Fragility, between -1 and 1.
    """
    return (x_f - x_u) / (x_f + x_u)


def dimensionless_rate(loading_rate, x_m, k_c, kT=KT25):
    """Return the dimensionless loading rate ``x_m r / (4 kT k_c)``.

    Args:
        loading_rate (float): Loading rate (pN/s).
        x_m (float): Molecular extension (nm).
        k_c (float): Transition rate at the coexistence force (1/s).
        kT (float): Thermal energy (pN nm).

    Returns:
        float: Dimensionless rate.
    """
    return x_m * loading_rate / (4.0 * kT * k_c)


def ensemble_rate_ratio(mu, x_m, delta_f, kT=KT25):
    """Return the force- to extension-ensemble rate ratio ``Omega``.

    ``Omega = exp[-(1 - mu^2) x_m |delta_f| / (8 kT)]`` (Monge et al., Phys.
    Rev. E 98, 032146, eq. 7). Kinetic rates at coexistence are slower at
    controlled force than at controlled extension.

    Args:
        mu (float): Fragility.
        x_m (float): Molecular extension (nm).
        delta_f (float): Force jump of an unfolding event in the extension
            ensemble (pN).
        kT (float): Thermal energy (pN nm).

    Returns:
        float: ``k_force / k_extension``.
    """
    return float(np.exp(-(1.0 - mu**2) / 8.0 * x_m * abs(delta_f) / kT))
