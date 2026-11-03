"""Bias of the Jarzynski estimator and an improved estimator.

Implements Palassini and Ritort, Phys. Rev. Lett. 107, 060601 (2011). Work
values are in kT. The lower tail of the work distribution is modelled as
(their eq. 2)

    p(W) ~ q Omega^(alpha-1) |W - Wc|^(-alpha)
           exp(-|W - Wc|^delta / Omega^delta),     W << Wc,

with ``delta > 1`` (``delta = 2`` is Gaussian), and

    mu = (delta - 1) (Omega / delta)^(delta / (delta - 1)),
    lambda = (delta - 1) ln N / mu.

``mu`` equals the dissipated work ``Dc = Wc - dF`` in the Jarzynski consistent
limit, which the improved estimator assumes. :func:`bias_rem` is the scaling
limit (eq. 4), :func:`bias_extreme` (eq. 5) is valid for ``lambda << 1`` and
:func:`bias_critical` (eq. 6) for ``lambda <~ 1``. The closed expression for
``lambda > 1`` is in the supplement of the paper and is not implemented:
:func:`bias` switches to the Gaussian power law (eq. 7) above ``lambda = 1.3``
(eq. 6 turns negative there, which is impossible since the Jarzynski estimator
never underestimates on average) and always returns a non-negative bias.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc, gamma
from scipy.stats import gaussian_kde

from .estimators import jarzynski

EULER_GAMMA = 0.5772156649015329


def mu_from_omega(omega, delta=2.0):
    """Return the tail parameter ``mu`` from the tail width ``Omega``.

    Args:
        omega (float): Tail width ``Omega``.
        delta (float): Tail exponent (``delta > 1``).

    Returns:
        float: ``mu`` in kT.
    """
    return (delta - 1.0) * (omega / delta) ** (delta / (delta - 1.0))


def omega_from_mu(mu, delta=2.0):
    """Return the tail width ``Omega`` from the tail parameter ``mu``.

    Args:
        mu (float): Tail parameter (kT).
        delta (float): Tail exponent (``delta > 1``).

    Returns:
        float: ``Omega``.
    """
    return delta * (mu / (delta - 1.0)) ** ((delta - 1.0) / delta)


def scaling_lambda(n, mu, delta=2.0):
    """Return ``lambda = (delta - 1) ln N / mu``.

    Args:
        n (int): Number of work values.
        mu (float): Tail parameter (kT).
        delta (float): Tail exponent.

    Returns:
        float: The scaling variable ``lambda``.
    """
    return (delta - 1.0) * np.log(n) / mu


def bias_rem(n, mu, d_c=None, delta=2.0):
    """Return the scaling-limit bias (eq. 4).

    Args:
        n (int): Number of work values.
        mu (float): Tail parameter (kT).
        d_c (float | None): Characteristic dissipation ``Dc``; ``mu`` if None.
        delta (float): Tail exponent.

    Returns:
        float: Bias in kT.
    """
    d_c = mu if d_c is None else d_c
    lam = scaling_lambda(n, mu, delta)
    if lam > 1.0:
        return d_c - mu
    return d_c + mu * (lam - delta * lam ** (1.0 / delta)) / (delta - 1.0)


def bias_extreme(n, mu, delta=2.0, alpha=0.0, q=None, d_c=None):
    """Return the mean bias for ``lambda << 1`` (eq. 5).

    Args:
        n (int): Number of work values.
        mu (float): Tail parameter (kT).
        delta (float): Tail exponent.
        alpha (float): Power-law exponent of the tail prefactor.
        q (float | None): Tail normalisation; required unless ``delta = 2`` and
            ``alpha = 0`` (Gaussian, ``q = pi**-0.5``).
        d_c (float | None): Characteristic dissipation; ``mu`` if None.

    Returns:
        float: Bias in kT (may be negative outside the range of validity).

    Raises:
        ValueError: If ``q`` is needed and missing.
    """
    q = _default_q(q, delta, alpha)
    lam = scaling_lambda(n, mu, delta)
    bracket = (
        EULER_GAMMA
        + (1.0 - alpha - delta) / delta * np.log(np.log(n))
        + np.log(q / delta)
    )
    return (
        bias_rem(n, mu, d_c, delta) - lam ** ((1.0 - delta) / delta) * bracket
    )


def bias_critical(n, mu, delta=2.0, alpha=0.0, q=None, d_c=None):
    """Return the mean bias for ``lambda <~ 1`` (eq. 6).

    For a Gaussian distribution it equals ``ln 2`` at ``lambda = 1``.

    Args:
        n (int): Number of work values.
        mu (float): Tail parameter (kT).
        delta (float): Tail exponent.
        alpha (float): Power-law exponent of the tail prefactor.
        q (float | None): Tail normalisation; required unless ``delta = 2`` and
            ``alpha = 0`` (Gaussian, ``q = pi**-0.5``).
        d_c (float | None): Characteristic dissipation; ``mu`` if None.

    Returns:
        float: Bias in kT (may be negative outside the range of validity).

    Raises:
        ValueError: If ``q`` is needed and missing.
    """
    q = _default_q(q, delta, alpha)
    lam = scaling_lambda(n, mu, delta)
    p = lam ** ((1.0 - delta) / delta)
    theta = (
        (p - 1.0) * np.sqrt(delta * np.log(n)) / np.sqrt(2.0 * (delta - 1.0))
    )
    bracket = (
        EULER_GAMMA
        + (1.0 - alpha - delta / 2.0) / delta * np.log(np.log(n))
        + 0.5 * np.log(np.pi * q * q / (2.0 * delta * (delta - 1.0)))
        + theta**2
        + np.log(erfc(theta))
    )
    return bias_rem(n, mu, d_c, delta) + EULER_GAMMA - p * bracket


def bias_power_law(n, d_mean):
    """Return the empirical power-law bias for Gaussian work (eq. 7).

    ``<B_N> = <D> N**(-z)`` with ``z = -ln(ln 2 / <D>) / <D>``. The exponent
    is positive, so that the bias decreases with ``N``, only for ``<D> > ln
    2``; below that the fit does not apply (the bias there is already small
    and the formula would grow with ``N``).

    Args:
        n (int): Number of work values.
        d_mean (float): Mean dissipated work ``<D>`` (kT).

    Returns:
        float: Bias in kT.

    Raises:
        ValueError: If ``d_mean <= ln 2``, where the power law is invalid.
    """
    if d_mean <= np.log(2.0):
        raise ValueError(
            f"the power law of eq. 7 needs <D> > ln 2 (got {d_mean:.3g} kT)"
        )
    z = -np.log(np.log(2.0) / d_mean) / d_mean
    return d_mean * n ** (-z)


LAMBDA_EXTREME = (
    0.45  # below: eq. 5 is the more accurate of eqs. 5 and 6 (vs. brute force)
)
LAMBDA_CRITICAL = 1.3  # above: eq. 6 is no longer reliable


def bias(n, mu, delta=2.0, alpha=0.0, q=None, d_c=None):
    """Return the expected bias of the Jarzynski estimator.

    The expression is chosen by regime: eq. 5 for ``lambda < 0.45``, eq. 6 up
    to ``lambda = 1.3``, and beyond that the empirical power law of eq. 7 for
    ``delta = 2`` and ``mu > ln 2`` (with ``<D> = mu``), the scaling-limit
    bias of eq. 4 for ``delta = 2`` and ``mu <= ln 2`` (where the power law
    does not apply), or eq. 6 otherwise. The result is limited to the range
    from zero to the characteristic dissipation, the bias of a single work
    value.

    Args:
        n (int): Number of work values.
        mu (float): Tail parameter (kT).
        delta (float): Tail exponent.
        alpha (float): Power-law exponent of the tail prefactor.
        q (float | None): Tail normalisation; required unless ``delta = 2`` and
            ``alpha = 0``.
        d_c (float | None): Characteristic dissipation; ``mu`` if None.

    Returns:
        float: Bias in kT, between 0 and ``d_c``.
    """
    lam = float(scaling_lambda(n, mu, delta))
    d_c_val = mu if d_c is None else d_c
    if lam < LAMBDA_EXTREME:
        b = bias_extreme(n, mu, delta, alpha, q, d_c)
    elif lam <= LAMBDA_CRITICAL or delta != 2.0:
        b = bias_critical(n, mu, delta, alpha, q, d_c)
    elif mu > np.log(2.0):
        b = bias_power_law(n, mu)
    else:  # converged regime with small dissipation: scaling-limit bias
        b = bias_rem(n, mu, d_c, delta)
    return float(min(max(b, 0.0), max(d_c_val, 0.0)))


def gaussian_bias(n, variance, method="auto"):
    """Return the expected Jarzynski bias for Gaussian work values.

    For Gaussian work ``mu = Dc = <D> = variance / 2``, ``delta = 2``, ``alpha
    = 0`` and ``q = pi**-0.5``.

    Args:
        n (int): Number of work values.
        variance (float): Variance of the work distribution (kT^2).
        method (str): ``"auto"`` (see :func:`bias`), ``"critical"`` (eq. 6),
            ``"extreme"`` (eq. 5) or ``"power"`` (eq. 7).

    Returns:
        float: Bias in kT.
    """
    mu = 0.5 * variance
    if method == "power":
        return bias_power_law(n, mu)
    if method == "auto":
        return bias(n, mu, 2.0, 0.0, np.pi**-0.5)
    fn = {"critical": bias_critical, "extreme": bias_extreme}[method]
    return fn(n, mu, 2.0, 0.0, np.pi**-0.5)


def _default_q(q, delta, alpha):
    """Return the tail normalisation, defaulting to the Gaussian value.

    Args:
        q (float | None): Given normalisation.
        delta (float): Tail exponent.
        alpha (float): Power-law exponent of the tail prefactor.

    Returns:
        float: ``q``.

    Raises:
        ValueError: If ``q`` is None and the tail is not Gaussian.
    """
    if q is not None:
        return q
    if delta == 2.0 and alpha == 0.0:
        return np.pi**-0.5  # Gaussian tail
    raise ValueError(
        "q (tail normalisation) is required unless delta = 2 and alpha = 0"
    )


@dataclass
class TailFit:
    """Parameters of a fitted lower tail.

    Attributes:
        w_c (float): Characteristic work ``Wc`` (maximum of the distribution).
        omega (float): Tail width ``Omega``.
        q (float): Tail normalisation.
        delta (float): Tail exponent (fixed in the fit).
        mu (float): Tail parameter ``mu`` (kT).
    """

    w_c: float
    omega: float
    q: float
    delta: float
    mu: float


def fit_lower_tail(w, delta=2.0, w_c=None):
    """Fit the lower tail of a work distribution.

    The tail ``W <= Wc`` is fitted to eq. 2 with ``alpha = 0`` (as the paper
    suggests) at fixed ``delta`` by maximum likelihood, without histogram
    binning, so it is stable for small samples. With ``x = Wc - W >= 0`` the
    tail is a generalised half-normal: ``Omega^delta = delta <x^delta>`` and
    ``q = P(W <= Wc) / Gamma(1 + 1/delta)`` (``pi**-0.5`` for a Gaussian with
    ``Wc`` at its mean).

    Args:
        w (array-like): Work values (kT).
        delta (float): Tail exponent; 2 for near-Gaussian tails.
        w_c (float | None): Characteristic work. Defaults to the maximum of the
            kernel-smoothed distribution.

    Returns:
        TailFit: Fitted parameters.

    Raises:
        ValueError: If fewer than 3 values lie below ``w_c``.
    """
    w = np.asarray(w, dtype=float)
    if w_c is None:
        grid = np.linspace(w.min(), w.max(), 512)
        w_c = float(grid[np.argmax(gaussian_kde(w)(grid))])
    x = w_c - w[w <= w_c]
    if x.size < 3:
        raise ValueError("too few work values below W_c to fit the tail")
    omega = float((delta * np.mean(x**delta)) ** (1.0 / delta))
    q = float(x.size / w.size / gamma(1.0 + 1.0 / delta))
    return TailFit(float(w_c), omega, q, delta, mu_from_omega(omega, delta))


@dataclass
class CorrectedEstimate:
    """Result of the bias-corrected Jarzynski estimator.

    Attributes:
        dF_N (float): Plain Jarzynski estimate (kT).
        bias (float): Estimated bias (kT).
        dF_star (float): Corrected estimate ``dF_N - bias`` (kT).
        lam (float): Scaling variable ``lambda``.
        tail (TailFit): Fitted tail.
    """

    dF_N: float  # plain Jarzynski estimate
    bias: float
    dF_star: float  # bias-corrected estimate
    lam: float
    tail: TailFit


def jarzynski_corrected(w, delta=2.0, w_c=None, method="auto"):
    """Return the bias-corrected unidirectional free-energy estimate.

    ``dF*_N = dF_N - <B_N>``. The tail parameters are fitted by maximum
    likelihood at fixed ``delta`` (2 for near-Gaussian tails; the result
    depends strongly on it for non-Gaussian work) and ``Dc`` is set to ``mu``.
    The tail fit uses the spread of all values below ``Wc``, so a single
    non-physical work (for example from a trajectory that does not span the
    integration range) inflates ``mu`` and the bias; remove such outliers
    first.

    Args:
        w (array-like): Work values (kT) of one direction.
        delta (float): Tail exponent.
        w_c (float | None): Characteristic work; see :func:`fit_lower_tail`.
        method (str): ``"auto"`` (see :func:`bias`), ``"critical"`` (eq. 6) or
            ``"extreme"`` (eq. 5); the last two are applied as they are and can
            be negative outside their range.

    Returns:
        CorrectedEstimate: The plain and corrected estimates.
    """
    w = np.asarray(w, dtype=float)
    tail = fit_lower_tail(w, delta, w_c)
    float(scaling_lambda(w.size, tail.mu, delta))
    if method == "auto":
        b = bias(w.size, tail.mu, delta, 0.0, tail.q)
    else:
        fn = {"critical": bias_critical, "extreme": bias_extreme}[method]
        b = float(fn(w.size, tail.mu, delta, 0.0, tail.q))
    d_n = float(jarzynski(w))
    return CorrectedEstimate(
        d_n, b, d_n - b, float(scaling_lambda(w.size, tail.mu, delta)), tail
    )
