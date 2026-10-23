"""Free-energy estimators from nonequilibrium work values.

Convention: the works ``w_f`` (forward) and ``w_r`` (reverse) are the work done
*on* the system along the forward process 0 -> 1 and its time reversal 1 -> 0,
in units of kT. The free-energy difference is ``dG = G(1) - G(0)``, so ``<w_f>
>= dG >= -<w_r>``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp


def _check_work(w, name="work"):
    """Validate work values.

    Args:
        w (array-like): Work values.
        name (str): Name used in error messages.

    Returns:
        numpy.ndarray: The values as a float array.

    Raises:
        ValueError: If empty or containing NaN or infinite values.
    """
    w = np.asarray(w, dtype=float)
    if w.size == 0:
        raise ValueError(f"{name} values are empty")
    if not np.all(np.isfinite(w)):
        raise ValueError(f"{name} values contain NaN or inf")
    return w


def jarzynski(w, axis=-1):
    """Return the Jarzynski estimate ``-ln <exp(-w)>``.

    Computed with a log-sum-exp for numerical stability.

    Args:
        w (array-like): Work values (kT).
        axis (int): Axis holding the samples.

    Returns:
        float | numpy.ndarray: Free-energy estimate (kT).

    Raises:
        ValueError: If ``w`` is empty or not finite.
    """
    w = _check_work(w)
    return -(logsumexp(-w, axis=axis) - np.log(w.shape[axis]))


def jarzynski_reverse(w_r, axis=-1):
    """Return the forward ``dG`` from reverse-process work values.

    The estimate is ``+ln <exp(-w_r)>``.

    Args:
        w_r (array-like): Reverse work values (kT).
        axis (int): Axis holding the samples.

    Returns:
        float | numpy.ndarray: Free-energy estimate (kT).
    """
    return -jarzynski(w_r, axis=axis)


def _bar_arg(w_f, w_r, dg):
    """Return the Bennett arguments of the two work sets.

    Args:
        w_f (numpy.ndarray): Forward work values.
        w_r (numpy.ndarray): Reverse work values.
        dg (float): Free-energy difference.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Arguments of the Fermi
        function for forward and reverse samples.
    """
    m = np.log(w_f.size / w_r.size)
    return m + w_f - dg, m - w_r - dg


def bennett(w_f, w_r):
    """Return the Bennett acceptance ratio estimate of ``dG``.

    Solves ``sum_F fermi(M + w_f - dG) = sum_R fermi(-M + w_r + dG)`` with ``M
    = ln(nF / nR)`` by Brent's method, bracketing the root with the Jensen
    bounds ``-<w_r> <= dG <= <w_f>``.

    Args:
        w_f (array-like): Forward work values (kT).
        w_r (array-like): Reverse work values (kT).

    Returns:
        float: Free-energy difference (kT).

    Raises:
        ValueError: If a work set is empty or not finite.
    """
    w_f = _check_work(w_f, "forward work").ravel()
    w_r = _check_work(w_r, "reverse work").ravel()
    m = np.log(w_f.size / w_r.size)

    def g(dg):
        return expit(-(m + w_f - dg)).sum() - expit(-(-m + w_r + dg)).sum()

    lo, hi = -w_r.mean(), w_f.mean()  # Jensen bounds on Delta G
    if lo > hi:  # sampling noise; widen
        lo, hi = hi, lo
    pad = 1.0 + (hi - lo)
    lo, hi = lo - pad, hi + pad
    while g(lo) > 0:
        lo -= pad
    while g(hi) < 0:
        hi += pad
    return brentq(g, lo, hi, xtol=1e-12)


def bennett_error(w_f, w_r, dg=None):
    """Return the asymptotic standard error of the Bennett estimate.

    Uses the expression of Shirts et al., Phys. Rev. Lett. 91, 140601 (2003).

    Args:
        w_f (array-like): Forward work values (kT).
        w_r (array-like): Reverse work values (kT).
        dg (float | None): Bennett estimate; computed if None.

    Returns:
        float: Standard error (kT).
    """
    w_f = _check_work(w_f, "forward work").ravel()
    w_r = _check_work(w_r, "reverse work").ravel()
    if dg is None:
        dg = bennett(w_f, w_r)
    n_f, n_r = w_f.size, w_r.size
    n = n_f + n_r
    x_f, x_r = _bar_arg(w_f, w_r, dg)
    s = np.mean(1.0 / (2.0 + 2.0 * np.cosh(np.concatenate([x_f, x_r]))))
    return np.sqrt((1.0 / s - (n / n_f + n / n_r)) / n)


def matching(w_f, w_r, n_grid=200):
    """Return the Gaussian "matching" estimate of ``dG``.

    Intended for strongly irreversible pulling, where the forward and reverse
    work distributions do not overlap. Both distributions are modelled as
    Gaussians and ``dG`` is the value for which ``ln[P_F(W) / P_R(-W)] = W -
    dG`` best holds, i.e. the average of ``W - ln[P_F(W) / P_R(-W)]`` between
    the two means. For exactly Gaussian, CFT-consistent data it reduces to
    ``(mu_F + mu_R) / 2``. Severino et al., J. Stat. Mech. (2019) 124001,
    sec. 5.2, report matching 3-5 kT below Bennett for their CD4L12 data; see
    also Palassini and Ritort, Phys. Rev. Lett. 107, 060601.

    Args:
        w_f (array-like): Forward work values (kT).
        w_r (array-like): Reverse work values (kT).
        n_grid (int): Grid points between the two means.

    Returns:
        float: Free-energy difference (kT).

    Raises:
        ValueError: If a work set has fewer than two finite values.
    """
    w_f = _check_work(w_f, "forward work").ravel()
    w_r = -_check_work(
        w_r, "reverse work"
    ).ravel()  # P_R(-W): sign-flipped reverse work
    if w_f.size < 2 or w_r.size < 2:
        raise ValueError("matching needs at least 2 work values per direction")
    mu_f, s_f = w_f.mean(), w_f.std(ddof=1)
    mu_r, s_r = w_r.mean(), w_r.std(ddof=1)
    lo, hi = sorted((mu_f, mu_r))
    w = np.linspace(lo, hi, n_grid)
    ln_ratio = (
        -0.5 * ((w - mu_f) / s_f) ** 2
        - np.log(s_f)
        + 0.5 * ((w - mu_r) / s_r) ** 2
        + np.log(s_r)
    )
    return float(np.mean(w - ln_ratio))
