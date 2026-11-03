"""Alignment of an experimental unzipping curve to a theoretical one.

The force signal of both curves is discretised into ``n_levels`` letters and
aligned globally with Needleman-Wunsch (scoring matrix ``10 - (a - b)^2 / 10``,
linear gap penalty; J. Mol. Biol. 48, 443 (1970)). The extension offset between
matched points is smoothed with a spline. An optional last-rip step adds the
shifts of the final high-force region.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import make_smoothing_spline

try:  # optional acceleration
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):
        """Do nothing: fall back to plain Python without numba.

        Args:
            *args: Decorated function, when used without arguments.
            **kwargs: Ignored decorator options.

        Returns:
            Callable: The function itself, or an identity decorator.
        """
        return args[0] if args and callable(args[0]) else (lambda f: f)


@njit(cache=True)
def _nw(
    a, b, gap, scale
):  # Note: O(n*m) time and memory; banded alignment if n > ~2e4
    """Align two level sequences by dynamic programming.

    Args:
        a (numpy.ndarray): First sequence (float levels).
        b (numpy.ndarray): Second sequence.
        gap (float): Linear gap penalty.
        scale (float): Divisor of the squared level difference in the score.

    Returns:
        tuple[float, numpy.ndarray, numpy.ndarray]: Total score and the index
        of each aligned column in ``a`` and ``b`` (``-1`` marks a gap).
    """
    n, m = a.size, b.size
    score = np.empty((n + 1, m + 1))
    move = np.zeros(
        (n + 1, m + 1), dtype=np.int8
    )  # 0 diag, 1 up (gap in b), 2 left (gap in a)
    for i in range(n + 1):
        score[i, 0] = -gap * i
        move[i, 0] = 1
    for j in range(m + 1):
        score[0, j] = -gap * j
        move[0, j] = 2
    move[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d = a[i - 1] - b[j - 1]
            diag = score[i - 1, j - 1] + 10.0 - d * d / scale
            up = score[i - 1, j] - gap
            left = score[i, j - 1] - gap
            if diag >= up and diag >= left:
                score[i, j], move[i, j] = diag, 0
            elif up >= left:
                score[i, j], move[i, j] = up, 1
            else:
                score[i, j], move[i, j] = left, 2
    ia = np.empty(n + m, dtype=np.int64)
    ib = np.empty(n + m, dtype=np.int64)
    k = 0
    i, j = n, m
    while i > 0 or j > 0:
        mv = move[i, j]
        if mv == 0:
            ia[k], ib[k] = i - 1, j - 1
            i -= 1
            j -= 1
        elif mv == 1:
            ia[k], ib[k] = i - 1, -1
            i -= 1
        else:
            ia[k], ib[k] = -1, j - 1
            j -= 1
        k += 1
    return score[n, m], ia[:k][::-1], ib[:k][::-1]


def discretise(f, lo, hi, n_levels=22):
    """Discretise forces into integer levels ``1..n_levels``.

    Args:
        f (array-like): Forces (pN).
        lo (float): Lower edge of the range.
        hi (float): Upper edge of the range.
        n_levels (int): Number of levels.

    Returns:
        numpy.ndarray: Level of each force (ceiling of the scaled force).
    """
    df = (hi - lo) / n_levels
    return np.clip(
        np.ceil((np.asarray(f, dtype=float) - lo) / df), 1, n_levels
    )


def needleman_wunsch(levels_a, levels_b, gap=50.0, scale=10.0):
    """Align two integer-level sequences globally.

    Args:
        levels_a (array-like): First sequence of levels.
        levels_b (array-like): Second sequence of levels.
        gap (float): Linear gap penalty.
        scale (float): Divisor of the squared level difference in the score.

    Returns:
        tuple[float, numpy.ndarray, numpy.ndarray]: Score and the aligned
        column indices in each sequence (``-1`` marks a gap).
    """
    return _nw(
        np.asarray(levels_a, float),
        np.asarray(levels_b, float),
        float(gap),
        float(scale),
    )


@dataclass
class Alignment:
    """Result of aligning two curves.

    Attributes:
        x_theory (numpy.ndarray): Extension of the matched theoretical points.
        x_shift (numpy.ndarray): Raw shift ``x_exp - x_theory`` at those
            points.
        score (float): Alignment score.
    """

    x_theory: np.ndarray  # extension of the matched theoretical points
    x_shift: np.ndarray  # raw shift x_exp - x_theory at those points
    score: float

    def shift(self, x, lam=(1 - 1e-4) / 1e-4):
        """Evaluate the smoothed extension shift.

        Args:
            x (array-like): Theoretical extensions where the shift is
                evaluated.
            lam (float | None): Smoothing penalty of
                :func:`scipy.interpolate.make_smoothing_spline`. The default
                ``(1 - p) / p`` with ``p = 1e-4`` gives a very smooth, nearly
                linear shift; None selects it by cross-validation.

        Returns:
            numpy.ndarray: Shift (nm) at ``x``.
        """
        xs, idx = np.unique(self.x_theory, return_index=True)
        spline = make_smoothing_spline(xs, self.x_shift[idx], lam=lam)
        return spline(np.asarray(x, dtype=float))


def last_rip_shift(
    x_exp, f_exp, x_theo, f_theo, x_end_exp, x_end_theo, f_threshold
):
    """Return the extension shifts of the final high-force region.

    Points beyond the aligned range with force above a threshold are kept in
    both curves. The experimental extensions are binned by the *theoretical*
    force levels (sorted ascending), and the shift is the mean experimental
    extension of each bin minus the theoretical extension of the level.

    Args:
        x_exp (numpy.ndarray): Experimental extension (nm).
        f_exp (numpy.ndarray): Experimental force (pN).
        x_theo (numpy.ndarray): Theoretical extension (nm).
        f_theo (numpy.ndarray): Theoretical force (pN).
        x_end_exp (float): End of the aligned range in the experiment.
        x_end_theo (float): End of the aligned range in the theory.
        f_threshold (float): Minimum force of the points used.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Theoretical extension and shift;
        empty bins are dropped.
    """
    t = (x_theo > x_end_theo) & (f_theo > f_threshold)
    e = (x_exp > x_end_exp) & (f_exp > f_threshold)
    if t.sum() < 2 or e.sum() < 1:
        return np.array([]), np.array([])
    order = np.argsort(f_theo[t])
    xt, ft = x_theo[t][order], f_theo[t][order]
    which = (
        np.searchsorted(ft, f_exp[e], side="right") - 1
    )  # bin i spans [ft[i], ft[i+1])
    xe = x_exp[e]
    keep, shift = [], []
    for i in range(ft.size):
        m = which == i
        if m.any():
            keep.append(xt[i])
            shift.append(xe[m].mean() - xt[i])
    return np.array(keep), np.array(shift)


def align_curves(
    x_exp,
    f_exp,
    x_theo,
    f_theo,
    x_start,
    x_end_exp,
    x_end_theo,
    n_levels=22,
    gap=50.0,
    last_rip=False,
    threshold_sigmas=1.8,
):
    """Align an experimental curve with a theoretical curve.

    Args:
        x_exp (array-like): Experimental extension (nm).
        f_exp (array-like): Experimental force (pN).
        x_theo (array-like): Theoretical extension (nm).
        f_theo (array-like): Theoretical force (pN).
        x_start (float): Start of the aligned range.
        x_end_exp (float): End of the range in the experimental curve.
        x_end_theo (float): End of the range in the theoretical curve.
        n_levels (int): Number of force levels.
        gap (float): Linear gap penalty.
        last_rip (bool): Add the shifts of the final high-force region beyond
            the aligned range (see :func:`last_rip_shift`).
        threshold_sigmas (float): The last-rip region is ``f > mean +
            threshold_sigmas * std`` of the aligned theoretical forces.

    Returns:
        Alignment: Matched points, raw shifts and score.

    Raises:
        ValueError: If a curve has no points in the range.
    """
    x_exp, f_exp = np.asarray(x_exp, float), np.asarray(f_exp, float)
    x_theo, f_theo = np.asarray(x_theo, float), np.asarray(f_theo, float)
    se = (x_exp >= x_start) & (x_exp < x_end_exp)
    st = (x_theo >= x_start) & (x_theo < x_end_theo)
    xe, fe, xt, ft = x_exp[se], f_exp[se], x_theo[st], f_theo[st]
    if xe.size == 0 or xt.size == 0:
        raise ValueError(
            "no points of one of the curves lie inside the alignment range"
        )
    lo = min(fe.min(), ft.min()) - 1e-4
    hi = max(fe.max(), ft.max()) + 1e-4
    score, ia, ib = needleman_wunsch(
        discretise(fe, lo, hi, n_levels), discretise(ft, lo, hi, n_levels), gap
    )
    both = (ia >= 0) & (ib >= 0)
    x_t, shift = xt[ib[both]], xe[ia[both]] - xt[ib[both]]
    if last_rip:
        lx, ls = last_rip_shift(
            x_exp,
            f_exp,
            x_theo,
            f_theo,
            x_end_exp,
            x_end_theo,
            ft.mean() + threshold_sigmas * ft.std(),
        )
        x_t, shift = np.concatenate([x_t, lx]), np.concatenate([shift, ls])
    return Alignment(x_t, shift, float(score))
