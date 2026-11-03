"""Rupture and refolding event detection, and effective stiffness.

The rip of a trajectory is located as the window with the largest linear-fit
residual plus the largest step of an undecimated Haar detector; force and
extension before and after the jump are averaged over a few points.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from ..constants import KT25
from ..polymer import fjc_extension


@dataclass
class Event:
    """A rip found in a force-extension trajectory.

    Attributes:
        kind (str): ``"u"`` (unfolding) or ``"f"`` (folding).
        force (float): Force just before the jump (pN): the rupture force for
            unfolding, the folding force for folding.
        extension (float): Extension at the jump (nm).
        jump (float): Absolute force change across the event (pN).
        index (int): Sample index of the last point before the jump.
    """

    kind: str  # "u" (unfolding) or "f" (folding)
    force: float  # force just before the jump (rupture / folding force)
    extension: float
    jump: float  # |force change| across the event (pN)
    index: int  # sample index of the last point before the jump


def _step_signal(a, n):
    """Return a step detector (undecimated Haar) of a signal.

    At sample ``i`` it is the mean of the ``n`` samples up to and including
    ``i`` minus the mean of the next ``n``; unlike the decimated transform it
    also sees steps that fall inside a sample pair.

    Args:
        a (numpy.ndarray): Signal.
        n (int): Averaging length.

    Returns:
        numpy.ndarray: Detector, NaN where the neighbourhood is incomplete.
    """
    c = np.concatenate([[0.0], np.cumsum(a)])
    out = np.full(a.size, np.nan)
    i = np.arange(n - 1, a.size - n)
    out[i] = (c[i + 1] - c[i + 1 - n]) / n - (c[i + 1 + n] - c[i + 1]) / n
    return out


def find_event(x, f, kind, window=15.0, n_avg=4, jump_range=(0.85, 2.0)):
    """Locate the single rip of a trajectory.

    The window (non-overlapping) with the largest linear-fit residual contains
    the rip; a residual criterion is robust for shallow force ramps, where an
    R^2 criterion is not. Inside it the largest step (mean of ``n_avg`` points
    before minus after) gives the position. If the jump lies outside
    ``jump_range`` the max - min of the window is used instead.

    Args:
        x (array-like): Extension (nm).
        f (array-like): Force (pN).
        kind (str): ``"u"`` if the force drops at the rip, ``"f"`` if it jumps
            up.
        window (float): Window width (nm).
        n_avg (int): Points averaged on each side of the jump.
        jump_range (tuple[float, float]): Accepted range of the jump (pN).

    Returns:
        Event: The detected event.

    Raises:
        ValueError: If ``kind`` is invalid or the trajectory is too short.
    """
    if kind not in ("u", "f"):
        raise ValueError("kind must be 'u' (unfolding) or 'f' (folding)")
    x, f = np.asarray(x, float), np.asarray(f, float)
    edges = np.arange(x.min(), x.max() + window, window)
    best, best_misfit = None, -np.inf
    for lo, hi in itertools.pairwise(edges):
        sel = np.flatnonzero((x >= lo) & (x <= hi))
        if sel.size < 6:
            continue
        p = np.polyfit(x[sel], f[sel], 1)
        ss_res = np.sum((f[sel] - np.polyval(p, x[sel])) ** 2)
        misfit = (
            ss_res / sel.size
        )  # a rip is a step, which a line fits worst in absolute terms
        if misfit > best_misfit:
            best, best_misfit = sel, misfit
    if best is None:
        raise ValueError("trajectory too short to locate an event")
    d = _step_signal(f[best], n_avg)
    j = int(
        np.nanargmax(d) if kind == "u" else np.nanargmin(d)
    )  # last point before the jump
    before = best[max(j - n_avg + 1, 0) : j + 1]
    after = best[j + 1 : j + 1 + n_avg]
    f_before, f_after = f[before].mean(), f[after].mean()
    jump = abs(f_before - f_after)
    if not (jump_range[0] <= jump <= jump_range[1]):
        jump = f[best].max() - f[best].min()
        f_before = f[best].max() if kind == "u" else f[best].min()
    return Event(
        kind,
        float(f_before),
        float(x[before].mean()),
        float(jump),
        int(best[j]),
    )


def released_extension(jump, k_eff, f_after, hairpin_d=2.0, kT=KT25):
    """Return the extension of the released single strand.

    ``x_ss = jump / k_eff + x_dipole(f_after)``, from the force jump at the
    stiffness of the setup.

    Args:
        jump (float): Force jump across the event (pN).
        k_eff (float): Effective stiffness (pN/nm).
        f_after (float): Force after the jump (pN).
        hairpin_d (float): Hairpin diameter (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        float: Extension in nm.
    """
    return jump / k_eff + float(
        fjc_extension(f_after, hairpin_d, hairpin_d, kT)
    )


def effective_stiffness(x, f, x_range=None):
    """Fit the stiffness of a force-extension branch.

    Args:
        x (array-like): Extension (nm).
        f (array-like): Force (pN).
        x_range (tuple[float, float] | None): Fit only ``x`` in this range.

    Returns:
        tuple[float, float]: Slope ``k_eff`` (pN/nm) and its standard error.

    Raises:
        ValueError: If fewer than 3 points are available.
    """
    x, f = np.asarray(x, float), np.asarray(f, float)
    if x.size < 3:
        raise ValueError(
            "need at least 3 points for a slope with an error estimate"
        )
    if x_range is not None:
        keep = (x >= x_range[0]) & (x <= x_range[1])
        x, f = x[keep], f[keep]
    p, cov = np.polyfit(x, f, 1, cov=True)
    return float(p[0]), float(np.sqrt(cov[0, 0]))


@dataclass
class Stiffness:
    """Effective stiffness measured just before the rips.

    Attributes:
        k (float): Median slope of the folded branch (pN/nm).
        err (float): Bootstrap standard error of the median (pN/nm).
        n (int): Number of trajectories used.
        slopes (numpy.ndarray): Slope of each trajectory (pN/nm).
    """

    k: float
    err: float
    n: int
    slopes: np.ndarray


def stiffness_before_rip(
    trajectories,
    rips,
    window=10.0,
    guard=1.0,
    min_points=25,
    n_boot=300,
    rng=None,
):
    """Measure the effective stiffness as the slope just before the rip.

    The effective stiffness of the setup (trap and handles in series) that
    enters ``contributions(..., handles="spring")`` is the slope of the
    force-extension curve of the folded branch at the force of the rip, where
    the handles are stiffer than at low force. For each unfolding trajectory
    the force is fitted linearly against the extension over the ``window`` nm
    that end ``guard`` nm before its rip; the stiffness is the median over
    trajectories.

    The result is statistically demanding: the handle energy of the
    effective-stiffness method is hundreds of kT, so a 1% error of the
    stiffness shifts the free energy by several kT. Check the error and the
    dependence on ``window``.

    Args:
        trajectories (list[tuple]): Unfolding trajectories as ``(x, f)``
            pairs of extension (nm) and force (pN) in time order.
        rips (list[Event | float]): The rip of each trajectory, as an
            :class:`Event` or its extension (nm).
        window (float): Length of the fitted range (nm).
        guard (float): Distance between the end of the range and the rip
            (nm), which keeps the fit clear of the jump.
        min_points (int): Minimum samples in the range; trajectories with
            fewer are skipped.
        n_boot (int): Bootstrap resamples of the trajectories.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        Stiffness: Median slope, its error and the per-trajectory slopes.

    Raises:
        ValueError: If the inputs differ in length, ``window`` or ``guard`` is
            not positive, or no trajectory has enough points.
    """
    if len(trajectories) != len(rips):
        raise ValueError("one rip per trajectory is required")
    if window <= 0 or guard < 0:
        raise ValueError("window must be positive and guard non-negative")
    slopes = []
    for (x, f), rip in zip(trajectories, rips, strict=True):
        x, f = np.asarray(x, float), np.asarray(f, float)
        x_rip = rip.extension if isinstance(rip, Event) else float(rip)
        sel = (x >= x_rip - guard - window) & (x <= x_rip - guard)
        if sel.sum() >= min_points and np.ptp(x[sel]) > 0:
            slopes.append(np.polyfit(x[sel], f[sel], 1)[0])
    if not slopes:
        raise ValueError("no trajectory has enough points before its rip")
    slopes = np.array(slopes)
    gen = np.random.default_rng(rng)
    boot = np.median(gen.choice(slopes, size=(n_boot, slopes.size)), axis=1)
    return Stiffness(
        float(np.median(slopes)), float(boot.std()), slopes.size, slopes
    )
