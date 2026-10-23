"""Split cyclic force traces into unfolding and folding trajectories.

A pulling/relaxing cycle crosses ``f_max`` once going up and ``f_min`` once
going down; each excursion above ``f_max`` (below ``f_min``) is one cluster of
samples separated from the next by at least ``gap`` seconds, represented by its
extreme. Segments run between consecutive extremes: min -> max is ``u``
(unfolding), max -> min is ``f`` (folding).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Segment:
    """A trajectory inside a trace.

    Attributes:
        kind (str): ``"u"`` (unfolding) or ``"f"`` (folding).
        slice (slice): Samples of the trajectory.
    """

    kind: str  # "u" or "f"
    slice: slice


def _cluster_extremes(t, f, mask, gap, use_max):
    """Return the extreme sample of each excursion.

    Args:
        t (numpy.ndarray): Time (s).
        f (numpy.ndarray): Force (pN).
        mask (numpy.ndarray): True for samples in an excursion.
        gap (float): Time gap (s) that separates two excursions.
        use_max (bool): Take the maximum (True) or the minimum force.

    Returns:
        numpy.ndarray: Sample index of the extreme of each excursion.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.array([], dtype=int)
    breaks = np.flatnonzero(np.diff(t[idx]) >= gap) + 1
    out = []
    for group in np.split(idx, breaks):
        out.append(
            group[np.argmax(f[group]) if use_max else np.argmin(f[group])]
        )
    return np.array(out, dtype=int)


def split_cycles(t, f, f_min, f_max, gap=1.0):
    """Split a trace into unfolding and folding segments.

    Args:
        t (array-like): Time (s), increasing.
        f (array-like): Force (pN).
        f_min (float): Force below which the folded end of a cycle is reached
            (pN).
        f_max (float): Force above which the unfolded end of a cycle is reached
            (pN).
        gap (float): Time gap (s) separating two excursions.

    Returns:
        list[Segment]: Segments in time order.
    """
    t, f = np.asarray(t, float), np.asarray(f, float)
    tops = _cluster_extremes(t, f, f > f_max, gap, True)
    bottoms = _cluster_extremes(t, f, f < f_min, gap, False)
    events = sorted([(i, "max") for i in tops] + [(i, "min") for i in bottoms])
    segments = []
    for (i0, e0), (i1, e1) in itertools.pairwise(events):
        if (
            e0 == e1
        ):  # two extremes of the same kind in a row: no complete excursion
            continue
        segments.append(
            Segment("u" if e0 == "min" else "f", slice(i0 + 1, i1))
        )
    return segments


def read_tweezers_trace(
    path, f_col=18, x_cols=(25, 27), t_col=31, t_range=None
):
    """Read a raw instrument file.

    Args:
        path (str | os.PathLike): File to read; ``#`` lines are comments.
        f_col (int): One-based force column; the absolute value is used.
        x_cols (tuple[int, int]): One-based columns of the two trap positions;
            their mean is the extension.
        t_col (int): One-based time column.
        t_range (tuple[float, float] | None): Keep only this time interval.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Time ``t``,
        extension ``x`` and force ``f``.
    """
    data = np.loadtxt(path, comments="#", ndmin=2)
    t, f = data[:, t_col - 1], np.abs(data[:, f_col - 1])
    x = data[:, [c - 1 for c in x_cols]].mean(axis=1)
    if t_range is not None:
        keep = (t > t_range[0]) & (t < t_range[1])
        t, x, f = t[keep], x[keep], f[keep]
    return t, x, f


def write_segments(t, x, f, segments, directory):
    """Write segments as trajectory files with an index.

    Files ``u1.txt``, ``f1.txt``, ... have the columns index, x, t, f; a
    ``LOG.txt`` lists their names.

    Args:
        t (numpy.ndarray): Time (s).
        x (numpy.ndarray): Extension (nm).
        f (numpy.ndarray): Force (pN).
        segments (list[Segment]): Segments from :func:`split_cycles`.
        directory (str | os.PathLike): Output folder, created if needed.

    Returns:
        list[str]: Names of the trajectories written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    counts = {"u": 0, "f": 0}
    names = []
    for seg in segments:
        counts[seg.kind] += 1
        name = f"{seg.kind}{counts[seg.kind]}"
        s = seg.slice
        n = len(t[s])
        np.savetxt(
            directory / f"{name}.txt",
            np.column_stack([np.arange(1, n + 1), x[s], t[s], f[s]]),
        )
        names.append(name)
    (directory / "LOG.txt").write_text("\n".join(names) + "\n")
    return names


def read_status_file(path):
    """Read an instrument file with a ``Status`` column.

    The columns used are ``#CycleCount/n X_force Y_force Z_force A_dist-Y
    B_dist-Y time(sec) Status``; any extra columns are ignored. The header is
    the first line that contains ``Status``.

    Args:
        path (str | os.PathLike): File to read.

    Returns:
        dict: Arrays ``count``, ``fx``, ``fy``, ``fz``, ``x``, ``t`` and
        ``status``, with ``x = (A_dist-Y + B_dist-Y) / 2``.
    """
    lines = Path(path).read_text().splitlines()
    head = next(i for i, ln in enumerate(lines) if "Status" in ln)
    names = lines[head].split()
    wanted = (
        "#CycleCount/n",
        "X_force",
        "Y_force",
        "Z_force",
        "A_dist-Y",
        "B_dist-Y",
        "time(sec)",
        "Status",
    )
    col = {n: names.index(n) for n in wanted}
    rows = [
        ln.split()
        for ln in lines[head + 1 :]
        if ln.strip() and not ln.startswith("#")
    ]
    data = np.array([[float(v) for v in r[: len(names)]] for r in rows])
    return {
        "count": data[:, col["#CycleCount/n"]],
        "fx": data[:, col["X_force"]],
        "fy": data[:, col["Y_force"]],
        "fz": data[:, col["Z_force"]],
        "x": 0.5 * (data[:, col["A_dist-Y"]] + data[:, col["B_dist-Y"]]),
        "t": data[:, col["time(sec)"]],
        "status": data[:, col["Status"]].astype(int),
    }


def split_by_status(status, fold_code=130, unfold_code=131):
    """Split a trace into maximal runs of constant status.

    Args:
        status (array-like): Status code of each sample.
        fold_code (int): Code of folding runs (``"f"``).
        unfold_code (int): Code of unfolding runs (``"u"``).

    Returns:
        list[Segment]: One segment per run; other codes are ignored.
    """
    status = np.asarray(status)
    edges = np.flatnonzero(np.diff(status) != 0) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [status.size]])
    kinds = {fold_code: "f", unfold_code: "u"}
    return [
        Segment(kinds[status[a]], slice(a, b))
        for a, b in zip(starts, stops, strict=True)
        if status[a] in kinds
    ]
