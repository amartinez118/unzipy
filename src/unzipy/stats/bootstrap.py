"""Bootstrap utilities (uniform resampling with replacement)."""

from __future__ import annotations

import numpy as np


def _rng(rng):
    """Return a random generator.

    Args:
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        numpy.random.Generator: The generator.
    """
    return (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )


def bootstrap(data, statistic, n_boot=1000, rng=None):
    """Return the bootstrap distribution of a statistic.

    Resampling is done in chunks so that memory stays bounded.

    Args:
        data (array-like): One-dimensional sample.
        statistic (callable): Receives an array of shape ``(n, N)`` and must
            reduce the last axis, e.g. ``lambda w: jarzynski(w, axis=-1)``.
        n_boot (int): Number of resamples.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        numpy.ndarray: Statistic of each resample, length ``n_boot``.
    """
    data = np.asarray(data, dtype=float)
    gen = _rng(rng)
    chunk = max(
        1, int(2e7 // data.size)
    )  # bounds memory to ~160 MB of indices
    out = [
        np.asarray(
            statistic(
                data[
                    gen.integers(
                        0, data.size, size=(min(chunk, n_boot - s), data.size)
                    )
                ]
            )
        )
        for s in range(0, n_boot, chunk)
    ]
    return np.concatenate(out)


def bootstrap_pdf(data, n_boot=1000, bins=None, limits=None, rng=None):
    """Estimate a probability density with bootstrap error bars.

    Bin edges are fixed from the sample, so every resample is histogrammed on
    the same grid. The density is normalised by the full sample size, so data
    outside ``limits`` reduce the integral accordingly.

    Args:
        data (array-like): One-dimensional sample.
        n_boot (int): Number of resamples.
        bins (int | array-like | None): Number of bins or bin edges. The Rice
            rule ``2 N**(1/3)`` is used if None.
        limits (tuple[float, float] | None): Range of the histogram.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Bin centres, mean
        density and its standard deviation over the resamples.
    """
    data = np.asarray(data, dtype=float)
    n = data.size
    if bins is None:
        bins = int(np.ceil(2 * n ** (1 / 3)))
    edges = np.histogram_bin_edges(data, bins=bins, range=limits)
    nb = len(edges) - 1
    width = np.diff(edges)
    which = np.searchsorted(edges, data, side="right") - 1
    which[data == edges[-1]] = nb - 1  # right edge is inclusive
    which[(data < edges[0]) | (data > edges[-1])] = (
        nb  # out of range: dump bin, still resampled
    )
    gen = _rng(rng)
    dens = np.empty((n_boot, nb))
    chunk = max(1, int(2e7 // n))
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        draw = which[gen.integers(0, n, size=(m, n))]
        flat = (draw + (nb + 1) * np.arange(m)[:, None]).ravel()
        counts = np.bincount(flat, minlength=m * (nb + 1)).reshape(m, nb + 1)[
            :, :nb
        ]
        dens[s : s + m] = counts / (n * width)
    return (
        0.5 * (edges[:-1] + edges[1:]),
        dens.mean(axis=0),
        dens.std(axis=0, ddof=1),
    )


def bin_with_errors(x, y, err, n_bins=10):
    """Average ``y`` in bins of ``x``, combining errors and scatter.

    The bin error adds in quadrature the propagated error of the mean,
    ``sqrt(sum err^2) / n``, and the scatter of the values in the bin.

    Args:
        x (array-like): Abscissa.
        y (array-like): Values to average.
        err (array-like): Errors of ``y``.
        n_bins (int): Number of equal bins between ``min(x)`` and ``max(x)``.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Bin positions,
        means and errors; empty bins are dropped.
    """
    x, y, err = (np.asarray(a, dtype=float) for a in (x, y, err))
    grid = np.linspace(x.min(), x.max(), n_bins)
    which = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = which == b
        if m.any():
            scatter = y[m].std(ddof=1) if m.sum() > 1 else 0.0
            out.append(
                (
                    grid[b],
                    y[m].mean(),
                    np.hypot(np.sqrt(np.sum(err[m] ** 2)) / m.sum(), scatter),
                )
            )
    return tuple(np.array(c) for c in zip(*out, strict=True))
