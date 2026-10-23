"""Convergence of the Jarzynski estimator with the sample size."""

from __future__ import annotations

import numpy as np

from .estimators import jarzynski


def jarzynski_vs_n(w, sizes, n_rep=100, rng=None):
    """Return the Jarzynski estimate from random subsets of trajectories.

    The result is the convergence curve of the estimator: its bias decays as
    rare low-work events are sampled.

    Args:
        w (array-like): Work values (kT).
        sizes (array-like): Subset sizes to evaluate.
        n_rep (int): Random subsets (drawn with replacement) per size.
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Mean and standard deviation of
        the estimate over the replicates, one entry per size.
    """
    w = np.asarray(w, dtype=float)
    gen = np.random.default_rng(rng)
    est = np.array(
        [
            jarzynski(
                w[gen.integers(0, w.size, size=(n_rep, int(n)))], axis=-1
            )
            for n in sizes
        ]
    )
    return est.mean(axis=1), est.std(axis=1, ddof=1)
