"""Gaussian mixtures: EM fit, responsibilities and classification.

The mixtures describe, for example, the distribution of free energies across
molecules; each sample is assigned to a component with its posterior
probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class GaussianMixture:
    """One-dimensional Gaussian mixture.

    Attributes:
        weights (numpy.ndarray): Component weights, summing to one.
        means (numpy.ndarray): Component means.
        sigmas (numpy.ndarray): Component standard deviations.
        log_likelihood (float): Log-likelihood of the fitted sample.
    """

    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    log_likelihood: float = np.nan

    def log_components(self, x):
        """Return ``ln(w_k N(x_i; mu_k, sigma_k))``.

        Args:
            x (array-like): Sample points.

        Returns:
            numpy.ndarray: Array of shape ``(len(x), k)``.
        """
        x = np.asarray(x, dtype=float)[:, None]
        z = (x - self.means) / self.sigmas
        return (
            np.log(self.weights)
            - 0.5 * z * z
            - np.log(self.sigmas)
            - 0.5 * np.log(2 * np.pi)
        )

    def pdf(self, x):
        """Return the mixture density.

        Args:
            x (array-like): Points where the density is evaluated.

        Returns:
            numpy.ndarray: Density at ``x``.
        """
        return np.exp(np.asarray(logsumexp(self.log_components(x), axis=1)))

    def responsibilities(self, x):
        """Return the posterior component probabilities.

        Args:
            x (array-like): Sample points.

        Returns:
            numpy.ndarray: Array ``r[i, k]``, the probability that sample ``i``
            belongs to component ``k``.
        """
        lc = self.log_components(x)
        return np.exp(lc - np.asarray(logsumexp(lc, axis=1, keepdims=True)))

    def classify(self, x, rng=None):
        """Draw a component for each sample from its posterior.

        Args:
            x (array-like): Sample points.
            rng (int | numpy.random.Generator | None): Seed or generator.

        Returns:
            numpy.ndarray: Component index of each sample.
        """
        r = self.responsibilities(x)
        u = np.random.default_rng(rng).uniform(size=(r.shape[0], 1))
        return (u > np.cumsum(r, axis=1)).sum(axis=1).clip(max=r.shape[1] - 1)


def fit_gmm(
    x,
    k,
    n_init=5,
    max_iter=500,
    tol=1e-7,
    rng=None,
    max_points=200_000,
    bins=4096,
):
    """Fit a one-dimensional Gaussian mixture by EM.

    The best of ``n_init`` random starts is returned. Samples beyond
    ``max_points`` are summarised by a histogram (weighted EM) with Sheppard's
    correction ``sigma^2 -> sigma^2 - h^2 / 12`` for the bin width ``h``, which
    keeps millions of samples fast at negligible loss of accuracy.

    Args:
        x (array-like): One-dimensional sample.
        k (int): Number of components.
        n_init (int): Random starts.
        max_iter (int): Maximum EM iterations per start.
        tol (float): Convergence threshold on the mean log-likelihood gain per
            sample.
        rng (int | numpy.random.Generator | None): Seed or generator.
        max_points (int): Sample size above which the histogram is used.
        bins (int): Histogram bins for large samples.

    Returns:
        GaussianMixture: Fitted mixture with components sorted by mean.

    Raises:
        ValueError: If ``k`` is not between 1 and the sample size.
    """
    x = np.asarray(x, dtype=float).ravel()
    if k < 1 or k > x.size:
        raise ValueError("need 1 <= k <= number of samples")
    gen = np.random.default_rng(rng)
    n_total, width = x.size, 0.0
    if x.size > max_points:
        counts, edges = np.histogram(x, bins=bins)
        keep = counts > 0
        x = 0.5 * (edges[1:] + edges[:-1])[keep]
        wgt = counts[keep].astype(float)
        width = edges[1] - edges[0]
    else:
        wgt = np.ones(x.size)
    mean_x = np.average(x, weights=wgt)
    std_x = np.sqrt(np.average((x - mean_x) ** 2, weights=wgt))
    best = None
    for _ in range(n_init):
        starts = gen.choice(x, k, replace=False, p=wgt / wgt.sum())
        m = GaussianMixture(
            np.full(k, 1.0 / k), starts, np.full(k, std_x / k + 1e-9)
        )
        prev = -np.inf
        for _ in range(max_iter):
            lc = m.log_components(x)
            ll = np.asarray(logsumexp(lc, axis=1))
            r = np.exp(lc - ll[:, None]) * wgt[:, None]
            nk = r.sum(axis=0) + 1e-12
            means = (r * x[:, None]).sum(axis=0) / nk
            var = (r * (x[:, None] - means) ** 2).sum(axis=0) / nk
            m = GaussianMixture(
                nk / n_total, means, np.sqrt(np.maximum(var, 1e-12))
            )
            total = (wgt * ll).sum()
            if (
                total - prev
            ) / n_total < tol:  # mean log-likelihood gain per sample
                break
            prev = total
        m.log_likelihood = float(
            (wgt * np.asarray(logsumexp(m.log_components(x), axis=1))).sum()
        )
        if best is None or m.log_likelihood > best.log_likelihood:
            best = m
    assert best is not None
    sigmas = np.sqrt(np.maximum(best.sigmas**2 - width**2 / 12.0, 1e-12))
    order = np.argsort(best.means)
    return GaussianMixture(
        best.weights[order],
        best.means[order],
        sigmas[order],
        best.log_likelihood,
    )
