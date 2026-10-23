import numpy as np

from unzipy.stats import (
    grubbs_outliers,
    iqr_outliers,
    kde_pdf,
    optimal_bandwidth,
)


def test_grubbs_finds_planted_outliers_only():
    x = np.random.default_rng(0).normal(0, 1, 200)
    assert grubbs_outliers(x).sum() <= 1  # Gaussian sample: essentially none
    x[[3, 77]] = [12.0, -15.0]
    m = grubbs_outliers(x)
    assert m[3] and m[77] and m.sum() <= 3


def test_iqr_deterministic():
    x = np.r_[np.arange(100.0), 1000.0]
    assert iqr_outliers(x).tolist() == [False] * 100 + [True]


def test_kde_bandwidth_and_accuracy():
    x = np.random.default_rng(1).normal(0, 1, 3000)
    h = optimal_bandwidth(x)
    silverman = 1.06 * x.std() * x.size ** (-0.2)
    assert 0.5 * silverman < h < 2 * silverman
    g = np.linspace(-2.5, 2.5, 11)
    ref = np.exp(-g * g / 2) / np.sqrt(2 * np.pi)
    assert np.max(np.abs(kde_pdf(x, g) - ref)) < 0.03


def test_gmm_recovers_components_and_classifies():
    from unzipy.stats import fit_gmm

    rng = np.random.default_rng(0)
    x = np.concatenate(
        [
            rng.normal(20, 2, 3000),
            rng.normal(33, 3.5, 4000),
            rng.normal(46, 2.3, 1000),
        ]
    )
    m = fit_gmm(x, 3, rng=1)
    np.testing.assert_allclose(m.means, [20, 33, 46], atol=0.3)
    np.testing.assert_allclose(m.sigmas, [2, 3.5, 2.3], rtol=0.1)
    np.testing.assert_allclose(m.weights, [3 / 8, 4 / 8, 1 / 8], atol=0.02)
    np.testing.assert_allclose(m.responsibilities(x).sum(axis=1), 1.0)
    labels = m.classify(x, rng=2)
    truth = np.repeat([0, 1, 2], [3000, 4000, 1000])
    assert (
        np.mean(labels == truth) > 0.85
    )  # overlap between neighbours limits accuracy
    cls = m.classify(np.array([20.0, 46.0]), rng=3)
    assert cls[0] == 0 and cls[1] == 2
    # density integrates to one
    g = np.linspace(0, 70, 7001)
    np.testing.assert_allclose(np.trapezoid(m.pdf(g), g), 1.0, atol=1e-3)


def test_gmm_binned_path_for_huge_samples():
    from unzipy.stats import fit_gmm

    rng = np.random.default_rng(1)
    x = np.concatenate(
        [rng.normal(20, 2, 300_000), rng.normal(33, 3.5, 300_000)]
    )
    m = fit_gmm(
        x, 2, n_init=2, rng=2, max_points=100_000
    )  # triggers the histogram path
    np.testing.assert_allclose(m.means, [20, 33], atol=0.05)
    np.testing.assert_allclose(m.sigmas, [2, 3.5], rtol=0.03)
