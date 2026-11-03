import itertools

import numpy as np
import pytest

from unzipy.fluctuation import (
    bennett,
    bennett_error,
    bias,
    bias_power_law,
    jarzynski,
    jarzynski_corrected,
    jarzynski_reverse,
)
from unzipy.stats import bootstrap, bootstrap_pdf

DG, SIG = (
    5.0,
    2.0,
)  # Gaussian work: <w_f> = DG + s^2/2, <w_r> = -DG + s^2/2 (Crooks-consistent)


def sample(rng, nf, nr):
    s = np.sqrt(SIG**2)
    return rng.normal(DG + s**2 / 2, s, nf), rng.normal(-DG + s**2 / 2, s, nr)


def test_jarzynski_gaussian():
    rng = np.random.default_rng(0)
    w_f, w_r = sample(rng, 400_000, 400_000)
    assert abs(jarzynski(w_f) - DG) < 0.1
    assert abs(jarzynski_reverse(w_r) - DG) < 0.1


def test_jarzynski_is_stable_for_large_work():
    assert np.isclose(
        jarzynski(np.array([1e4, 1e4 + 1])), 1e4 - np.log((1 + np.exp(-1)) / 2)
    )


def test_bennett_gaussian_and_unequal_sizes():
    rng = np.random.default_rng(1)
    w_f, w_r = sample(rng, 2000, 700)
    est, err = bennett(w_f, w_r), bennett_error(w_f, w_r)
    assert abs(est - DG) < 4 * err


def test_bennett_error_calibration():
    rng = np.random.default_rng(2)
    est, err = [], []
    for _ in range(300):
        w_f, w_r = sample(rng, 100, 60)
        e = bennett(w_f, w_r)
        est.append(e)
        err.append(bennett_error(w_f, w_r, e))
    assert abs(np.mean(est) - DG) < 0.1
    assert 0.8 < np.mean(err) / np.std(est) < 1.25


def test_bootstrap_resampling_is_uniform():
    # resampling must be uniform over all indices (a common bug draws
    # element i from U(1, i))
    data = np.arange(1000.0)
    mean_of_resamples = bootstrap(
        data, lambda a: a.mean(axis=-1), n_boot=2000, rng=0
    )
    assert abs(mean_of_resamples.mean() - data.mean()) < 3
    np.testing.assert_allclose(
        mean_of_resamples.std(), data.std() / np.sqrt(1000), rtol=0.1
    )


def test_bootstrap_pdf_normalised_and_accurate():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 20_000)
    c, p, e = bootstrap_pdf(x, n_boot=300, bins=40, rng=4)
    np.testing.assert_allclose(
        np.sum(p * np.diff(np.histogram_bin_edges(x, 40))), 1.0, rtol=1e-9
    )
    ref = np.exp(-c * c / 2) / np.sqrt(2 * np.pi)
    assert np.all(np.abs(p - ref) < 5 * e + 0.01)


def test_bootstrap_pdf_with_limits_keeps_full_normalisation():
    x = np.random.default_rng(5).normal(0, 1, 5000)
    _c, p, e = bootstrap_pdf(x, n_boot=200, bins=20, limits=(-1, 1), rng=6)
    inside = np.mean(
        np.abs(x) <= 1
    )  # density integrates to the in-range fraction
    np.testing.assert_allclose(np.sum(p * 0.1), inside, rtol=0.01)
    assert np.all(e > 0)


def test_crooks_slope_and_crossing():
    from unzipy.fluctuation import crooks_crossing, crooks_logratio

    rng = np.random.default_rng(7)
    w_f, w_r = sample(rng, 20_000, 20_000)
    w, lr, err = crooks_logratio(w_f, w_r, bins=30, n_boot=200, rng=8)
    slope, dg = crooks_crossing(w, lr, err)
    assert abs(slope - 1) < 0.1
    assert abs(dg - DG) < 0.2


def test_jarzynski_vs_n_converges():
    from unzipy.fluctuation import jarzynski_vs_n

    w_f, _ = sample(np.random.default_rng(9), 50_000, 1)
    mean, std = jarzynski_vs_n(w_f, [10, 1000, 20_000], n_rep=50, rng=10)
    assert (
        mean[0] > mean[-1] > DG - 0.3
    )  # biased high for few samples, converging down
    assert std[0] > std[-1]


def test_effective_beta_error_matches_finite_differences():
    from unzipy.fluctuation import effective_beta

    vg, vw, eg, ew = 4.0, 9.0, 0.3, 0.5
    beta, err, _approx = effective_beta(vg, vw, eg, ew)
    assert np.isclose(beta, 9 / 13)
    h = 1e-6
    d_vg = (effective_beta(vg + h, vw)[0] - effective_beta(vg - h, vw)[0]) / (
        2 * h
    )
    d_vw = (effective_beta(vg, vw + h)[0] - effective_beta(vg, vw - h)[0]) / (
        2 * h
    )
    assert np.isclose(err, np.hypot(d_vg * eg, d_vw * ew))


def test_dissipated_work_limits():
    from unzipy.fluctuation import dissipated_work, dissipated_work_cycle

    # slow driving: W -> r * int exp(-mu y) sech^3(y) dy  (= r pi/2 for mu = 0)
    np.testing.assert_allclose(
        dissipated_work(1e-3, 0.0) / 1e-3, np.pi / 2, rtol=2e-3
    )
    from scipy.integrate import quad

    mu = 0.4
    slope = quad(lambda y: np.exp(-mu * y) / np.cosh(y) ** 3, -30, 30)[0]
    np.testing.assert_allclose(
        dissipated_work(1e-3, mu) / 1e-3, slope, rtol=2e-3
    )
    rates = [0.01, 0.1, 1.0, 10.0]
    w = [dissipated_work_cycle(r, 0.3) for r in rates]
    assert all(
        b > a for a, b in itertools.pairwise(w)
    )  # dissipation grows with rate


def test_matching_and_wdis():
    from unzipy.fluctuation import estimate_free_energy, matching

    rng = np.random.default_rng(12)
    # strongly irreversible: sigma^2 = 40 kT^2 -> W_diss = 20 kT,
    # non-overlapping distributions
    w_f, w_r = rng.normal(70 + 20, 6.3, 400), rng.normal(-70 + 20, 6.3, 400)
    assert abs(matching(w_f, w_r) - 70) < 1.0
    fe = estimate_free_energy(w_f, w_r)
    assert abs(fe.wdis - 20) < 1.5


def test_ensemble_rate_ratio_matches_published_value():
    from unzipy.fluctuation import (
        dimensionless_rate,
        ensemble_rate_ratio,
        fragility,
    )

    # PRE 98, 032146: x_m = 19.8 nm, delta_f = 1.1 pN, mu = -0.3
    # gives Omega = 0.55 at 298 K
    assert abs(ensemble_rate_ratio(-0.3, 19.8, 1.1) - 0.55) < 0.01
    assert fragility(5.0, 15.0) == -0.5
    assert abs(dimensionless_rate(10.0, 20.0, 0.5, 4.0) - 25.0) < 1e-12


def test_jarzynski_bias_model_palassini_ritort():
    from unzipy.fluctuation import bias_critical, bias_extreme, gaussian_bias

    # paper: for Gaussian work, eq. (6) gives <B_N> = ln 2 at lambda = 1
    for d in (4.0, 8.0, 16.0):
        assert abs(gaussian_bias(np.exp(d), 2 * d) - np.log(2)) < 1e-12
    # brute-force bias of -ln <exp(-D)> for D ~ N(<D>, 2<D>)  (DeltaF = 0)
    rng = np.random.default_rng(0)
    d_mean = 8.0
    for n, reps in [(64, 20000), (256, 5000), (4096, 500)]:
        d = rng.normal(d_mean, np.sqrt(2 * d_mean), (reps, n))
        numeric = -np.log(np.mean(np.exp(-d), axis=1)).mean()
        assert (
            abs(gaussian_bias(n, 2 * d_mean) - numeric) < 0.1 * numeric + 0.05
        ), n  # lambda in [0.5, 1]
    # eq. (5) is the small-lambda form (lambda = 0.17 and 0.35)
    for n in (4, 16):
        d = rng.normal(d_mean, np.sqrt(2 * d_mean), (20000, n))
        numeric = -np.log(np.mean(np.exp(-d), axis=1)).mean()
        assert (
            abs(gaussian_bias(n, 2 * d_mean, "extreme") - numeric)
            < 0.1 * numeric
        )
    with pytest.raises(ValueError):
        bias_critical(
            100, 5.0, delta=1.5
        )  # q needed outside the Gaussian case
    assert np.isfinite(
        bias_extreme(100, 5.0, delta=1.5, q=1.5, alpha=-0.5)
    )  # non-Gaussian tail runs


def test_jarzynski_corrected_reduces_the_bias():
    from unzipy.fluctuation import fit_lower_tail, jarzynski_corrected

    rng = np.random.default_rng(1)
    d_mean = 8.0
    tail = fit_lower_tail(rng.normal(d_mean, np.sqrt(2 * d_mean), 100_000))
    assert abs(tail.mu - d_mean) < 0.8 and abs(tail.q - np.pi**-0.5) < 0.06
    plain, corrected = [], []
    for _ in range(300):
        r = jarzynski_corrected(
            rng.normal(d_mean, np.sqrt(2 * d_mean), 100)
        )  # DeltaF = 0
        plain.append(r.dF_N)
        corrected.append(r.dF_star)
    assert np.mean(plain) > 1.4 and abs(np.mean(corrected)) < 0.5 * np.mean(
        plain
    )


@pytest.mark.parametrize(
    "mu, n",
    [(2.0, 200), (2.0, 1000), (4.0, 1000), (4.0, 5000), (8.0, 100_000)],
)
def test_auto_bias_is_nonnegative_and_accurate_for_lambda_above_one(mu, n):
    """Check that the bias is never negative and matches simulation.

    Eq. 6 alone goes negative for lambda > ~1.5, which is impossible;
    bias() must not.
    """
    from unzipy.fluctuation import bias, gaussian_bias

    rng = np.random.default_rng(10)
    reps = max(200, int(3e6 / n))
    d = rng.normal(mu, np.sqrt(2 * mu), (reps, n))
    numeric = -np.log(np.mean(np.exp(-d), axis=1)).mean()
    est = bias(n, mu, 2.0, 0.0, np.pi**-0.5)
    assert est >= 0 and est == gaussian_bias(n, 2 * mu)
    assert abs(est - numeric) < 0.1, (mu, n, est, numeric)


def test_bias_is_bounded_and_vanishes_for_converged_small_dissipation():
    # small mu with many samples: lambda far above 1 and mu < ln 2, where
    # the empirical power law would grow with N
    for mu in (0.05, 0.3, 0.5, 0.69):
        for n in (10, 100, 1000, 10**5):
            b = bias(n, mu)
            assert 0.0 <= b <= mu + 1e-12
    assert bias(10**5, 0.3) < 1e-6
    # more samples never increase the bias at fixed mu
    ns = np.array([10, 30, 100, 300, 1000, 10**4, 10**5])
    for mu in (0.3, 1.0, 3.0):
        b = np.array([bias(int(n), mu) for n in ns])
        assert np.all(np.diff(b) <= 1e-9)


def test_power_law_rejects_dissipation_below_ln2():
    with pytest.raises(ValueError):
        bias_power_law(100, 0.5)
    assert 0.0 < bias_power_law(100, 2.0) < 2.0


def test_corrected_jarzynski_stays_close_to_plain_for_benign_work():
    for seed in range(20):
        w = np.random.default_rng(seed).normal(100.0, 0.8, 150)  # D = 0.32
        ce = jarzynski_corrected(w)
        assert 0.0 <= ce.bias < 0.3
        assert abs(ce.dF_star - ce.dF_N) < 0.3
