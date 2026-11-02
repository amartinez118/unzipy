"""Tests of the elastic-model fitters against synthetic curves."""

import numpy as np
import pytest

from unzipy.fitting import (
    Model,
    compare,
    dipole,
    efjc,
    ewlc,
    fit,
    fjc,
    ss_fjc,
    ss_wlc,
    trap,
    wlc,
)

F = np.linspace(1.0, 40.0, 200)


def noisy(model, truth, sigma, rng, f=F):
    """Return extensions of ``model`` at ``truth`` plus Gaussian noise."""
    x = model.extension(f, truth)
    return x + rng.normal(0, sigma, x.size)


def within(report, truth, n_sigma=4.0):
    """Check that every fitted parameter is within ``n_sigma`` of the truth."""
    for k in report.free:
        err = report.errors[k]
        assert abs(report.values[k] - truth[k]) < n_sigma * err, (k, report)


CASES = {
    "wlc": (Model(wlc()), {"L": 100.0, "p": 0.9, "shift": 3.0}, 0.5),
    "wlc_bouchiat": (
        Model(wlc(bouchiat=True)),
        {"L": 100.0, "p": 0.9, "shift": 3.0},
        0.5,
    ),
    "ewlc_scaled": (
        Model(ewlc()),
        {"L": 250.0, "p": 40.0, "Y": 900.0, "shift": 2.0},
        0.3,
    ),
    "ewlc_additive": (
        Model(ewlc(form="additive")),
        {"L": 250.0, "p": 40.0, "S": 1200.0, "shift": 2.0},
        0.3,
    ),
    "fjc": (Model(fjc()), {"L": 100.0, "b": 1.5, "shift": 1.0}, 0.4),
    "efjc": (
        Model(efjc()),
        {"L": 100.0, "b": 1.5, "Y": 600.0, "shift": 1.0},
        0.2,
    ),
    "ss_wlc": (
        Model(ss_wlc(), offset=False),
        {"n": 44.0, "d0": 0.59, "p": 1.35},
        0.1,
    ),
    "ss_fjc": (
        Model(ss_fjc(), offset=False),
        {"n": 44.0, "d0": 0.59, "b": 1.5},
        0.1,
    ),
}
STARTS = {
    "wlc": {"L": 80.0, "p": 1.3},
    "wlc_bouchiat": {"L": 80.0, "p": 1.3},
    "ewlc_scaled": {"L": 200.0, "p": 30.0, "Y": 500.0},
    "ewlc_additive": {"L": 200.0, "p": 30.0, "S": 800.0},
    "fjc": {"L": 80.0, "b": 2.0},
    "efjc": {"L": 80.0, "b": 2.0, "Y": 400.0},
    "ss_wlc": {"n": 30.0, "p": 1.0},
    "ss_fjc": {"n": 30.0, "b": 1.0},
}


@pytest.mark.parametrize("name", CASES)
def test_models_recover_parameters_in_both_variables(name):
    model, truth, sigma = CASES[name]
    rng = np.random.default_rng(1)
    start = dict(STARTS[name])
    fixed = {"d0": truth["d0"]} if "d0" in truth else None
    # x(f): noise on the extension
    x = noisy(model, truth, sigma, rng)
    r = fit(model, F, x, start=start, fixed=fixed, sigma=sigma)
    assert r.success
    within(r, truth)
    np.testing.assert_allclose(r.predict(F), model.extension(F, truth), atol=3)
    # f(x): noise on the force, as in a position-controlled experiment
    x_clean = model.extension(F, truth)
    f_noisy = F + rng.normal(0, 0.2, F.size)
    r2 = fit(
        model,
        f_noisy,
        x_clean,
        variable="f(x)",
        start=start,
        fixed=fixed,
        sigma=0.2,
    )
    assert r2.success and r2.variable == "f(x)"
    within(r2, truth)
    high = F >= 8  # f(x) is ill-conditioned near zero force
    np.testing.assert_allclose(r2.predict(x_clean)[high], F[high], atol=1.5)


def test_noiseless_fits_agree_between_variables():
    model, truth, _ = CASES["wlc"]
    x = model.extension(F, truth)
    a = fit(model, F, x, start=STARTS["wlc"])
    b = fit(model, F, x, variable="f(x)", start=STARTS["wlc"])
    for k in truth:
        assert abs(a.values[k] - truth[k]) < 1e-4
        assert abs(b.values[k] - truth[k]) < 1e-3


def test_dipole_and_trap_elements():
    rng = np.random.default_rng(2)
    f = np.linspace(2, 30, 120)
    d = Model(dipole(), offset=False)
    x = noisy(d, {"d": 2.0}, 0.02, rng, f)
    r = fit(d, f, x, start={"d": 1.0}, sigma=0.02)
    within(r, {"d": 2.0})
    tr = Model(trap(), offset=False)
    truth = {"k0": 0.07, "alpha": 4e-4}
    x = noisy(tr, truth, 0.3, rng, f)
    held = fit(tr, f, x, start={"k0": 0.05})  # alpha fixed at 0 by default
    assert "alpha" not in held.free
    freed = fit(tr, f, x, start={"k0": 0.05}, free=["alpha"])
    assert "alpha" in freed.free
    within(freed, truth)


def test_series_model_with_handles_strand_and_trap():
    rng = np.random.default_rng(3)
    model = Model(ewlc(label="handles"), ss_wlc(), trap(), offset=True)
    truth = {
        "handles.L": 270.0,
        "handles.p": 20.0,
        "handles.Y": 200.0,
        "ss.n": 52.0,
        "ss.d0": 0.665,
        "ss.p": 0.75,
        "trap.k0": 0.068,
        "trap.alpha": 0.0,
        "shift": 10.0,
    }
    f = np.linspace(3, 25, 250)
    x = noisy(model, truth, 0.4, rng, f)
    fixed = {k: truth[k] for k in ("handles.p", "handles.Y", "ss.d0", "ss.p")}
    r = fit(
        model,
        f,
        x,
        fixed=fixed,
        start={"handles.L": 200.0, "ss.n": 40.0, "trap.k0": 0.05},
        sigma=0.4,
    )
    assert r.success
    assert set(r.free) == {"handles.L", "ss.n", "trap.k0", "shift"}
    within(r, truth)


def test_fixed_bounds_and_parameter_handling():
    model, truth, sigma = CASES["wlc"]
    x = noisy(model, truth, sigma, np.random.default_rng(4))
    r = fit(model, F, x, start=STARTS["wlc"], fixed={"p": 0.9})
    assert "p" not in r.free and r.values["p"] == 0.9
    assert np.isnan(r.errors["p"])
    held_at_start = fit(
        model, F, x, start={"L": 90.0, "p": 1.0}, fixed={"p": None}
    )
    assert held_at_start.values["p"] == 1.0
    bound = fit(model, F, x, start=STARTS["wlc"], bounds={"L": (50.0, 99.0)})
    assert bound.values["L"] <= 99.0 + 1e-9
    assert "L" in bound.free


def test_uncertainties_are_calibrated():
    model, truth, sigma = CASES["wlc"]
    rng = np.random.default_rng(5)
    est, err = [], []
    for _ in range(150):
        x = noisy(model, truth, sigma, rng)
        r = fit(model, F, x, start=STARTS["wlc"])  # sigma unknown
        est.append(r.values["p"])
        err.append(r.errors["p"])
    assert abs(np.mean(est) - truth["p"]) < 0.01
    assert 0.8 < np.mean(err) / np.std(est) < 1.25
    # absolute sigma: reduced chi-square is about one for the right sigma
    x = noisy(model, truth, sigma, rng)
    r = fit(model, F, x, start=STARTS["wlc"], sigma=sigma, absolute_sigma=True)
    assert 0.7 < r.reduced_chi2 < 1.3


def test_bootstrap_intervals_cover_the_truth():
    model, truth, sigma = CASES["wlc"]
    x = noisy(model, truth, sigma, np.random.default_rng(6))
    r = fit(model, F, x, start=STARTS["wlc"])
    boot = r.bootstrap(n_boot=200, level=0.95, rng=7)
    assert boot["samples"].shape[1] == len(r.free)
    for k in r.free:
        lo, hi = boot["ci"][k]
        assert lo < truth[k] < hi
        assert 0.5 < (hi - lo) / (2 * 1.96 * r.errors[k]) < 2.0


def test_model_comparison_selects_the_true_model():
    rng = np.random.default_rng(8)
    truth_model, truth, sigma = CASES["wlc"]
    x = noisy(truth_model, truth, sigma, rng)
    good = fit(truth_model, F, x, start=STARTS["wlc"])
    bad = fit(Model(fjc()), F, x, start={"L": 80.0, "b": 2.0})
    rows = compare(wlc=good, fjc=bad)
    assert rows[0]["label"] == "wlc" and rows[0]["delta_aic"] == 0
    assert rows[1]["delta_aic"] > 10 and rows[1]["delta_bic"] > 10
    assert abs(sum(r["weight"] for r in rows) - 1) < 1e-12
    assert good.aic < bad.aic and good.bic < bad.bic


def test_negative_forces_and_report_helpers():
    model, truth, sigma = CASES["wlc"]
    rng = np.random.default_rng(9)
    f = np.concatenate([[-0.3, -0.1, 0.0], F])  # noise around zero force
    x = noisy(model, truth, sigma, rng, np.clip(f, 0, None))
    r = fit(model, f, x, start=STARTS["wlc"])
    assert r.success
    text = r.summary()
    assert "L" in text and "+/-" in text and "AIC" in text
    corr = r.correlation
    assert corr.shape == (3, 3) and np.allclose(np.diag(corr), 1)


@pytest.mark.parametrize(
    "call",
    [
        lambda: fit(Model(wlc()), F, F, start={"L": 1, "p": 1}, variable="y"),
        lambda: fit(Model(wlc()), F, F[:-1], start={"L": 1, "p": 1}),
        lambda: fit(
            Model(wlc()), F, np.full(F.size, np.nan), start={"L": 1, "p": 1}
        ),
        lambda: fit(Model(wlc()), F, F, start={"L": 1}),  # missing p
        lambda: fit(Model(wlc()), F, F, start={"L": 1, "q": 1}),  # unknown
        lambda: fit(Model(wlc()), F, F, free=["nope"], start={"L": 1, "p": 1}),
        lambda: fit(
            Model(wlc(), offset=False),
            F,
            F,
            fixed={"L": 1.0, "p": 1.0},
        ),  # nothing free
        lambda: fit(Model(wlc()), F[:3], F[:3], start={"L": 1, "p": 1}),
        lambda: fit(Model(wlc()), F, F, start={"L": 1, "p": 1}, sigma=-1.0),
        lambda: Model(),
        lambda: Model(wlc(), wlc()),  # repeated label
        lambda: ewlc(form="odd"),
        lambda: compare(
            only=fit(Model(wlc()), F, F, start={"L": 100, "p": 1})
        ),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
