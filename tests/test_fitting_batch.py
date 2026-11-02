"""Tests of batch and global fits over many curves."""

import csv

import numpy as np
import pytest

from unzipy.fitting import (
    Curve,
    Model,
    as_curves,
    compare,
    fit_batch,
    fit_global,
    load_curves,
    wlc,
)

MODEL = Model(wlc())
F = np.linspace(1.0, 40.0, 120)
START = {"L": 80.0, "p": 1.3}


def make_curve(truth, rng, sigma=0.5, label=None, f=F):
    """Return a noisy WLC curve for the given true parameters."""
    x = MODEL.extension(f, truth) + rng.normal(0, sigma, f.size)
    return Curve(f, x, sigma, label)


def test_batch_fits_each_molecule_and_tabulates(tmp_path):
    rng = np.random.default_rng(1)
    truths = [
        {"L": 80.0 + 5 * i, "p": 0.8 + 0.05 * i, "shift": float(i)}
        for i in range(8)
    ]
    curves = [
        make_curve(t, rng, label=f"mol{i}") for i, t in enumerate(truths)
    ]
    rep = fit_batch(MODEL, curves, start=START)
    assert len(rep.ok) == 8 and not rep.errors
    for t, r in zip(truths, rep.reports, strict=True):
        for k in r.free:
            assert abs(r.values[k] - t[k]) < 4.5 * r.errors[k]
    rows = rep.table()
    assert [r["label"] for r in rows] == [f"mol{i}" for i in range(8)]
    assert {"L", "L_err", "p", "p_err", "shift", "rms", "aic"} <= set(rows[0])
    np.testing.assert_allclose(
        rep.values("L"), [t["L"] for t in truths], atol=4
    )
    summ = rep.summary()
    assert abs(summ["L"]["mean"] - np.mean([t["L"] for t in truths])) < 3
    assert summ["L"]["n"] == 8 and summ["L"]["sem"] < summ["L"]["std"]
    out = tmp_path / "fits.csv"
    rep.to_csv(out)
    read = list(csv.DictReader(out.open()))
    assert (
        len(read) == 8 and abs(float(read[3]["L"]) - rep.values("L")[3]) < 1e-9
    )


def test_batch_skips_or_raises_on_a_bad_curve():
    rng = np.random.default_rng(2)
    good = [
        make_curve({"L": 100.0, "p": 0.9, "shift": 1.0}, rng, label=f"g{i}")
        for i in range(3)
    ]
    bad = Curve(F, np.full(F.size, np.nan), None, "bad")
    rep = fit_batch(MODEL, [*good[:1], bad, *good[1:]], start=START)
    assert list(rep.ok) == ["g0", "g1", "g2"]
    assert "bad" in rep.errors and "finite" in rep.errors["bad"]
    row = [r for r in rep.table() if r["label"] == "bad"][0]
    assert row["success"] is False and "finite" in row["error"]
    with pytest.raises(ValueError):
        fit_batch(MODEL, [bad], on_error="raise", start=START)


def test_batch_accepts_per_curve_starts_and_curve_forms():
    rng = np.random.default_rng(3)
    a = make_curve({"L": 100.0, "p": 0.9, "shift": 0.0}, rng)
    b = make_curve({"L": 150.0, "p": 0.9, "shift": 0.0}, rng)
    items = [(a.f, a.x), {"f": b.f, "x": b.x, "label": "second"}]
    rep = fit_batch(
        MODEL, items, start=[{"L": 90.0, "p": 1.0}, {"L": 140.0, "p": 1.0}]
    )
    assert rep.labels == ["curve0", "second"] and len(rep.ok) == 2
    assert abs(rep.values("L")[1] - 150.0) < 6


def test_global_fit_shares_parameters_and_tightens_them():
    rng = np.random.default_rng(4)
    shifts = [0.0, 3.0, -2.0, 6.0, 1.5, -4.0]
    curves = [
        make_curve({"L": 100.0, "p": 0.9, "shift": s}, rng, label=f"m{i}")
        for i, s in enumerate(shifts)
    ]
    g = fit_global(
        MODEL, curves, shared=["L", "p"], start={**START, "shift": 0.0}
    )
    assert g.success and g.shared.keys() == {"L", "p"}
    assert abs(g.shared["p"] - 0.9) < 4 * g.shared_errors["p"]
    assert abs(g.shared["L"] - 100.0) < 4 * g.shared_errors["L"]
    for s, vals, errs in zip(
        shifts, g.per_curve, g.per_curve_errors, strict=True
    ):
        assert abs(vals["shift"] - s) < 4 * errs["shift"]
    assert g.n == 6 * F.size and g.dof == g.n - (2 + 6)
    assert g.n_free == len(g.free) == 8
    # sharing L and p constrains them far better than the individual fits
    indep = fit_batch(MODEL, curves, start=START)
    mean_err_p = np.mean([r.errors["p"] for r in indep.ok.values()])
    assert g.shared_errors["p"] < 0.6 * mean_err_p
    assert set(g.values_for("m2")) == {"L", "p", "shift"}
    np.testing.assert_allclose(
        g.predict("m1", F), MODEL.extension(F, g.values_for("m1"))
    )
    rows = g.table()
    assert rows[0]["label"] == "m0" and "shift_err" in rows[0]
    text = g.summary()
    assert "shared p" in text and "AIC" in text
    with pytest.raises(ValueError):
        g.values_for("nope")


def test_global_fit_with_different_contours_and_start_lists():
    rng = np.random.default_rng(5)
    lengths = [80.0, 100.0, 120.0, 140.0]
    curves = [
        make_curve({"L": L, "p": 1.1, "shift": 2.0}, rng, label=f"m{i}")
        for i, L in enumerate(lengths)
    ]
    g = fit_global(
        MODEL,
        curves,
        shared=["p", "shift"],
        start={"p": 1.5, "shift": 0.0, "L": [70.0, 90.0, 110.0, 130.0]},
    )
    assert g.success
    assert abs(g.shared["p"] - 1.1) < 4 * g.shared_errors["p"]
    np.testing.assert_allclose([v["L"] for v in g.per_curve], lengths, atol=5)


def test_global_fit_in_force_variable():
    rng = np.random.default_rng(6)
    curves = []
    for i, s in enumerate([0.0, 4.0, -3.0]):
        truth = {"L": 100.0, "p": 0.9, "shift": s}
        x = MODEL.extension(F, truth)
        f_noisy = F + rng.normal(0, 0.2, F.size)
        curves.append(Curve(f_noisy, x, 0.2, f"m{i}"))
    g = fit_global(
        MODEL,
        curves,
        shared=["L", "p"],
        variable="f(x)",
        start={**START, "shift": 0.0},
    )
    assert g.success and g.variable == "f(x)"
    assert abs(g.shared["p"] - 0.9) < 4 * g.shared_errors["p"]
    x = MODEL.extension(F, g.values_for("m1"))
    high = F >= 8
    np.testing.assert_allclose(g.predict("m1", x)[high], F[high], atol=1.0)


def test_global_errors_are_calibrated():
    est, err = [], []
    for seed in range(40):
        rng = np.random.default_rng(100 + seed)
        curves = [
            make_curve(
                {"L": 100.0, "p": 0.9, "shift": s},
                rng,
                label=f"m{i}",
                f=np.linspace(1, 40, 60),
            )
            for i, s in enumerate([0.0, 2.0, -3.0, 5.0])
        ]
        g = fit_global(
            MODEL, curves, shared=["L", "p"], start={**START, "shift": 0.0}
        )
        est.append(g.shared["p"])
        err.append(g.shared_errors["p"])
    assert abs(np.mean(est) - 0.9) < 0.02
    assert 0.7 < np.mean(err) / np.std(est) < 1.4


def test_global_versus_independent_selection_by_aic():
    rng = np.random.default_rng(7)

    def both(p_values):
        curves = [
            make_curve({"L": 100.0, "p": p, "shift": 1.0}, rng, label=f"m{i}")
            for i, p in enumerate(p_values)
        ]
        g = fit_global(
            MODEL, curves, shared=["L", "p"], start={**START, "shift": 0.0}
        )
        return compare(
            global_=g, independent=fit_batch(MODEL, curves, start=START)
        )

    same = both([0.9] * 6)  # truly shared: the global fit has the lower AIC
    assert same[0]["label"] == "global_"
    varied = both(
        [0.5, 0.8, 1.1, 1.4, 1.7, 2.0]
    )  # p really differs per molecule
    assert varied[0]["label"] == "independent"
    assert varied[1]["delta_aic"] > 10


def test_load_curves_reads_files_by_stem(tmp_path):
    rng = np.random.default_rng(8)
    for name in ("b", "a", "c"):
        c = make_curve({"L": 100.0, "p": 0.9, "shift": 0.0}, rng)
        np.savetxt(
            tmp_path / f"{name}.txt",
            np.column_stack([np.arange(F.size), c.x, np.zeros(F.size), c.f]),
        )
    curves = load_curves(tmp_path, sigma=0.5)
    assert [c.label for c in curves] == ["a", "b", "c"]
    np.testing.assert_allclose(curves[0].f, F)
    picked = load_curves(tmp_path, names=["c", "a"])
    assert [c.label for c in picked] == ["c", "a"]
    rep = fit_batch(MODEL, curves, start=START)
    assert len(rep.ok) == 3 and rep.table()[0]["label"] == "a"


@pytest.mark.parametrize(
    "call",
    [
        lambda: as_curves([]),
        lambda: as_curves([(F, F, 1.0, "x", "extra")]),
        lambda: as_curves([Curve(F, F, None, "d"), Curve(F, F, None, "d")]),
        lambda: fit_batch(MODEL, [(F, F)], on_error="ignore", start=START),
        lambda: fit_batch(MODEL, [(F, F)], start=[START, START]),
        lambda: fit_global(MODEL, [(F, F)], ["L"], start=START),
        lambda: fit_global(
            MODEL, [(F, F), (F, F)], ["nope"], start={**START, "shift": 0.0}
        ),
        lambda: fit_global(
            MODEL, [(F, F), (F, F)], ["L", "p", "shift"], start=START
        ),  # nothing per curve
        lambda: fit_global(MODEL, [(F, F), (F, F)], [], start=START),
        lambda: fit_global(
            MODEL, [(F, F), (F, F)], ["L"], start={**START, "p": [1.0]}
        ),  # wrong list length
        lambda: fit_global(
            MODEL, [(F, F), (F, F)], ["L"], variable="y", start=START
        ),
        lambda: fit_global(
            MODEL,
            [(F[:2], F[:2]), (F[:2], F[:2])],
            ["L"],
            start=START,
        ),  # too few points
        lambda: load_curves("/nonexistent-folder-for-unzipy"),
    ],
)
def test_invalid_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
