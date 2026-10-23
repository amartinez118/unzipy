import itertools

import numpy as np

from unzipy.constants import KT25
from unzipy.polymer import wlc_extension
from unzipy.unzipping import (
    fit_ssdna_fjc,
    fit_ssdna_wlc,
    rupture_rates,
    rupture_statistics,
)


def test_wlc_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    f = np.linspace(2, 40, 300)
    contour, k = 2 * 0.575 * 1000, 0.077
    x = (
        12.0
        + contour * wlc_extension(f, 0.9)
        + f / k
        + rng.normal(0, 0.5, f.size)
    )
    fit = fit_ssdna_wlc(f, x, contour, k)
    assert abs(fit.params["P"] - 0.9) < 3 * fit.errors["P"] + 0.01
    assert abs(fit.params["shift"] - 12.0) < 3 * fit.errors["shift"] + 0.1
    assert fit.residual_rms < 0.7


def test_fjc_fit_runs_and_is_consistent():
    from unzipy.polymer import fjc_extension

    f = np.linspace(5, 40, 100)
    x = 3.0 + fjc_extension(f, 1.8, 400.0) + f / 0.077
    fit = fit_ssdna_fjc(f, x, 400.0)
    np.testing.assert_allclose(
        [fit.params["shift"], fit.params["b"]], [3.0, 1.8], rtol=1e-4
    )


K0, XB, RATE = (
    1e-3,
    1.0,
    5.0,
)  # Bell-Evans: k(F) = K0 exp(F XB / kT), loading rate 5 pN/s


def bell_sample(n, rng):
    u = rng.uniform(size=n)
    return KT25 / XB * np.log(1 - np.log(u) * RATE * XB / (K0 * KT25))


def test_rupture_rates_recovers_bell_evans():
    rng = np.random.default_rng(1)
    forces = bell_sample(8000, rng)
    rc = rupture_rates(forces, RATE, n_boot=40, rng=2)
    ok = np.isfinite(rc.rate) & (rc.survival > 0.2) & (rc.survival < 0.8)
    F = rc.force[ok]
    true = K0 * np.exp(F * XB / KT25)
    np.testing.assert_allclose(rc.rate[ok], true, rtol=0.25)
    slope = np.polyfit(F, np.log(rc.rate[ok]), 1)[0]
    assert abs(slope - XB / KT25) < 0.1 * XB / KT25
    # survival is normalised
    assert rc.survival[0] > 0.99 and rc.survival[-1] < 0.01


def test_rupture_statistics():
    rng = np.random.default_rng(3)
    st = rupture_statistics(rng.normal(15, 2, 4000))
    assert (
        abs(st.mean - 15) < 0.15
        and abs(st.mode - 15) < 0.7
        and abs(st.std - 2) < 0.1
    )


def test_equilibrium_fdc_force_is_derivative_of_free_energy():
    from unzipy.unzipping import UnzippingSetup, equilibrium_fdc

    rng = np.random.default_rng(5)
    cost = tuple(rng.uniform(1.5, 4.5, 120))  # random pair energies (kT)
    setup = UnzippingSetup(cost)
    lam = np.arange(20.0, 700.0, 1.0)  # trap displacement incl. f/k ~ 200 nm
    fdc = equilibrium_fdc(setup, lam)
    dF = np.gradient(fdc.free_energy, lam) * setup.kT  # pN
    np.testing.assert_allclose(fdc.force[2:-2], dF[2:-2], rtol=1e-3, atol=1e-3)
    assert np.all(np.diff(fdc.mean_open) > -1e-6)  # monotonic opening
    assert fdc.mean_open[0] < 1 and fdc.mean_open[-1] > 115
    plateau = fdc.force[(fdc.mean_open > 20) & (fdc.mean_open < 100)]
    assert (
        8 < plateau.mean() < 25
    )  # typical unzipping force of dsDNA is ~10-20 pN
    np.testing.assert_allclose(fdc.probabilities.sum(axis=0), 1.0)


def test_split_cycles_labels_and_counts(tmp_path):
    from unzipy.io import read_curve
    from unzipy.unzipping import split_cycles, write_segments

    rng = np.random.default_rng(0)
    t = np.arange(0, 100, 0.01)
    tri = 4 + 18 * np.abs(
        ((t / 10.0) % 1.0) * 2 - 1
    )  # 22 -> 4 -> 22 ... period 10 s
    tri = 26 - tri + 0  # start low: 4 at t=0, peaks 22 at t=5
    f = tri + rng.normal(0, 0.2, t.size)
    x = 100 + 10 * f
    segs = split_cycles(t, f, f_min=5.0, f_max=21.0)
    kinds = "".join(s.kind for s in segs)
    assert kinds.count("u") >= 8 and kinds.count("f") >= 8
    assert all(
        a != b for a, b in itertools.pairwise(kinds)
    )  # strictly alternating
    for s in segs:  # u rises, f falls
        d = f[s.slice][-5:].mean() - f[s.slice][:5].mean()
        assert (d > 0) == (s.kind == "u")
    names = write_segments(t, x, f, segs, tmp_path)
    xs, fs = read_curve(tmp_path / f"{names[0]}.txt")
    assert xs.size == fs.size > 100


def test_stretching_fit_recovers_bead_and_handle_work():
    from unzipy.unzipping import StretchModel, fit_stretching_model

    rng = np.random.default_rng(11)
    truth = StretchModel(
        k_b0=0.075, alpha=4e-4, handle_p=22.0, n_ss=52.0, lam0=15.0
    )
    ff, fu = np.linspace(2, 17, 150), np.linspace(10, 24, 150)
    lf = truth.lam_folded(ff) + rng.normal(0, 0.4, ff.size)
    lu = truth.lam_unfolded(fu) + rng.normal(0, 0.4, fu.size)
    fit = fit_stretching_model(
        ff,
        lf,
        fu,
        lu,
        start=StretchModel(),
        free=("k_b0", "alpha", "dP", "lam0", "n_ss"),
    )
    assert fit.rms < 0.6
    f_min, f_max = 2.0, 22.0
    w_true = sum(truth.bead_handles_work(f_min, f_max))
    w_fit = sum(fit.model.bead_handles_work(f_min, f_max))
    assert (
        abs(w_fit - w_true) / w_true < 0.02
    )  # ~1000 kT of stretching known to ~2 %
    assert abs(fit.params["n_ss"] - 52.0) < 3
    # DG0 bookkeeping: planted DG0 is returned
    dg = 65.0 + truth.molecular_work(f_min, f_max) + w_true
    assert abs(truth.zero_force_free_energy(dg, f_min, f_max) - 65.0) < 1e-9


def test_effective_stiffness_limit_of_stretching_work():
    from unzipy.unzipping import StretchModel

    m = StretchModel(
        handle_n=60.0, handle_db=0.34, handle_p=1.5, handle_y=17.0, k_b0=0.065
    )
    f_min, f_max = 13.0, 17.0
    w_b, w_h = m.bead_handles_work(f_min, f_max)
    k_eff = 1.0 / (
        m.inv_k_bead(15.0) + m.inv_k_handles(15.0)
    )  # stiffness in the middle of the range
    approx = (f_max**2 - f_min**2) / (2 * k_eff * m.kT)  # eq. 13
    assert abs((w_b + w_h) - approx) / approx < 0.03


def _rip_curve(rng, kind, x_rip=100.0, jump=1.3):
    x = np.arange(60.0, 160.0, 0.2)
    k = 0.07
    f = 12 + k * (x - 60) * 0.3
    after = x > x_rip
    f = (
        np.where(after, f - jump, f)
        if kind == "u"
        else np.where(after, f + jump, f)
    )
    return x, f + rng.normal(0, 0.12, x.size)


def test_find_event_locates_unfolding_and_folding_rips():
    from unzipy.unzipping import find_event

    rng = np.random.default_rng(21)
    for kind in "uf":
        x, f = _rip_curve(rng, kind)
        ev = find_event(x, f, kind)
        assert abs(ev.extension - 100.0) < 2.0, (kind, ev)
        assert abs(ev.jump - 1.3) < 0.25
        assert abs(ev.force - (f[(x > 98) & (x < 100)].mean())) < 0.4


def test_effective_stiffness_and_released_extension():
    from unzipy.unzipping import effective_stiffness, released_extension

    x = np.linspace(20, 80, 300)
    f = (
        13
        + 0.065 * (x - 20)
        + np.random.default_rng(0).normal(0, 0.05, x.size)
    )
    k, err = effective_stiffness(x, f)
    assert abs(k - 0.065) < 3 * err + 1e-4
    assert (
        abs(released_extension(1.3, 0.065, 15.0) - (1.3 / 0.065 + 1.8)) < 0.5
    )


def test_bin_with_errors():
    from unzipy.stats import bin_with_errors

    x = np.linspace(0, 10, 500)
    g, m, e = bin_with_errors(x, x**2, np.full(x.size, 0.1), n_bins=5)
    assert (
        g.size == m.size == e.size == 5
        and np.all(np.diff(m) > 0)
        and np.all(e > 0)
    )


def test_split_by_status_and_reader(tmp_path):
    from unzipy.unzipping import read_status_file, split_by_status

    header = (
        "#CycleCount/n X_force Y_force Z_force A_dist-Y B_dist-Y "
        "time(sec) Status"
    )
    status = [0, 0, 131, 131, 131, 0, 130, 130, 131, 131]
    rows = [
        f"{i} 0.1 {-5.0 - i} 0.0 {100 + i} {102 + i} {i * 0.01} {s}"
        for i, s in enumerate(status)
    ]
    f = tmp_path / "raw.txt"
    f.write_text("comment line\n" + header + "\n" + "\n".join(rows) + "\n")
    d = read_status_file(f)
    np.testing.assert_allclose(d["x"][:3], [101.0, 102.0, 103.0])
    segs = split_by_status(d["status"])
    assert [(s.kind, s.slice.start, s.slice.stop) for s in segs] == [
        ("u", 2, 5),
        ("f", 6, 8),
        ("u", 8, 10),
    ]


def test_released_nucleotides_inverts_the_wlc():
    from unzipy.polymer import released_nucleotides, wlc_extension

    n = 44.0
    f = np.array([5.0, 12.0, 20.0])
    x = n * 0.59 * wlc_extension(f, 1.35)
    np.testing.assert_allclose(released_nucleotides(f, x, 1.35, 0.59), n)
