import numpy as np

from unzipy.unzipping.align import align_curves, discretise, needleman_wunsch


def test_nw_identical_and_gapped():
    a = np.array([3, 5, 7, 9, 4])
    score, ia, ib = needleman_wunsch(a, a)
    assert score == 50 and np.array_equal(ia, ib) and (ia >= 0).all()
    score, ia, ib = needleman_wunsch(
        a, np.delete(a, 2)
    )  # one deletion -> one gap
    assert (
        (ib == -1).sum() == 0
        and (ia == -1).sum() == 0
        or ((ia < 0) | (ib < 0)).sum() == 1
    )


def test_discretise_range():
    lv = discretise(np.linspace(1, 20, 100), 0.9, 20.1)
    assert lv.min() == 1 and lv.max() == 22


_rng = np.random.default_rng(42)
_LEVELS = _rng.uniform(
    5, 20, 140
)  # sequence-like rips: random plateaus every ~3 nm


def sawtooth(x):
    idx = np.clip(
        ((np.asarray(x) + 50) / 3.0).astype(int), 0, _LEVELS.size - 1
    )
    return _LEVELS[idx]


def test_recovers_smooth_extension_shift():
    rng = np.random.default_rng(0)
    xt = np.arange(0, 300, 0.25)
    ft = sawtooth(xt)

    def shift(x):
        return 8.0 + 0.01 * x

    xe = np.arange(0, 300, 0.25)
    fe = sawtooth(xe - shift(xe)) + rng.normal(0, 0.1, xe.size)
    al = align_curves(xe, fe, xt, ft, x_start=0, x_end_exp=290, x_end_theo=290)
    grid = np.linspace(40, 240, 20)
    expected = shift(grid) / (
        1 - 0.01
    )  # shift is defined at the theoretical coordinate
    assert np.median(np.abs(al.shift(grid) - expected)) < 1.0


def test_last_rip_shift_recovers_offset_in_the_high_force_tail():
    from unzipy.unzipping.align import last_rip_shift

    # beyond the aligned range both curves climb monotonically to a
    # high-force end rip
    xt = np.linspace(300, 340, 400)
    ft = 15 + 0.5 * (xt - 300)
    offset = 6.0
    xe = np.linspace(300, 350, 800)
    fe = 15 + 0.5 * (
        xe - offset - 300
    )  # experiment = theory shifted by +offset in x
    x, s = last_rip_shift(xe, fe, xt, ft, 299.0, 299.0, f_threshold=20.0)
    assert x.size > 20 and np.all(np.diff(x) > 0)
    assert abs(np.median(s) - offset) < 0.2
    assert (
        last_rip_shift(xe, fe, xt, ft, 299.0, 299.0, f_threshold=1e3)[0].size
        == 0
    )


def test_align_curves_with_last_rip_appends_points():
    rng = np.random.default_rng(3)
    xt = np.arange(0, 330, 0.25)
    ft = np.where(
        xt < 300, sawtooth(xt), 14 + 0.6 * (xt - 300)
    )  # final rising rip region
    xe = xt.copy()
    fe_core = np.where(
        xe < 300, sawtooth(xe - 5.0), 14 + 0.6 * (xe - 5.0 - 300)
    )
    fe = fe_core + rng.normal(0, 0.1, xe.size)
    plain = align_curves(xe, fe, xt, ft, 0, 290, 290)
    full = align_curves(xe, fe, xt, ft, 0, 290, 290, last_rip=True)
    assert full.x_theory.size > plain.x_theory.size
