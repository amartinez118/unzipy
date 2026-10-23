import numpy as np
import pytest

from unzipy.constants import KT25
from unzipy.fluctuation.position_ensemble import (
    PullingModel,
    contributions,
    estimate_free_energy,
    works_from_directory,
    zero_force_free_energy,
)


def write_curve(path, x, f):
    np.savetxt(
        path, np.column_stack([np.arange(len(x)), x, np.zeros(len(x)), f])
    )


def test_works_from_directory_signs_and_range(tmp_path):
    k = 0.2  # linear spring f = k x
    x = np.linspace(0, 40, 801)
    write_curve(tmp_path / "u1.txt", x, k * x)
    write_curve(
        tmp_path / "f1.txt", x[::-1], k * x[::-1]
    )  # refolding: x decreasing
    write_curve(tmp_path / "bad.txt", x, k * x)  # ignored (no u/f prefix)
    write_curve(
        tmp_path / "fdc.txt", x, k * x
    )  # ignored (starts with f but is not f<N>.txt)
    write_curve(tmp_path / "foo.txt", x, k * x)
    x[5] = np.nan
    write_curve(tmp_path / "u2.txt", x, k * x)  # skipped (NaN)
    w_f, w_r, fmin, fmax = works_from_directory(tmp_path, (10, 30))
    expected = 0.5 * k * (30**2 - 10**2) / KT25
    np.testing.assert_allclose(w_f, [expected], rtol=1e-6)
    np.testing.assert_allclose(w_r, [-expected], rtol=1e-6)
    np.testing.assert_allclose([fmin, fmax], [k * 10, k * 30], atol=1e-9)


def test_zero_force_free_energy_recovers_planted_value():
    rng = np.random.default_rng(0)
    m = PullingModel()
    g_ss, g_h, g_d = contributions(3.0, 15.0, m, k_eff=0.07)
    g0, sig = 40.0, 1.5
    dg = g0 + g_ss + g_h - g_d  # measured DG for planted DG0
    w_f = rng.normal(dg + sig**2 / 2, sig, 3000)
    w_r = rng.normal(-dg + sig**2 / 2, sig, 3000)
    est, err = zero_force_free_energy(w_f, w_r, 3.0, 15.0, m, k_eff=0.07)
    assert abs(est - g0) < 3 * err and err < 0.2
    fe = estimate_free_energy(w_f, w_r)
    assert (
        abs(fe.wdis_f - sig**2 / 2) < 0.15
        and abs(fe.wdis_r - sig**2 / 2) < 0.15
    )


def test_handles_option_validation():
    with pytest.raises(ValueError):
        contributions(3.0, 15.0)  # spring needs k_eff
    assert contributions(3.0, 15.0, handles="wlc")[1] > 0


def test_published_table1_dna_short():
    """Reproduce Table 1 of Severino et al., J. Stat. Mech. (2019) 124001.

    CD4 DNA with short handles: DW_m = 17.4 kT and DW_h + DW_b = 225 kT for
    (f_min, f_max) = (13, 17) pN and k_eff = 0.065 pN/nm.
    """
    from unzipy.fluctuation.position_ensemble import (
        PullingModel,
        contributions,
    )

    g_ss, g_h, g_d = contributions(
        13.0, 17.0, PullingModel.dna_short(), k_eff=0.065
    )
    assert abs((g_ss - g_d) - 17.4) < 0.5
    assert abs(g_h - 225) < 3
    g_ss, _, g_d = contributions(
        18.0, 22.0, PullingModel.rna_long(), k_eff=0.067
    )
    assert abs((g_ss - g_d) - 28) < 1  # RNA long: DW_m = 28 kT


def test_models_follow_the_sequence_length():
    from unzipy.fluctuation import ForceModel, PullingModel
    from unzipy.simulation import HairpinSim

    for stem_len in (8, 20, 33):
        seq = (
            "G" * stem_len + "TTTT" + "C" * stem_len
        )  # any length: 2*stem + loop
        m = PullingModel.from_sequence(seq, loop_nt=4, ss_d0=0.59)
        assert (
            m.n_bases == stem_len
            and abs(m.ss_contour - (2 * stem_len + 4) * 0.59) < 1e-9
        )
        f = ForceModel.from_sequence(seq, loop_nt=4)
        assert f.ss_nt == len(seq)
    sim = HairpinSim.from_sequence(
        "GCGCAGCCTTTTGGCTGCGC", loop_nt=4, loop_dg=1.14
    )  # 8 bp
    assert sim.n_bp == 8 and sim.nloop == 4
    with pytest.raises(ValueError):
        PullingModel.from_sequence("GCGCGCG", loop_nt=4)  # odd stem


def test_two_handles_are_counted_twice():
    from unzipy.fluctuation.position_ensemble import (
        PullingModel,
        contributions,
    )

    one = contributions(3.0, 15.0, PullingModel(n_handles=1), handles="wlc")[1]
    two = contributions(3.0, 15.0, PullingModel(n_handles=2), handles="wlc")[1]
    assert abs(two - 2 * one) < 1e-12


def test_interpolated_edges_make_the_work_exact_for_coarse_sampling(tmp_path):
    k = 0.2
    x = np.arange(
        0.3, 40.0, 1.7
    )  # coarse, offset: no sample lands on the range limits
    write_curve(tmp_path / "u1.txt", x, k * x)
    write_curve(tmp_path / "f1.txt", x[::-1], k * x[::-1])
    exact = 0.5 * k * (30.0**2 - 10.0**2) / KT25
    w_f, w_r, _, _ = works_from_directory(tmp_path, (10, 30))
    np.testing.assert_allclose([w_f[0], w_r[0]], [exact, -exact], rtol=1e-9)
    w_f2, _, _, _ = works_from_directory(
        tmp_path, (10, 30), interpolate_edges=False
    )
    assert abs(w_f2[0] - exact) > 1e-3 * exact  # samples alone miss the edges
