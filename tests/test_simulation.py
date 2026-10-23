"""Detailed-balance check of the pulling simulation.

Delta F between two trap positions, obtained through Bennett from forward
and reverse simulated work, must equal the equilibrium value computed
directly from the model landscape.
"""

import numpy as np

from unzipy.fluctuation import bennett, bennett_error
from unzipy.landscape import stack_energies
from unzipy.simulation import HairpinSim

GBP = tuple(np.append(stack_energies("GCGAGCCATAATTTCATCTG"), 1.14))
SIM = HairpinSim(GBP)


def window(sim):
    lam = np.arange(40.0, 450.0, 0.5)
    gN = sim.landscape(lam)[-1]
    return lam[np.argmax(gN < 10)], lam[np.argmax(gN < -10)], lam


def exact_delta_f(sim, l1, l2):
    lam = np.arange(l1, l2 + 1e-9, 0.01)
    folded = np.trapezoid(sim.force(lam, 0), lam) / sim.kT

    def g(x):
        return sim.landscape(np.array([x]))[-1, 0]

    return folded - np.log((1 + np.exp(-g(l2))) / (1 + np.exp(-g(l1))))


def block_work(traj, sim, l1, l2, forward):
    f = traj["f"].reshape(-1, sim.n_meas).mean(axis=1)
    start = traj["lam_start"]
    lam = start + (1 if forward else -1) * sim.dl * np.arange(f.size)
    sel = (lam >= l1) & (lam <= l2)
    return (1 if forward else -1) * np.trapezoid(f[sel], dx=sim.dl) / sim.kT


def test_simulated_work_recovers_equilibrium_free_energy():
    l1, l2, _ = window(SIM)
    assert 50 < l1 < l2 < 400
    fwd = SIM.simulate(300, "forward", rng=1)
    rev = SIM.simulate(300, "reverse", rng=2)
    w_f = np.array([block_work(t, SIM, l1, l2, True) for t in fwd])
    w_r = np.array([block_work(t, SIM, l1, l2, False) for t in rev])
    est, err = bennett(w_f, w_r), bennett_error(w_f, w_r)
    ref = exact_delta_f(SIM, l1, l2)
    assert abs(est - ref) < max(4 * err, 0.5), (est, err, ref)
    assert w_f.mean() > ref > -w_r.mean()  # second law


def test_trajectories_have_the_expected_shape_and_rupture():
    traj = SIM.simulate(20, "forward", rng=3)
    for t in traj:
        assert t["lam"].size == t["f"].size == t["sigma"].size
        assert (
            t["sigma"][0] == 0 and t["sigma"][-1] == 1
        )  # starts folded, ends unfolded
    ruptures = [t["f"][np.argmax(t["sigma"] == 1)] for t in traj]
    assert 8 < np.mean(ruptures) < 22
    rev = SIM.simulate(20, "reverse", rng=4)
    assert all(t["sigma"][0] == 1 and t["sigma"][-1] == 0 for t in rev)


def test_seeded_runs_reproduce():
    a = SIM.simulate(3, "forward", rng=7)[0]["f"]
    b = SIM.simulate(3, "forward", rng=7)[0]["f"]
    np.testing.assert_array_equal(a, b)


def test_explicit_nt_open_matches_default_and_reader(tmp_path):
    from unzipy.io import read_mfold_input

    nb = len(GBP)
    explicit = HairpinSim(
        GBP, nt_open=tuple([2 * n for n in range(nb)] + [2 * nb + 4])
    )
    lam = np.array([100.0, 200.0, 300.0])
    np.testing.assert_allclose(explicit.landscape(lam), SIM.landscape(lam))

    lines = [f"1 -30 {nb} molecule"]
    lines += [f"{2 * i} {g} {i}" for i, g in enumerate(GBP)] + [
        f"{2 * nb + 4} 0.0 {nb}"
    ]
    lines += ["2 -10 2 molecule", "0 -2.5 0", "4 1.0 1", "8 0.0 2"]
    f = tmp_path / "input.txt"
    f.write_text("\n".join(lines) + "\n")
    mols = read_mfold_input(f)
    assert (
        len(mols) == 2
        and mols[0]["gbp"] == GBP
        and mols[0]["nt_open"][-1] == 2 * nb + 4
    )
    assert mols[1]["nt_open"] == (0, 4, 8) and mols[1]["gbp"] == (-2.5, 1.0)
    HairpinSim(mols[1]["gbp"], nt_open=mols[1]["nt_open"]).landscape(
        np.array([100.0])
    )
