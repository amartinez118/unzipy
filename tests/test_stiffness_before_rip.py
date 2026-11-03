"""Tests of the effective stiffness measured just before the rip."""

import numpy as np
import pytest

from unzipy.constants import KT25
from unzipy.unzipping import (
    Event,
    StretchModel,
    stiffness_before_rip,
)

MODEL = StretchModel(k_b0=0.06, n_ss=44.0, ss_p=1.35, ss_d0=0.59)


def unfolding_curves(n, f_rip=13.0, noise=0.15, seed=0, points=3000):
    """Folded-branch trajectories of the model with a rip at ``f_rip``."""
    rng = np.random.default_rng(seed)
    forces = np.linspace(6.0, f_rip, points)
    lam = MODEL.lam_folded(forces)
    curves, rips = [], []
    for _ in range(n):
        shift = rng.normal(0, 5.0)
        f_noisy = forces + rng.normal(0, noise, points)
        curves.append((lam + shift, f_noisy))
        rips.append(float(lam[-1] + shift))
    return curves, rips


def test_recovers_the_local_slope_of_a_nonlinear_branch():
    curves, rips = unfolding_curves(60)
    st = stiffness_before_rip(curves, rips, window=10.0, guard=1.0, rng=0)
    # exact slope df/dlam at the force of the rip from the model compliance
    f_rip = 13.0
    exact = 1.0 / (MODEL.inv_k_bead(f_rip) + MODEL.inv_k_handles(f_rip))
    assert abs(st.k - exact) < 4 * st.err + 0.001
    assert st.n == 60 and st.slopes.size == 60 and st.err > 0


def test_low_force_slope_is_wrong_for_the_handle_energy():
    curves, rips = unfolding_curves(60, seed=1)
    f_min, f_max = 12.0, 14.0
    w_b, w_h = MODEL.bead_handles_work(f_min, f_max)
    exact = w_b + w_h

    def spring(k):
        return 0.5 * (f_max**2 - f_min**2) / (KT25 * k)

    k_rip = stiffness_before_rip(curves, rips, window=10.0, rng=0).k
    # low-force slope: the first 10 nm of the folded branch
    low = [np.polyfit(x[:300], f[:300], 1)[0] for x, f in curves]
    err_rip = abs(spring(k_rip) - exact) / exact
    err_low = abs(spring(np.median(low)) - exact) / exact
    assert err_rip < 0.03
    assert err_low > 0.05 and err_low > 2.5 * err_rip


def test_error_decreases_with_the_number_of_trajectories():
    c1, r1 = unfolding_curves(10, seed=2, noise=0.3)
    c2, r2 = unfolding_curves(160, seed=3, noise=0.3)
    e1 = stiffness_before_rip(c1, r1, rng=0).err
    e2 = stiffness_before_rip(c2, r2, rng=0).err
    assert e2 < e1


def test_accepts_event_objects_and_skips_short_trajectories():
    curves, rips = unfolding_curves(5, seed=4)
    events = [Event("u", 13.0, r, 0.8, 0) for r in rips]
    a = stiffness_before_rip(curves, events, rng=0)
    b = stiffness_before_rip(curves, rips, rng=0)
    assert a.k == b.k
    short = (np.array([0.0, 1.0]), np.array([1.0, 2.0]))
    c = stiffness_before_rip([*curves, short], [*rips, 1.0], rng=0)
    assert c.n == 5


def test_invalid_inputs_raise():
    curves, rips = unfolding_curves(3, seed=5)
    with pytest.raises(ValueError):
        stiffness_before_rip(curves, rips[:2])
    with pytest.raises(ValueError):
        stiffness_before_rip(curves, rips, window=0.0)
    with pytest.raises(ValueError):
        stiffness_before_rip(curves, rips, guard=-1.0)
    with pytest.raises(ValueError):
        stiffness_before_rip(curves, [r + 5000 for r in rips])
