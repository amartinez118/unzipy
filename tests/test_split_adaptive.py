"""Tests of the adaptive cycle splitter on synthetic pulling traces."""

import numpy as np
import pytest

from unzipy.unzipping import split_cycles, split_cycles_adaptive

DT = 0.001  # s


def synthetic_trace(blocks, seed=0, noise=0.15, lag=15):
    """Triangular force cycles with changing limits and a turn-around lag.

    Args:
        blocks: list of (n_cycles, f_min, f_max, samples_per_leg).
        seed: random seed.
        noise: force noise (pN).
        lag: samples the force keeps rising after the trap has reversed.

    Returns:
        t, f, x and the sample index of every true turning point (the
        position extremum) with its kind.
    """
    rng = np.random.default_rng(seed)
    f_parts, x_parts, turns, n0 = [], [], [], 0
    x_pos = 0.0
    for n_cycles, lo, hi, leg in blocks:
        for _ in range(n_cycles):
            for going_up in (True, False):
                a, b = (lo, hi) if going_up else (hi, lo)
                ramp = np.linspace(a, b, leg, endpoint=False)
                f_parts.append(ramp)
                x_parts.append(x_pos + (ramp - a) / (hi - lo) * 100.0)
                turns.append((n0, "min" if going_up else "max"))
                n0 += leg
                x_pos = x_parts[-1][-1] if going_up else 0.0
    f = np.concatenate(f_parts) + rng.normal(0, noise, n0)
    x = np.concatenate(x_parts)
    x = np.abs(np.diff(np.concatenate([[0], x]))).cumsum() * 0 + x
    t = np.arange(n0) * DT
    return t, f, x, turns


def test_adaptive_finds_every_cycle_when_the_limits_change():
    blocks = [(15, 8.0, 15.0, 1500), (15, 5.0, 20.0, 2500)]
    t, f, x, turns = synthetic_trace(blocks)
    segs = split_cycles_adaptive(t, f)
    # 30 cycles -> 59 complete segments between 60 turning points
    assert len(segs) >= 58
    kinds = [s.kind for s in segs]
    assert all(a != b for a, b in zip(kinds, kinds[1:], strict=False))
    # thresholds suited to the wide block are never reached by the narrow
    # first block, so a fixed pair of limits loses half of the cycles
    fixed = split_cycles(t, f, f_min=5.5, f_max=19.5, gap=0.5)
    assert len(fixed) <= 32


def test_boundaries_agree_with_the_true_turning_points():
    t, f, x, turns = synthetic_trace([(20, 8.0, 15.0, 2000)], noise=0.1)
    segs = split_cycles_adaptive(t, f)
    starts = np.array([s.slice.start for s in segs])
    truth = np.array([i for i, _ in turns])
    err = np.array([np.min(np.abs(truth - s)) for s in starts])
    assert (
        np.median(err) < 100
    )  # smoothed force: within a few percent of a leg
    assert err.max() < 400


def test_position_refinement_sharpens_the_turning_points():
    # the trap reverses exactly at the turning sample; the force plateau
    # around it is noisy, so the position is the better locator
    t, f, x, turns = synthetic_trace([(20, 8.0, 15.0, 2000)], noise=0.3)
    truth = np.array([i for i, _ in turns])
    plain = split_cycles_adaptive(t, f)
    refined = split_cycles_adaptive(t, f, x=x)
    e_plain = np.array([np.min(np.abs(truth - s.slice.start)) for s in plain])
    e_ref = np.array([np.min(np.abs(truth - s.slice.start)) for s in refined])
    assert np.median(e_ref) <= np.median(e_plain)
    assert np.median(e_ref) <= 5
    assert [s.kind for s in refined] == [s.kind for s in plain]


def test_debris_after_the_protocol_is_ignored():
    t, f, x, _ = synthetic_trace([(10, 8.0, 15.0, 1500)])
    rng = np.random.default_rng(3)
    debris = rng.normal(-1.0, 0.2, 5000)  # tether lost
    f2 = np.concatenate([f, debris])
    t2 = np.arange(f2.size) * DT
    segs = split_cycles_adaptive(t2, f2)
    assert 18 <= len(segs) <= 20
    assert max(s.slice.stop for s in segs) <= f.size + 10


def test_flat_trace_has_no_segments():
    t = np.arange(5000) * DT
    f = np.random.default_rng(1).normal(10.0, 0.05, t.size)
    assert split_cycles_adaptive(t, f, prominence=2.0) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(smooth=0),
        dict(min_gap=0.0),
        dict(refine=-1),
    ],
)
def test_invalid_parameters_raise(kwargs):
    t = np.arange(5000) * DT
    f = np.sin(t)
    with pytest.raises(ValueError):
        split_cycles_adaptive(t, f, **kwargs)


def test_shape_and_length_checks():
    t = np.arange(100) * DT
    with pytest.raises(ValueError):
        split_cycles_adaptive(t, np.zeros(99))
    with pytest.raises(ValueError):
        split_cycles_adaptive(t, np.zeros(100))  # shorter than 3 * smooth
    t = np.arange(5000) * DT
    with pytest.raises(ValueError):
        split_cycles_adaptive(t, np.zeros(5000), x=np.zeros(10))


def test_default_prominence_survives_huge_excursions_and_rip_bumps():
    t, f, x, _ = synthetic_trace([(25, 8.0, 15.0, 1500)], noise=0.2)
    f = f.copy()
    # a rip every cycle: a 0.8 pN dip, and a few tens-of-pN excursions
    for k in range(25):
        c = 3000 * k + 1000
        f[c : c + 40] -= 0.8
    for c in (5000, 20000, 41000):
        f[c : c + 150] += 45.0
    t = np.arange(f.size) * DT
    segs = split_cycles_adaptive(t, f)
    cycles = [s for s in segs if s.slice.stop - s.slice.start > 1000]
    assert len(cycles) >= 40  # the 50 true legs, minus those hit by spikes
