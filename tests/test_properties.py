"""Property-based tests of invariants over wide parameter ranges."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from unzipy.fluctuation.bias import (
    bias,
    mu_from_omega,
    omega_from_mu,
)
from unzipy.fluctuation.estimators import bennett, jarzynski
from unzipy.polymer.fjc import fjc_extension
from unzipy.polymer.wlc import wlc_extension, wlc_force

SETTINGS = settings(max_examples=100, deadline=None)

persistence = st.floats(0.3, 60.0)
forces = st.floats(0.0, 400.0)
relative = st.floats(0.0, 0.999)
work = st.lists(st.floats(-5.0, 60.0), min_size=2, max_size=60)


@SETTINGS
@given(l=relative, p=persistence, bouchiat=st.booleans())
def test_wlc_force_then_extension_round_trips(l, p, bouchiat):
    f = wlc_force(l, p, bouchiat=bouchiat)
    back = wlc_extension(f, p, bouchiat=bouchiat)
    assert abs(float(back) - l) < 1e-6


@SETTINGS
@given(f=forces, p=persistence, bouchiat=st.booleans())
def test_wlc_extension_is_within_contour_and_monotonic(f, p, bouchiat):
    x0 = float(wlc_extension(f, p, bouchiat=bouchiat))
    x1 = float(wlc_extension(f + 0.5, p, bouchiat=bouchiat))
    assert 0.0 <= x0 <= 1.0 and x1 >= x0 - 1e-12


@SETTINGS
@given(f=st.floats(0.01, 300.0), b=st.floats(0.2, 10.0), L=st.floats(10, 1e3))
def test_fjc_is_bounded_odd_and_monotonic(f, b, L):
    x = float(fjc_extension(f, b, L))
    assert 0.0 < x < L
    assert abs(float(fjc_extension(-f, b, L)) + x) < 1e-9 * L
    assert float(fjc_extension(f * 1.1, b, L)) > x


@SETTINGS
@given(w=work)
def test_jarzynski_lies_between_min_and_mean(w):
    w = np.array(w)
    dg = float(jarzynski(w))
    assert w.min() - 1e-9 <= dg <= w.mean() + 1e-9


@SETTINGS
@given(w=work, shift=st.floats(-20, 20))
def test_jarzynski_shifts_with_the_work(w, shift):
    w = np.array(w)
    assert (
        abs(float(jarzynski(w + shift)) - float(jarzynski(w)) - shift) < 1e-8
    )


@SETTINGS
@given(w_f=work, w_r=work)
def test_bennett_is_antisymmetric_under_swapping_directions(w_f, w_r):
    a = float(bennett(w_f, w_r))
    b = float(bennett(w_r, w_f))
    assert abs(a + b) < 1e-6


@SETTINGS
@given(w_f=work, w_r=work, shift=st.floats(-20, 20))
def test_bennett_shifts_with_the_work(w_f, w_r, shift):
    base = float(bennett(w_f, w_r))
    moved = float(bennett(np.array(w_f) + shift, np.array(w_r) - shift))
    assert abs(moved - base - shift) < 1e-6


@SETTINGS
@given(
    n=st.integers(2, 10**6),
    mu=st.floats(0.1, 30.0),
)
def test_jarzynski_bias_is_never_negative(n, mu):
    assert bias(n, mu) >= 0.0


@SETTINGS
@given(omega=st.floats(0.1, 50.0), delta=st.floats(1.2, 4.0))
def test_tail_parameter_conversions_are_inverse(omega, delta):
    mu = mu_from_omega(omega, delta)
    assert abs(omega_from_mu(mu, delta) - omega) < 1e-8 * omega
