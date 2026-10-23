import numpy as np
import pytest

from unzipy.constants import KT25
from unzipy.polymer import (
    ewlc_energy,
    ewlc_extension,
    fjc_compliance,
    fjc_energy,
    fjc_extension,
    fjc_gibbs,
    wlc_compliance,
    wlc_energy,
    wlc_extension,
    wlc_force,
    wlc_gibbs,
)

P = 50.0
F = np.concatenate([[0.0], np.logspace(-4, 3.5, 400)])


@pytest.mark.parametrize("bouchiat", [False, True])
def test_wlc_round_trip(bouchiat):
    l = wlc_extension(F, P, bouchiat=bouchiat)
    assert np.all((l >= 0) & (l < 1))
    np.testing.assert_allclose(
        wlc_force(l[1:], P, bouchiat=bouchiat), F[1:], rtol=1e-7
    )


def test_wlc_limits():
    # low force: l = 2 f p / (3 kT);  high force: l = 1 - 0.5 sqrt(kT / (f p))
    np.testing.assert_allclose(
        wlc_extension(1e-4, P), 2e-4 * P / (3 * KT25), rtol=1e-3
    )
    f = 1e4
    np.testing.assert_allclose(
        wlc_extension(f, P), 1 - 0.5 * np.sqrt(KT25 / (f * P)), rtol=1e-4
    )


def test_wlc_monotonic_and_scalar():
    assert np.all(np.diff(wlc_extension(F, P)) > 0)
    assert np.all(np.diff(wlc_extension(F, P, bouchiat=True)) > 0)
    assert np.ndim(wlc_extension(5.0, P)) == 0


def test_wlc_rejects_bad_input():
    with pytest.raises(ValueError):
        wlc_extension(-1.0, P)
    with pytest.raises(ValueError):
        wlc_force(1.0, P)


def test_wlc_energy_is_integral_of_force():
    Lc, f = 100.0, np.linspace(0, 30, 200001)
    x = Lc * wlc_extension(f, P)
    np.testing.assert_allclose(
        np.trapezoid(f, x) / KT25, wlc_energy(f[-1], Lc, P), rtol=1e-6
    )
    # G = -int x df
    np.testing.assert_allclose(
        -np.trapezoid(x, f) / KT25, wlc_gibbs(f[-1], Lc, P), rtol=1e-6
    )


def test_wlc_compliance_matches_finite_difference():
    f, h = 8.0, 1e-5
    fd = (wlc_extension(f + h, P) - wlc_extension(f - h, P)) / (2 * h)
    np.testing.assert_allclose(wlc_compliance(f, P), fd, rtol=1e-6)


def test_ewlc_reduces_to_wlc():
    np.testing.assert_allclose(
        ewlc_extension(F, P, np.inf), wlc_extension(F, P)
    )


def test_fjc_limits_and_continuity():
    b, L = 1.5, 40.0
    np.testing.assert_allclose(
        fjc_extension(1e-6, b, L), L * b * 1e-6 / (3 * KT25), rtol=1e-6
    )
    f = np.array(
        [0.0, 1e-5, 0.999e-3 * KT25 / b, 1.001e-3 * KT25 / b, 5.0, 1e4]
    )
    assert np.all(np.isfinite(fjc_extension(f, b, L)))
    assert np.all(
        np.isfinite(fjc_gibbs(f, b, L))
    )  # no sinh overflow at large f
    np.testing.assert_allclose(fjc_extension(1e4, b, L), L, rtol=1e-3)


def test_fjc_energy_and_compliance():
    b, L = 2.0, 2.0
    f = np.linspace(0, 40, 400001)
    x = fjc_extension(f, b, L)
    np.testing.assert_allclose(
        np.trapezoid(f, x) / KT25, fjc_energy(f[-1], b, L), rtol=1e-6
    )
    np.testing.assert_allclose(
        -np.trapezoid(x, f) / KT25, fjc_gibbs(f[-1], b, L), rtol=1e-6
    )
    h = 1e-5
    fd = (fjc_extension(3 + h, b, L) - fjc_extension(3 - h, b, L)) / (2 * h)
    np.testing.assert_allclose(fjc_compliance(3.0, b, L), fd, rtol=1e-6)


def test_bouchiat_energy_and_compliance_consistent():
    Lc, f = 100.0, np.linspace(0, 30, 200001)
    x = Lc * wlc_extension(f, P, bouchiat=True)
    np.testing.assert_allclose(
        np.trapezoid(f, x) / KT25,
        wlc_energy(f[-1], Lc, P, bouchiat=True),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        -np.trapezoid(x, f) / KT25,
        wlc_gibbs(f[-1], Lc, P, bouchiat=True),
        rtol=1e-6,
    )
    h = 1e-5
    fd = (
        wlc_extension(8 + h, P, bouchiat=True)
        - wlc_extension(8 - h, P, bouchiat=True)
    ) / (2 * h)
    np.testing.assert_allclose(
        wlc_compliance(8.0, P, bouchiat=True), fd, rtol=1e-6
    )


def test_ewlc_energy_is_integral_of_force():
    Lc, S = 100.0, 1000.0
    f = np.linspace(0, 30, 200001)
    x = Lc * ewlc_extension(f, P, S)
    np.testing.assert_allclose(
        np.trapezoid(f, x) / KT25, ewlc_energy(f[-1], Lc, P, S), rtol=1e-6
    )
