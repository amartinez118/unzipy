"""Degenerate inputs raise clear errors or behave sensibly.

Nothing may return garbage silently.
"""

import numpy as np
import pytest

from unzipy.fluctuation import bennett, effective_beta, jarzynski, matching
from unzipy.landscape import coexistence_force, stack_energies
from unzipy.polymer import (
    fjc_compliance,
    fjc_energy,
    fjc_extension,
    fjc_gibbs,
    wlc_extension,
    wlc_force,
)
from unzipy.simulation import HairpinSim
from unzipy.stats import fit_gmm, grubbs_outliers, kde_pdf, optimal_bandwidth
from unzipy.unzipping import (
    align_curves,
    effective_stiffness,
    find_event,
    fit_ssdna_wlc,
)


@pytest.mark.parametrize("p", [0.75, 1.35, 50.0])
def test_wlc_inverse_is_finite_and_monotonic_up_to_absurd_forces(p):
    f = np.logspace(-6, 8, 4000)
    l = wlc_extension(f, p)
    assert (
        np.all(np.isfinite(l))
        and np.all((l >= 0) & (l < 1))
        and np.all(np.diff(l) >= 0)
    )
    ok = f <= 1e4
    np.testing.assert_allclose(wlc_force(l[ok], p), f[ok], rtol=1e-7)


def test_fjc_symmetry_in_force():
    for g in (fjc_gibbs, fjc_energy, fjc_compliance):
        np.testing.assert_allclose(g(-5.0, 1.5, 40.0), g(5.0, 1.5, 40.0))
    np.testing.assert_allclose(
        fjc_extension(-5.0, 1.5, 40.0), -fjc_extension(5.0, 1.5, 40.0)
    )


def test_grubbs_ignores_constant_data():
    assert not grubbs_outliers(np.ones(10)).any()


@pytest.mark.parametrize(
    "call",
    [
        lambda: jarzynski(np.array([])),
        lambda: jarzynski(np.array([1.0, np.nan])),
        lambda: bennett(np.array([]), np.array([1.0])),
        lambda: bennett(np.array([1.0, np.inf]), np.array([1.0])),
        lambda: matching(np.array([1.0]), np.array([-1.0, -2.0])),
        lambda: effective_beta(0.0, 4.0),
        lambda: kde_pdf(np.ones(10), np.array([1.0])),
        lambda: optimal_bandwidth(np.array([1.0, 2.0])),
        lambda: fit_gmm(np.array([1.0, 2.0]), 3),
        lambda: find_event(np.linspace(0, 100, 500), np.zeros(500), "x"),
        lambda: effective_stiffness(
            np.array([0.0, 1.0]), np.array([0.0, 1.0])
        ),
        lambda: fit_ssdna_wlc(
            np.array([5.0, 10.0]), np.array([100.0, 110.0]), 300.0
        ),
        lambda: align_curves(
            np.arange(10.0),
            np.ones(10),
            np.arange(10.0),
            np.ones(10),
            100,
            200,
            200,
        ),
        lambda: stack_energies("GCGA", na=0.0),
        lambda: coexistence_force(-5.0, 20),
        lambda: HairpinSim(()),
        lambda: HairpinSim((-2.0, -2.0, 1.14), f_min=30.0, f_max=5.0),
        lambda: HairpinSim((-2.0, -2.0, 1.14), nt_open=(0, 2)),
    ],
)
def test_degenerate_inputs_raise_value_error(call):
    with pytest.raises(ValueError):
        call()
