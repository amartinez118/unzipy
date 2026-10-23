"""Independent check of the force-ensemble pipeline.

Builds a quasi-static two-state force protocol from the elastic model,
integrates Gibbs energies and works numerically, and requires the
pipeline to recover DG0.
"""

import numpy as np
from scipy.optimize import brentq

from unzipy.fluctuation import force_ensemble as fe
from unzipy.polymer import fjc_extension, wlc_extension

M = fe.ForceModel()
F_MIN, F_MAX = 4.0, 22.0


def x_folded(f):
    return fe.folded_extension(f, M)


def x_unfolded(f):
    return M.n_handles * M.handle_length * wlc_extension(
        f, M.handle_p, M.kT
    ) + M.ss_contour * wlc_extension(f, M.ss_p, M.kT)


def gibbs_numeric(xfun, f, n=200_001):
    g = np.linspace(0, f, n)
    return -np.trapezoid(xfun(g), g) / M.kT


def write_protocol(tmp_path, dg0, shifted_by):
    fc = brentq(
        lambda f: (
            dg0 + gibbs_numeric(x_unfolded, f) - gibbs_numeric(x_folded, f)
        ),
        1,
        60,
    )
    assert F_MIN < fc < F_MAX
    up = np.concatenate(
        [np.linspace(F_MIN, fc, 4001), np.linspace(fc, F_MAX, 4001)]
    )
    xs = (
        np.concatenate([x_folded(up[:4001]), x_unfolded(up[4001:])])
        - shifted_by
    )
    for name, f, x in [("u1", up, xs), ("f1", up[::-1], xs[::-1])]:
        np.savetxt(tmp_path / f"{name}.txt", np.column_stack([f, x]))
    return fc


def test_recovers_dg0_from_quasi_static_protocol(tmp_path):
    dg0 = 55.0
    write_protocol(tmp_path, dg0, shifted_by=float(x_folded(F_MIN)))
    w_f, w_r = fe.works_from_directory(tmp_path, (F_MIN, F_MAX))
    est, _err = fe.zero_force_free_energy(w_f, w_r, F_MIN, F_MAX, M)
    np.testing.assert_allclose(est, dg0, atol=1e-2)


def test_reverse_work_has_opposite_sign_and_unsorted_matches(tmp_path):
    write_protocol(tmp_path, 55.0, shifted_by=float(x_folded(F_MIN)))
    w_f, w_r = fe.works_from_directory(tmp_path, (F_MIN, F_MAX))
    w_f2, w_r2 = fe.works_from_directory(
        tmp_path, (F_MIN, F_MAX), sort_by_force=False
    )
    assert (
        w_f[0] < 0 < w_r[0]
    )  # measured relative to x0: -int x df is negative going up
    np.testing.assert_allclose(
        [w_f, w_r], [w_f2, w_r2], atol=2e-2
    )  # tie ordering at the jump


def test_fjc_diameter_branch_is_used():
    assert (
        fjc_extension(F_MIN, 2.0, 2.0) > 0
        and fe.gibbs_terms(F_MIN, F_MAX)[2] < 0
    )
