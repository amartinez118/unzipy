"""Additional tests for code paths that need an independent check."""

import numpy as np
import pytest

from unzipy.cli import main
from unzipy.constants import KT25
from unzipy.fluctuation import work_force, work_position
from unzipy.io import read_table
from unzipy.unzipping import (
    UnzippingSetup,
    equilibrium_fdc,
    needleman_wunsch,
    read_tweezers_trace,
    rupture_rates,
)


def test_refolding_rates_recover_bell_evans_in_the_folding_direction():
    """Recover Bell-Evans rates for refolding, mirroring the unfolding test.

    The force decreases at rate r and k_UN(F) = k0 exp(-F x / kT).
    """
    k0, x, r, f0 = 5.0, 1.0, 5.0, 40.0
    rng = np.random.default_rng(4)
    u = rng.uniform(size=8000)
    e = np.exp(-f0 * x / KT25) - r * x / (k0 * KT25) * np.log(u)
    forces = -KT25 / x * np.log(e)  # inverse of the survival function S(F)
    rc = rupture_rates(forces, r, unfolding=False, n_boot=40, rng=5)
    ok = np.isfinite(rc.rate) & (rc.survival > 0.2) & (rc.survival < 0.8)
    F = rc.force[ok]
    np.testing.assert_allclose(
        rc.rate[ok], k0 * np.exp(-F * x / KT25), rtol=0.3
    )


def test_numba_and_python_needleman_wunsch_agree():
    from unzipy.unzipping import align

    rng = np.random.default_rng(0)
    a, b = (
        rng.integers(1, 23, 300).astype(float),
        rng.integers(1, 23, 280).astype(float),
    )
    fast = needleman_wunsch(a, b)
    slow = getattr(align._nw, "py_func", align._nw)(a, b, 50.0, 10.0)
    assert fast[0] == slow[0]
    np.testing.assert_array_equal(fast[1], slow[1])
    np.testing.assert_array_equal(fast[2], slow[2])


def test_work_definitions_for_a_linear_spring():
    k = 0.1
    x = np.linspace(2, 30, 400)
    f = k * x
    np.testing.assert_allclose(
        work_position(x, f), 0.5 * k * (x[-1] ** 2 - x[0] ** 2) / KT25
    )
    np.testing.assert_allclose(
        work_force(x, f), -(f[-1] ** 2 - f[0] ** 2) / (2 * k * KT25)
    )
    # Legendre relation W_x = W_f + (x1 f1 - x0 f0)/kT
    np.testing.assert_allclose(
        work_position(x, f),
        work_force(x, f) + (x[-1] * f[-1] - x[0] * f[0]) / KT25,
    )


@pytest.mark.parametrize("strand", ["fjc", "wlc"])
def test_equilibrium_fdc_both_strand_models(strand):
    setup = UnzippingSetup(tuple(np.full(40, 3.0)), strand=strand)
    lam = np.arange(30.0, 400.0, 2.0)
    fdc = equilibrium_fdc(setup, lam)
    dF = np.gradient(fdc.free_energy, lam) * setup.kT
    np.testing.assert_allclose(fdc.force[3:-3], dF[3:-3], rtol=2e-3, atol=2e-3)


def test_io_readers(tmp_path):
    f = tmp_path / "comma.txt"
    f.write_text("# header\n1,5 2,25\n3,0 4,5\n")
    np.testing.assert_allclose(
        read_table(f, decimal=","), [[1.5, 2.25], [3.0, 4.5]]
    )
    raw = tmp_path / "trace.txt"
    cols = np.zeros((6, 31))
    cols[:, 17] = [-5, -6, -7, 8, 9, 10]  # force (sign ignored)
    cols[:, 24], cols[:, 26] = 100.0, 102.0  # trap positions
    cols[:, 30] = np.arange(6.0)
    raw.write_text(
        "# comment\n" + "\n".join(" ".join(map(str, r)) for r in cols) + "\n"
    )
    _t, x, f_ = read_tweezers_trace(raw)
    np.testing.assert_allclose(f_, [5, 6, 7, 8, 9, 10])
    np.testing.assert_allclose(x, 101.0)
    t2, _, _ = read_tweezers_trace(raw, t_range=(1.5, 4.5))
    np.testing.assert_allclose(t2, [2, 3, 4])


def test_cli_force_split_and_module_entry(tmp_path, capsys):
    import json
    import subprocess
    import sys

    # force-controlled: quasi-static ramps (a linear x(f) is enough to
    # exercise the code path)
    f = np.linspace(5, 20, 300)
    for name, ff in [("u1", f), ("u2", f), ("f1", f[::-1]), ("f2", f[::-1])]:
        np.savetxt(
            tmp_path / f"{name}.txt", np.column_stack([ff, 10.0 + 5.0 * ff])
        )
    main(
        [
            "neqwork-force",
            str(tmp_path),
            "--f-range",
            "5",
            "20",
            "--n-bp",
            "20",
            "--loop-nt",
            "4",
            "--preset",
            "dna",
        ]
    )
    res = json.loads(capsys.readouterr().out)
    assert res["n_forward"] == 2 and res["n_reverse"] == 2 and "dG0" in res
    # split a synthetic cyclic trace
    t = np.arange(0, 60, 0.01)
    force = 12.5 + 8.5 * np.sin(2 * np.pi * t / 20 - np.pi / 2) * 1.2
    cols = np.zeros((t.size, 31))
    cols[:, 17], cols[:, 24], cols[:, 26], cols[:, 30] = (
        force,
        100 + 5 * force,
        100 + 5 * force,
        t,
    )
    raw = tmp_path / "raw.txt"
    np.savetxt(raw, cols)
    out = tmp_path / "events"
    main(["split", str(raw), "--f-min", "5", "--f-max", "20", "-o", str(out)])
    res = json.loads(capsys.readouterr().out)
    assert res["segments"] >= 4 and res["unfolding"] >= 2
    # `python -m unzipy` works as an entry point and importing __main__ has
    # no side effects
    p = subprocess.run(
        [sys.executable, "-m", "unzipy", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0 and "unzipy" in p.stdout
    import unzipy.__main__  # noqa: F401  (must not run the CLI)
