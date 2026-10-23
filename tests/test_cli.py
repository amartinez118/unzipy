import json

import numpy as np
import pytest

from unzipy.cli import main
from unzipy.constants import KT25


def test_cli_simulate_then_neqwork(tmp_path, capsys):
    out = tmp_path / "sim"
    assert (
        main(
            [
                "simulate",
                "GCGCGCGCAAAAGCGCGCGC",
                "--loop-nt",
                "4",
                "--loop-dg",
                "1.14",
                "-n",
                "6",
                "--seed",
                "1",
                "-o",
                str(out),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        len(list(out.glob("u*.txt"))) == 6
        and len(list(out.glob("f*.txt"))) == 6
    )


def test_cli_neqwork_position_reports_free_energy(tmp_path, capsys):
    k = 0.2
    x = np.linspace(0, 40, 801)
    for name, xs in [("u1", x), ("u2", x), ("f1", x[::-1]), ("f2", x[::-1])]:
        np.savetxt(
            tmp_path / f"{name}.txt",
            np.column_stack(
                [np.arange(xs.size), xs, np.zeros(xs.size), k * xs]
            ),
        )
    main(
        [
            "neqwork-position",
            str(tmp_path),
            "--x-range",
            "10",
            "30",
            "--k-eff",
            "0.2",
            "--f-min",
            "2",
            "--f-max",
            "6",
            "--n-bp",
            "20",
            "--loop-nt",
            "4",
            "--preset",
            "dna",
        ]
    )
    res = json.loads(capsys.readouterr().out)
    expected = 0.5 * k * (30**2 - 10**2) / KT25
    assert res["n_forward"] == 2 and abs(res["dG_bennett"] - expected) < 1e-3
    assert "dG0" in res


def test_cli_requires_command():
    with pytest.raises(SystemExit):
        main([])


def test_cli_rejects_incomplete_molecule_and_missing_forces(tmp_path, capsys):
    x = np.linspace(0, 40, 101)
    for name, xs in [("u1", x), ("f1", x[::-1])]:
        np.savetxt(
            tmp_path / f"{name}.txt",
            np.column_stack(
                [np.arange(xs.size), xs, np.zeros(xs.size), 0.2 * xs]
            ),
        )
    base = [
        "neqwork-position",
        str(tmp_path),
        "--x-range",
        "10",
        "30",
        "--loop-nt",
        "4",
    ]
    # no molecule length
    assert main([*base, "--preset", "dna"]) == 2
    assert "exactly one of --sequence and --n-bp" in capsys.readouterr().err
    # no single-strand elasticity
    assert main([*base, "--n-bp", "20"]) == 2
    assert "--preset" in capsys.readouterr().err
    # k_eff without branch-fit forces
    assert (
        main([*base, "--n-bp", "20", "--preset", "dna", "--k-eff", "0.07"])
        == 2
    )
    assert "--f-min" in capsys.readouterr().err
    # sequence inconsistent with the loop
    assert main([*base, "--sequence", "GCGCGCG", "--preset", "rna"]) == 2
    capsys.readouterr()
    # the molecule length reaches the model: 2 * n_bp + loop nucleotides


def test_cli_force_molecule_length_reaches_the_model(tmp_path, capsys):
    import json

    f = np.linspace(5, 20, 200)
    for name, ff in [("u1", f), ("u2", f), ("f1", f[::-1]), ("f2", f[::-1])]:
        np.savetxt(
            tmp_path / f"{name}.txt", np.column_stack([ff, 10.0 + 5.0 * ff])
        )
    base = [
        "neqwork-force",
        str(tmp_path),
        "--f-range",
        "5",
        "20",
        "--loop-nt",
        "4",
        "--preset",
        "dna",
    ]
    main([*base, "--n-bp", "20"])
    assert (
        json.loads(capsys.readouterr().out)["released_nt"] == 44
    )  # 20 bp + 4-nt loop
    main([*base, "--n-bp", "36"])
    assert json.loads(capsys.readouterr().out)["released_nt"] == 76
    main(
        [*base, "--sequence", "GC" * 15 + "TTTT" + "GC" * 15]
    )  # 64-nt hairpin
    assert json.loads(capsys.readouterr().out)["released_nt"] == 64
