import numpy as np
import pytest

from unzipy.landscape import (
    coexistence_force,
    hairpin_stem,
    kcal_to_kT,
    sequential_landscape,
    stack_energies,
)

# per-stack ddG printed by MFOLD for hairpin stacks (DNA, 25 C, 1 M Na+)
MFOLD = {
    "GC": -2.53,
    "CG": -2.50,
    "GG": -2.08,
    "CC": -2.08,
    "AA": -1.27,
    "AT": -1.12,
    "TA": -0.84,
    "CA": -1.72,
    "GA": -1.57,
    "AG": -1.53,
    "CT": -1.53,
}


@pytest.mark.parametrize("dimer", MFOLD)
def test_stack_energies_match_mfold(dimer):
    assert abs(stack_energies(dimer)[0] - MFOLD[dimer]) < 0.015


def test_full_stem_matches_mfold_listing():
    # first 11 stacks of GCGAGCCATAATTTCATCTG against the MFOLD listing
    got = stack_energies("GCGAGCCATAATTTCATCTG")[:11]
    ref = [
        -2.53,
        -2.50,
        -1.57,
        -1.53,
        -2.53,
        -2.08,
        -1.72,
        -1.12,
        -0.84,
        -1.27,
        -1.12,
    ]
    np.testing.assert_allclose(got, ref, atol=0.015)


def test_hairpin_stem_validation():
    assert hairpin_stem("GCGAAAAGCGC"[:0] + "GCGCAAAAGCGC", 4) == "GCGC"
    with pytest.raises(ValueError):
        hairpin_stem("GCGCAAAAGCGA", 4)


def test_kcal_to_kT():
    assert (
        abs(kcal_to_kT(25.0) - 502.9173 / 298.15) < 0.2
    )  # R = 1.9865e-3 kcal/(mol K)


def test_landscape_zero_force_is_stack_sum_and_force_tilts_it():
    stacks, loop = stack_energies("GCGAGCCATAATTTCATCTG"), 1.14
    n, g0 = sequential_landscape(stacks, loop, force=0.0)
    assert n[-1] == stacks.size + 1 and g0[0] == 0
    np.testing.assert_allclose(
        g0[-1], -(stacks.sum() + loop) * kcal_to_kT(25.0), rtol=1e-9
    )
    assert np.all(
        np.diff(g0)[:-1] > 0
    )  # opening costs energy (the last step sheds the loop penalty)
    _, g15 = sequential_landscape(stacks, loop, force=15.0)
    assert g15[-1] < g0[-1]  # force lowers the open state
    assert g15[1] - g15[0] < g0[1] - g0[0]  # and every opening step


def test_coexistence_force_balances_free_energies():
    stacks, loop = stack_energies("GCGAGCCATAATTTCATCTG"), 1.14
    dg0 = -(stacks.sum() + loop) * kcal_to_kT(25.0)
    f = coexistence_force(dg0, stacks.size + 1)
    _, g = sequential_landscape(stacks, loop, force=f)
    assert abs(g[-1]) < 1e-6 and 5 < f < 20


def test_salt_and_parameter_sets():
    s1 = stack_energies("GCGAGC", na=1.0)
    s01 = stack_energies("GCGAGC", na=0.1)
    np.testing.assert_allclose(
        s01 - s1, -0.114 * np.log(0.1)
    )  # lower salt destabilises
    h = stack_energies("GCGAGC", params="huguet2010")
    assert h.shape == s1.shape and np.all(h < 0)
    # reverse-complement-equivalent dimers share a parameter
    np.testing.assert_allclose(stack_energies("AC"), stack_energies("GT"))


def test_landscape_from_sequence_any_length():
    from unzipy.landscape import landscape_from_sequence

    for stem in ("GCGCAGCC", "GCGAGCCATAATTTCATCTG", "GC" * 15):
        seq = (
            stem
            + "TTTT"
            + "".join(
                {"A": "T", "T": "A", "G": "C", "C": "G"}[c] for c in stem[::-1]
            )
        )
        n, g = landscape_from_sequence(seq, force=12.0, loop_dg=1.14)
        assert n.size == len(stem) + 1 and g[0] == 0
