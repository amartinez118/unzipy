import numpy as np
import pytest

from unzipy.io import parse_mc_log
from unzipy.landscape import (
    count_mismatches,
    find_duplicates,
    random_mutant_library,
    reverse_complement,
)


def test_library_and_duplicates():
    assert reverse_complement("GCGA") == "TCGC"
    assert (
        count_mismatches("GCGA", "TCGC") == 0
        and count_mismatches("GCGA", "TCGA") == 1
    )
    template = "GCNNNNAAAANNNNGC"
    seqs, mism = random_mutant_library(2000, template, rng=1)
    assert all(len(s) == len(template) and "N" not in s for s in seqs)
    # 4 mutable sites per arm: mismatch count ~ Binomial(4, 3/4)
    assert abs(mism.mean() - 3.0) < 0.1 and mism.max() <= 4
    assert find_duplicates(["AAA", "ccc", "AAA", "CCC", "GGG"]) == [2, 3]
    with pytest.raises(ValueError):
        random_mutant_library(3, "ACGT")


def test_parse_mc_log(tmp_path):
    f = tmp_path / "mc.txt"
    f.write_text(
        "header line 1\nheader line 2\n"
        "E1 -1.5 E2 -1.7 Eloop 1.9 rejected\n"
        "T 0 accepted (downhill) 2.5e-1\n"
        "\n"
        "E1 -1.6 E2 -1.8 Eloop 1.8 accepted\n"
        "chi2 0.22\n"
        "E1 -9e-1\n"  # incomplete last step; scientific notation is kept
    )
    out = parse_mc_log(f, n_params=4, skip_lines=2)
    np.testing.assert_allclose(
        out, [[-1.5, -1.7, 1.9, 0.25], [-1.6, -1.8, 1.8, 0.22]]
    )
