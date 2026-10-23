"""Hairpin libraries: random stem mutants, mismatches and duplicates.

Folding the mutants needs a structure-prediction tool (e.g. MFOLD); read its
output with :func:`unzipy.io.read_mfold_input`.
"""

from __future__ import annotations

import numpy as np

_COMP = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence.

    Args:
        seq (str): Sequence of A, C, G, T.

    Returns:
        str: Reverse complement.
    """
    return seq.upper().translate(_COMP)[::-1]


def count_mismatches(arm5, arm3):
    """Count the mismatched pairs of a stem.

    Args:
        arm5 (str): 5' arm, written 5' to 3'.
        arm3 (str): 3' arm, written 5' to 3'.

    Returns:
        int: Number of non-complementary pairs.

    Raises:
        ValueError: If the arms have different lengths.
    """
    if len(arm5) != len(arm3):
        raise ValueError("arms must have the same length")
    return sum(
        a != b
        for a, b in zip(reverse_complement(arm5), arm3.upper(), strict=True)
    )


def random_mutant_library(n, template, positions=None, n_mut=8, rng=None):
    """Generate random hairpin variants of a template.

    Args:
        n (int): Number of variants.
        template (str): Hairpin (5' arm + loop + 3' arm) as one string.
        positions (list[int] | None): Indices of the ``2 * n_mut`` mutable
            sites, the first half on the 5' arm and the second half on the 3'
            arm. If None, the ``N`` placeholders of the template mark the
            sites.
        n_mut (int): Mutable sites per arm (unused when ``positions`` is
            given).
        rng (int | numpy.random.Generator | None): Seed or generator.

    Returns:
        tuple[list[str], numpy.ndarray]: The sequences and, for each, the
        number of non-complementary pairs among the mutated sites.

    Raises:
        ValueError: If no mutable sites are defined or their number is odd.
    """
    gen = np.random.default_rng(rng)
    template = template.upper()
    if positions is None:
        positions = [i for i, c in enumerate(template) if c == "N"]
        if not positions:
            raise ValueError(
                "give `positions` or mark mutable sites with N in the template"
            )
    if len(positions) % 2:
        raise ValueError(
            "need an even number of mutable sites (5' arm half, 3' arm half)"
        )
    half = len(positions) // 2
    bases = np.array(list("ACGT"))
    seqs, mism = [], np.empty(n, dtype=int)
    for k in range(n):
        draw = bases[gen.integers(0, 4, size=len(positions))]
        s = list(template)
        for pos, b in zip(positions, draw, strict=True):
            s[pos] = b
        seqs.append("".join(s))
        mism[k] = count_mismatches("".join(draw[:half]), "".join(draw[half:]))
    return seqs, mism


def find_duplicates(sequences):
    """Find repeated sequences, ignoring case.

    Args:
        sequences (list[str]): Sequences to check.

    Returns:
        list[int]: Indices of sequences that repeat an earlier one (all but
        the first occurrence).
    """
    seen, dup = set(), []
    for i, s in enumerate(sequences):
        key = s.upper()
        if key in seen:
            dup.append(i)
        seen.add(key)
    return dup
