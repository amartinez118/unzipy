"""Nearest-neighbour DNA stacking free energies at 1 M NaCl and beyond.

Two parameter sets are provided: SantaLucia, Proc. Natl. Acad. Sci. USA 95,
1460 (1998), and the unzipping-derived set of Huguet et al., Proc. Natl. Acad.
Sci. USA 107, 15431 (2010). The free energy of a stack is ``dG = dH - T dS - m
ln[Na+]``. The SantaLucia set reproduces the per-stack energies printed by
MFOLD at 25 C to within about 0.01 kcal/mol for perfect Watson-Crick stems.
Hairpin-loop energies depend on loop size, sequence and terminal mismatch and
are not derived here: pass them explicitly.
"""

from __future__ import annotations

import numpy as np

# Unique nearest neighbours (AA=TT, AC=GT, AG=CT, AT, CA=TG, CC=GG, CG,
# GA=TC, GC, TA) as (dH kcal/mol, dS cal/mol/K, m kcal/mol per ln[Na+]);
# dG = dH - T dS - m ln[Na+].
PARAMETER_SETS = {
    # SantaLucia, PNAS 95, 1460 (1998); salt slope 0.114 per stack
    "santalucia1998": {
        "AA": (-7.9, -22.2, 0.114),
        "AC": (-8.4, -22.4, 0.114),
        "AG": (-7.8, -21.0, 0.114),
        "AT": (-7.2, -20.4, 0.114),
        "CA": (-8.5, -22.7, 0.114),
        "CC": (-8.0, -19.9, 0.114),
        "CG": (-10.6, -27.2, 0.114),
        "GA": (-8.2, -22.2, 0.114),
        "GC": (-9.8, -24.4, 0.114),
        "TA": (-7.2, -21.3, 0.114),
    },
    # unzipping-derived set: Huguet et al., PNAS 107, 15431 (2010)
    "huguet2010": {
        "AA": (-7.28, -20.28, 0.145),
        "AC": (-5.8, -14.46, 0.099),
        "AG": (-5.21, -12.89, 0.07),
        "AT": (-4.63, -11.62, 0.117),
        "CA": (-8.96, -24.48, 0.091),
        "CC": (-8.57, -22.3, 0.063),
        "CG": (-9.66, -24.43, 0.132),
        "GA": (-8.16, -22.46, 0.155),
        "GC": (-10.1, -25.96, 0.079),
        "TA": (-8.31, -25.06, 0.091),
    },
}
_COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}


def _canonical(dimer):
    """Map a dimer to its unique nearest-neighbour key.

    Args:
        dimer (str): Two nucleotides, 5' to 3'.

    Returns:
        str: The dimer itself, or its reverse complement, whichever is a key
        of the parameter tables.
    """
    rc = _COMP[dimer[1]] + _COMP[dimer[0]]
    return dimer if dimer in PARAMETER_SETS["santalucia1998"] else rc


R_KCAL = 1.987204e-3  # kcal / (mol K)


def kcal_to_kT(temperature_c=25.0):
    """Return the conversion factor from kcal/mol to kT.

    Args:
        temperature_c (float): Temperature in degrees Celsius.

    Returns:
        float: ``1 / (R T)`` in mol/kcal.
    """
    return 1.0 / (R_KCAL * (temperature_c + 273.15))


def stack_energies(stem, temperature_c=25.0, na=1.0, params="santalucia1998"):
    """Return the stacking free energies along a Watson-Crick stem.

    A stem of ``n`` base pairs has ``n - 1`` stacks; entry ``i`` stacks pair
    ``i`` on pair ``i + 1``.

    Args:
        stem (str): The 5' strand of the stem (A, C, G, T).
        temperature_c (float): Temperature in degrees Celsius.
        na (float): Sodium concentration (mol/L).
        params (str | dict): Name of a set in ``PARAMETER_SETS`` or a table of
            ``(dH, dS, m)`` by unique nearest neighbour.

    Returns:
        numpy.ndarray: Free energies in kcal/mol.

    Raises:
        ValueError: If ``na`` is not positive or ``stem`` has other letters.
    """
    if na <= 0:
        raise ValueError("[Na+] must be positive")
    table = PARAMETER_SETS[params] if isinstance(params, str) else params
    stem = stem.upper()
    if set(stem) - set("ACGT"):
        raise ValueError("stem must contain only A, C, G, T")
    T = temperature_c + 273.15
    out = []
    for i in range(len(stem) - 1):
        dH, dS, m = table[_canonical(stem[i : i + 2])]
        out.append(dH - T * dS / 1000.0 - m * np.log(na))
    return np.array(out)


def hairpin_stem(seq, loop_nt):
    """Return the 5' stem of a perfect hairpin sequence.

    Args:
        seq (str): Full sequence (5' arm + loop + 3' arm).
        loop_nt (int): Nucleotides in the loop.

    Returns:
        str: The 5' arm.

    Raises:
        ValueError: If the length is not ``2 * stem + loop_nt`` or the 3' arm
            is not the reverse complement of the 5' arm.
    """
    seq = seq.upper()
    n, rest = divmod(len(seq) - loop_nt, 2)
    if rest or n < 2:
        raise ValueError("sequence length must equal 2 * stem + loop_nt")
    stem, tail = seq[:n], seq[n + loop_nt :]
    if "".join(_COMP[c] for c in stem[::-1]) != tail:
        raise ValueError(
            "the 3' arm is not the reverse complement of the 5' arm"
        )
    return stem
