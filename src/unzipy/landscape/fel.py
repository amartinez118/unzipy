"""Free-energy landscape of a hairpin opened sequentially under force.

State ``n`` (0..N) has its first ``n`` base pairs open; its free energy at
force ``f`` is (in kT)

    G(n) = [sum of the stacks still closed (+ loop)]
           + G_ssDNA(f; 2n nt) + G_diameter(f)

with ``G_ssDNA`` the force-ensemble WLC energy of the released strands (the
loop nucleotides are added in the last state) and ``G_diameter`` the FJC energy
of the hairpin diameter (absent once fully open). The landscape is referred to
state 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ..constants import kT as thermal_energy
from ..polymer import fjc_gibbs, wlc_gibbs
from .nn import kcal_to_kT


@dataclass(frozen=True)
class HairpinParams:
    """Elastic parameters of a hairpin landscape.

    Attributes:
        temperature (float): Temperature in degrees Celsius.
        ss_p (float): Single-strand persistence length (nm).
        ss_d0 (float): Contour length per nucleotide (nm).
        hairpin_d (float): Hairpin diameter (nm).
        loop_nt (int): Nucleotides in the hairpin loop.
    """

    temperature: float = 25.0  # Celsius
    ss_p: float = 1.35  # ssDNA persistence length (nm)
    ss_d0: float = 0.59  # interphosphate distance (nm)
    hairpin_d: float = 2.0  # hairpin diameter (nm)
    loop_nt: int = 4


def sequential_landscape(
    stack_dg, loop_dg, force, params: HairpinParams | None = None
):
    """Return the sequential-opening free-energy landscape.

    Args:
        stack_dg (array-like): Stacking energies (kcal/mol, negative), one per
            stack of the stem.
        loop_dg (float): Hairpin-loop energy (kcal/mol); it stays with the last
            closed pair.
        force (float): Applied force (pN).
        params (HairpinParams | None): Elastic parameters; defaults if None.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Number of open pairs
        ``n = 0..N`` (``N = len(stack_dg) + 1``) and the free energy in kT
        relative to state 0.
    """
    p = params or HairpinParams()
    kT = thermal_energy(p.temperature)
    g = np.append(np.asarray(stack_dg, dtype=float), loop_dg) * kcal_to_kT(
        p.temperature
    )
    nb = g.size
    n = np.arange(nb + 1)
    stack = np.append(np.cumsum(g[::-1])[::-1], 0.0)  # state n keeps g[n:]
    contour = 2 * n * p.ss_d0
    contour[-1] += p.loop_nt * p.ss_d0
    stretch = np.array(
        [
            float(wlc_gibbs(force, L, p.ss_p, kT)) if L > 0 else 0.0
            for L in contour
        ]
    )
    diam = np.full(
        nb + 1, float(fjc_gibbs(force, p.hairpin_d, p.hairpin_d, kT))
    )
    diam[-1] = 0.0
    G = stack + stretch + diam
    return n, G - G[0]


def coexistence_force(
    dg0_kT, n_bases, params: HairpinParams | None = None, bracket=(0.1, 100.0)
):
    """Return the force at which folded and unfolded states coexist.

    At that force the folded and fully unfolded hairpin have equal free energy.

    Args:
        dg0_kT (float): Zero-force unfolding free energy (kT), positive.
        n_bases (int): Base pairs of the stem.
        params (HairpinParams | None): Elastic parameters; defaults if None.
        bracket (tuple[float, float]): Force interval searched (pN).

    Returns:
        float: Coexistence force in pN.

    Raises:
        ValueError: If no coexistence force exists in ``bracket``.
    """
    p = params or HairpinParams()
    kT = thermal_energy(p.temperature)
    contour = (2 * n_bases + p.loop_nt) * p.ss_d0

    def delta(f):
        return (
            dg0_kT
            + wlc_gibbs(f, contour, p.ss_p, kT)
            - fjc_gibbs(f, p.hairpin_d, p.hairpin_d, kT)
        )

    try:
        return brentq(delta, *bracket)
    except ValueError as err:
        raise ValueError(
            f"no coexistence force in {bracket} pN for dG0 = {dg0_kT:.1f} kT: "
            "the free-energy difference must be positive and not exceed "
            "what stretching can pay"
        ) from err


def landscape_from_sequence(
    sequence, force, loop_dg, params: HairpinParams | None = None, na=1.0
):
    """Return the landscape of a perfect hairpin given by its sequence.

    Args:
        sequence (str): Full sequence (5' arm + loop + 3' arm), any length.
        force (float): Applied force (pN).
        loop_dg (float): Hairpin-loop energy (kcal/mol).
        params (HairpinParams | None): Elastic parameters; ``loop_nt`` gives
            the loop length. Defaults if None.
        na (float): Sodium concentration (mol/L).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Open pairs and free energy (kT);
        see :func:`sequential_landscape`.
    """
    from .nn import hairpin_stem, stack_energies

    p = params or HairpinParams()
    stem = hairpin_stem(sequence, p.loop_nt)
    return sequential_landscape(
        stack_energies(stem, p.temperature, na), loop_dg, force, p
    )
