"""Equilibrium force-distance curve of an unzipping experiment.

The molecule has ``n = 0..N`` open base pairs; the two released strands (``n``
nucleotides each, FJC with Kuhn length ``b`` or WLC with persistence length
``p``) are in series with the optical trap of stiffness ``k``. For a trap
position ``lam`` each state has a force ``f_n(lam)`` solving ``lam = f/k + 2
x_strand(f; n)`` and an energy

    E_n(lam) = f_n^2 / (2 k) + 2 U_strand(f_n; n) + sum_{j<=n} eps_j

in kT. The partition function ``Z = sum_n exp(-E_n)`` gives the mean force
``<f> = sum_n p_n f_n``, which equals ``dF/dlam`` with ``F = -ln Z`` and is the
measured curve (evaluated analytically, without numerical differentiation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from ..constants import kT as thermal_energy
from ..polymer import fjc_energy, fjc_extension, wlc_energy, wlc_extension


@dataclass(frozen=True)
class UnzippingSetup:
    """Parameters of an unzipping experiment.

    Attributes:
        pair_cost (tuple[float, ...]): Free energy (kT) needed to open each
            base pair.
        k_trap (float): Trap stiffness (pN/nm).
        d0 (float): Contour length per nucleotide (nm).
        strand (str): ``"fjc"`` or ``"wlc"`` model of the released strands.
        kuhn (float): Kuhn length for the FJC (nm).
        persistence (float): Persistence length for the WLC (nm).
        temperature (float): Temperature (Celsius).
    """

    pair_cost: tuple  # kT needed to open pair j (j = 1..N); len N
    k_trap: float = 0.077  # pN/nm
    d0: float = 0.59  # nm per nucleotide
    strand: str = "fjc"  # "fjc" or "wlc"
    kuhn: float = 1.10  # nm (fjc)
    persistence: float = 1.35  # nm (wlc)
    temperature: float = 25.0

    @property
    def n_pairs(self):
        """Return the number of base pairs.

        Returns:
            int: Length of ``pair_cost``.
        """
        return len(self.pair_cost)

    @property
    def kT(self):
        """Return the thermal energy.

        Returns:
            float: ``k_B T`` in pN nm.
        """
        return thermal_energy(self.temperature)

    def strand_extension(self, f, n):
        """Return the extension of one released strand.

        Args:
            f (array-like): Force (pN).
            n (array-like): Nucleotides of the strand.

        Returns:
            numpy.ndarray: Extension in nm.
        """
        L = n * self.d0
        if self.strand == "fjc":
            return L * fjc_extension(f, self.kuhn, 1.0, self.kT)
        return L * wlc_extension(f, self.persistence, self.kT)

    def strand_energy(self, f, n):
        """Return the elastic energy of one released strand.

        Args:
            f (array-like): Force (pN).
            n (array-like): Nucleotides of the strand.

        Returns:
            numpy.ndarray: Energy in kT (zero for ``n = 0``).
        """
        L = np.maximum(n * self.d0, 1e-12)
        if self.strand == "fjc":
            u = fjc_energy(f, self.kuhn, L, self.kT)
        else:
            u = wlc_energy(f, L, self.persistence, self.kT)
        return u * (n > 0)

    def forces(self, lam, hi=150.0):
        """Return the force of every state at given trap positions.

        The equation ``lam = f/k + 2 x_strand(f; n)`` is solved by vectorised
        bisection.

        Args:
            lam (array-like): Trap positions (nm).
            hi (float): Upper bound of the force search (pN).

        Returns:
            numpy.ndarray: Array ``f[n, i]`` in pN.
        """
        lam = np.asarray(lam, dtype=float)
        n = np.arange(self.n_pairs + 1).reshape(-1, 1)
        lo = np.zeros((n.size, lam.size))
        up = np.full_like(lo, hi)
        for _ in range(60):
            mid = 0.5 * (lo + up)
            x = mid / self.k_trap + 2 * self.strand_extension(mid, n)
            below = x < lam
            lo, up = np.where(below, mid, lo), np.where(below, up, mid)
        return 0.5 * (lo + up)

    def energies(self, f):
        """Return the energy of every state.

        Args:
            f (numpy.ndarray): Forces from :meth:`forces`.

        Returns:
            numpy.ndarray: Array ``E[n, i]`` in kT.
        """
        n = np.arange(self.n_pairs + 1).reshape(-1, 1)
        stack = np.concatenate([[0.0], np.cumsum(self.pair_cost)]).reshape(
            -1, 1
        )
        return (
            stack
            + f**2 / (2 * self.k_trap * self.kT)
            + 2 * self.strand_energy(f, n)
        )


@dataclass
class FDC:
    """Equilibrium force-distance curve.

    Attributes:
        lam (numpy.ndarray): Trap positions (nm).
        force (numpy.ndarray): Equilibrium mean force (pN).
        free_energy (numpy.ndarray): ``F(lam) = -ln Z`` (kT).
        mean_open (numpy.ndarray): Mean number of open base pairs.
        saddle_force (numpy.ndarray): Force of the most probable state (pN).
        probabilities (numpy.ndarray): State probabilities ``p[n, i]``.
    """

    lam: np.ndarray
    force: np.ndarray  # equilibrium mean force <f>
    free_energy: np.ndarray  # F(lam) = -ln Z  (kT)
    mean_open: np.ndarray  # <n>
    saddle_force: np.ndarray  # force of the most probable state
    probabilities: np.ndarray  # p[n, i]


def equilibrium_fdc(setup: UnzippingSetup, lam) -> FDC:
    """Compute the equilibrium force-distance curve.

    Args:
        setup (UnzippingSetup): Molecule and instrument.
        lam (array-like): Trap positions (nm).

    Returns:
        FDC: Mean force, free energy, mean opening, saddle-point force and
        state probabilities.
    """
    lam = np.asarray(lam, dtype=float)
    f = setup.forces(lam)
    E = setup.energies(f)
    logz = np.asarray(logsumexp(-E, axis=0))
    p = np.exp(-E - logz)
    n = np.arange(E.shape[0])[:, None]
    return FDC(
        lam,
        (p * f).sum(axis=0),
        -logz,
        (p * n).sum(axis=0),
        f[np.argmin(E, axis=0), np.arange(lam.size)],
        p,
    )
