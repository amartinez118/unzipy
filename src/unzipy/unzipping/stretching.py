"""Elastic contributions of the trap setup beyond effective stiffness.

Implements the procedure of Severino, Monge, Rissone, Ritort, J. Stat. Mech.
(2019) 124001 (sec. 5). The trap distance of a state is the sum of the
extensions of the elements in series,

    lam(f) = x_h(f) + x_b(f) + [x_d(f) | x_ss(f)] + lam0,

with extensible-WLC handles (``Lc -> Lc (1 + f/Y)``), a bead in a trap of
force-dependent stiffness ``k_b(f) = k_b0 + alpha f`` (``x_b = ln(1 + alpha f /
k_b0) / alpha``), the folded hairpin as a dipole of diameter ``d`` (``x_d``)
and the unfolded hairpin as an inextensible WLC with ``n`` released nucleotides
(``x_ss``). Fitting both branches of the force-distance curves gives the
parameters, from which the reversible work of bead and handles between
``f_min`` and ``f_max``,

    DW_b + DW_h = int f [1/k_b(f) + 1/k_h(f)] df,

is obtained and subtracted from the free-energy difference of the fluctuation
theorem. Units: pN, nm, kT.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from ..constants import KT25
from ..polymer import (
    ewlc_scaled_compliance,
    ewlc_scaled_extension,
    fjc_energy,
    fjc_extension,
    wlc_energy,
    wlc_extension,
)

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True)
class StretchModel:
    """Elastic model of trap, handles and molecule.

    Attributes:
        handle_n (float): Units in the handles (both together).
        handle_db (float): Contour length per handle unit (nm).
        handle_p (float): Handle persistence length (nm).
        handle_y (float): Handle Young modulus (pN).
        k_b0 (float): Trap stiffness at zero force (pN/nm).
        alpha (float): Force dependence of the trap stiffness (1/nm).
        n_ss (float): Nucleotides released on unfolding.
        ss_p (float): Single-strand persistence length (nm).
        ss_d0 (float): Contour length per nucleotide (nm).
        hairpin_d (float): Hairpin diameter (nm).
        lam0 (float): Offset of the trap distance (nm).
        kT (float): Thermal energy (pN nm).
    """

    handle_n: float = 1000.0  # handle units (both handles together)
    handle_db: float = 0.27  # nm per unit
    handle_p: float = 20.0  # nm
    handle_y: float = 200.0  # Young modulus (pN)
    k_b0: float = 0.07  # trap stiffness at zero force (pN/nm)
    alpha: float = 0.0  # d k_b / d f
    n_ss: float = 52.0  # released nucleotides
    ss_p: float = 0.75
    ss_d0: float = 0.665
    hairpin_d: float = 2.0
    lam0: float = 0.0
    kT: float = KT25

    # --- elements ----------------------------------------------------------
    @property
    def handle_contour(self):
        """Return the total contour length of the handles.

        Returns:
            float: Contour length in nm.
        """
        return self.handle_n * self.handle_db

    def x_handles(self, f):
        """Return the total extension of the handles.

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Extension in nm (extensible WLC).
        """
        return self.handle_contour * ewlc_scaled_extension(
            f, self.handle_p, self.handle_y, self.kT
        )

    def inv_k_handles(self, f):
        """Return the compliance of the handles, ``dx_h/df``.

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Compliance in nm/pN.
        """
        return self.handle_contour * ewlc_scaled_compliance(
            f, self.handle_p, self.handle_y, self.kT
        )

    def x_bead(self, f):
        """Return the bead displacement in the trap.

        ``ln(1 + alpha f / k_b0) / alpha`` (``f / k_b0`` for ``alpha = 0``).

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Displacement in nm.
        """
        f = np.asarray(f, dtype=float)
        if abs(self.alpha) < 1e-12:
            return f / self.k_b0
        return np.log1p(self.alpha * f / self.k_b0) / self.alpha

    def inv_k_bead(self, f):
        """Return the compliance of the trap, ``1 / k_b(f)``.

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Compliance in nm/pN.
        """
        return 1.0 / (self.k_b0 + self.alpha * np.asarray(f, dtype=float))

    def lam_folded(self, f):
        """Return the trap distance of the folded branch.

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Trap distance in nm.
        """
        return (
            self.x_handles(f)
            + self.x_bead(f)
            + fjc_extension(f, self.hairpin_d, self.hairpin_d, self.kT)
            + self.lam0
        )

    def lam_unfolded(self, f):
        """Return the trap distance of the unfolded branch.

        Args:
            f (array-like): Force (pN).

        Returns:
            numpy.ndarray: Trap distance in nm.
        """
        ss = self.n_ss * self.ss_d0 * wlc_extension(f, self.ss_p, self.kT)
        return self.x_handles(f) + self.x_bead(f) + ss + self.lam0

    # --- energies (kT) -----------------------------------------------------
    def bead_handles_work(self, f_min, f_max):
        """Return the reversible work to stretch bead and handles.

        Args:
            f_min (float): Lower force (pN).
            f_max (float): Upper force (pN).

        Returns:
            tuple[float, float]: Work of the bead and of the handles (kT).
        """
        f = 0.5 * (f_max - f_min) * _NODES + 0.5 * (f_max + f_min)
        w = 0.5 * (f_max - f_min) * _WEIGHTS
        return (
            float(np.sum(w * f * self.inv_k_bead(f)) / self.kT),
            float(np.sum(w * f * self.inv_k_handles(f)) / self.kT),
        )

    def molecular_work(self, f_min, f_max):
        """Return the molecular stretching contribution ``DW_m``.

        The WLC energy of the unfolded strand at ``f_max`` minus the dipole
        energy of the folded hairpin at ``f_min``.

        Args:
            f_min (float): Force of the folded state (pN).
            f_max (float): Force of the unfolded state (pN).

        Returns:
            float: ``DW_m`` in kT.
        """
        return float(
            wlc_energy(f_max, self.n_ss * self.ss_d0, self.ss_p, self.kT)
            - fjc_energy(f_min, self.hairpin_d, self.hairpin_d, self.kT)
        )

    def zero_force_free_energy(self, dg, f_min, f_max):
        """Return ``DG0 = DG - DW_m - DW_b - DW_h``.

        Args:
            dg (float): Free-energy difference of the fluctuation theorem (kT).
            f_min (float): Force of the folded state (pN).
            f_max (float): Force of the unfolded state (pN).

        Returns:
            float: Zero-force free energy in kT.
        """
        w_b, w_h = self.bead_handles_work(f_min, f_max)
        return dg - self.molecular_work(f_min, f_max) - w_b - w_h


@dataclass
class StretchFit:
    """Result of fitting the stretching model.

    Attributes:
        model (StretchModel): Model with the fitted parameters.
        params (dict): Fitted parameters by name.
        errors (dict): Standard errors by name.
        rms (float): Root-mean-square residual (nm).
    """

    model: StretchModel
    params: dict
    errors: dict
    rms: float


def fit_stretching_model(
    f_folded,
    lam_folded,
    f_unfolded,
    lam_unfolded,
    start: StretchModel | None = None,
    free=("k_b0", "alpha", "dP", "lam0", "n_ss"),
):
    """Fit the trap-distance model to both branches.

    The free parameters are fitted by least squares (trust-region reflective).
    Everything else keeps its value in ``start``, as in the paper (e.g. ``Y``
    and ``db`` fixed).

    Args:
        f_folded (array-like): Force (pN) of the folded branch.
        lam_folded (array-like): Trap distance (nm) of the folded branch.
        f_unfolded (array-like): Force (pN) of the unfolded branch.
        lam_unfolded (array-like): Trap distance (nm) of the unfolded branch.
        start (StretchModel | None): Initial model; defaults if None.
        free (tuple[str, ...]): Names of the free parameters: fields of
            :class:`StretchModel` and ``"dP"``, the relative deviation of the
            handle persistence length from ``start.handle_p``.

    Returns:
        StretchFit: Fitted model, parameters, errors and residual.
    """
    m0 = start or StretchModel()
    ff, lf = np.asarray(f_folded, float), np.asarray(lam_folded, float)
    fu, lu = np.asarray(f_unfolded, float), np.asarray(lam_unfolded, float)
    free = tuple(free)

    def build(theta):
        kw = dict(zip(free, theta, strict=True))
        if "dP" in kw:
            kw["handle_p"] = m0.handle_p * (1.0 + kw.pop("dP"))
        return replace(m0, **kw)

    def resid(theta):
        m = build(theta)
        return np.concatenate([m.lam_folded(ff) - lf, m.lam_unfolded(fu) - lu])

    theta0 = np.array([0.0 if k == "dP" else getattr(m0, k) for k in free])
    lo = np.array(
        [{"k_b0": 1e-3, "n_ss": 1.0, "dP": -0.9}.get(k, -np.inf) for k in free]
    )
    res = least_squares(resid, theta0, bounds=(lo, np.inf), x_scale="jac")
    dof = max(res.fun.size - len(free), 1)
    cov = np.linalg.pinv(res.jac.T @ res.jac) * np.sum(res.fun**2) / dof
    return StretchFit(
        build(res.x),
        dict(zip(free, res.x, strict=True)),
        dict(zip(free, np.sqrt(np.diag(cov)), strict=True)),
        float(np.sqrt(np.mean(res.fun**2))),
    )
