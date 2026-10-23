"""Force-controlled pulling: free energy at zero force.

Counterpart of :mod:`position_ensemble` for force control. Units: pN, nm, kT.

The work along a force protocol is ``W = -int x df / kT`` (signed along the
path); it estimates the Gibbs free-energy difference between the unfolded state
at ``f_max`` and the folded state at ``f_min``. Measured extensions are
relative to the folded branch at ``f_min`` (``x0``), so ``W_abs = W_meas -
x_F(f_min) (f_max - f_min) / kT`` is restored with the model before the
molecular contributions are subtracted:

    DG0 = DG - G_ss(f_max) - n_h [G_h(f_max) - G_h(f_min)]
          + G_diameter(f_min)

with ``G(f) = -int_0^f x df' / kT`` the (negative) Gibbs energy of each
element.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import KT25
from ..io import read_curve
from ..polymer import fjc_extension, fjc_gibbs, wlc_extension, wlc_gibbs
from .position_ensemble import TRAJECTORY, estimate_free_energy


@dataclass(frozen=True)
class ForceModel:
    """Elastic parameters of a force-controlled construct.

    The defaults describe the CD4 DNA hairpin (20 bp plus a 4-nt loop, i.e. 44
    nt) with two 29-bp handles; always set them for your molecule.

    Attributes:
        n_bp (int): Base pairs of the stem; each releases two nucleotides.
        loop_nt (int): Nucleotides of the loop.
        ss_p (float): Single-strand persistence length (nm).
        ss_d0 (float): Contour length per nucleotide (nm).
        handle_length (float): Contour length of one handle (nm).
        handle_p (float): Handle persistence length (nm).
        n_handles (int): Number of handles; their energies add.
        hairpin_d (float): Hairpin diameter (nm).
        kT (float): Thermal energy (pN nm).
    """

    n_bp: int = 20  # base pairs of the stem; each releases 2 nucleotides
    loop_nt: int = (
        4  # loop nucleotides (20 bp + tetraloop = 44 nt: CD4 hairpin)
    )
    ss_p: float = 1.35
    ss_d0: float = 0.59
    handle_length: float = (
        0.34 * 29
    )  # contour length of ONE handle (nm); two handles by default
    handle_p: float = 1.5
    n_handles: int = 2
    hairpin_d: float = 2.0
    kT: float = KT25

    @classmethod
    def from_sequence(cls, sequence, loop_nt=4, **kw):
        """Build the model from a hairpin sequence.

        The stem has ``(len(sequence) - loop_nt) / 2`` pairs. The released
        strand is ``2 n_bp + loop_nt`` nucleotides, so ``loop_nt=4`` describes
        a 44-nt hairpin of 20 bp plus a tetraloop.

        Args:
            sequence (str): Full hairpin sequence (5' arm + loop + 3' arm).
            loop_nt (int): Nucleotides of the loop.
            **kw: Other model fields.

        Returns:
            ForceModel: The model.

        Raises:
            ValueError: If the length is not ``2 * stem + loop_nt``.
        """
        n, rest = divmod(len(sequence) - loop_nt, 2)
        if rest or n < 1:
            raise ValueError("sequence length must be 2 * stem + loop_nt")
        return cls(n_bp=n, loop_nt=loop_nt, **kw)

    @property
    def ss_nt(self):
        """Return the released nucleotides: two per base pair plus the loop.

        Returns:
            int: Number of nucleotides.
        """
        return 2 * self.n_bp + self.loop_nt

    @property
    def ss_contour(self):
        """Return the contour length of the released strand.

        Returns:
            float: Contour length in nm.
        """
        return self.ss_nt * self.ss_d0


def folded_extension(f, m: ForceModel):
    """Return the absolute extension of the folded construct.

    Args:
        f (array-like): Force (pN).
        m (ForceModel): Elastic parameters.

    Returns:
        numpy.ndarray: Extension in nm (handles plus hairpin diameter).
    """
    return m.n_handles * m.handle_length * wlc_extension(
        f, m.handle_p, m.kT
    ) + fjc_extension(f, m.hairpin_d, m.hairpin_d, m.kT)


def gibbs_terms(f_min, f_max, m: ForceModel | None = None):
    """Return the Gibbs energies entering ``DG0``.

    Args:
        f_min (float): Force of the folded state (pN).
        f_max (float): Force of the unfolded state (pN).
        m (ForceModel | None): Elastic parameters; defaults if None.

    Returns:
        tuple[float, float, float]: Single strand at ``f_max``, handles
        between ``f_min`` and ``f_max``, and hairpin diameter at ``f_min``
        (kT).
    """
    m = m or ForceModel()
    g_ss = float(wlc_gibbs(f_max, m.ss_contour, m.ss_p, m.kT))
    g_h = m.n_handles * float(
        wlc_gibbs(f_max, m.handle_length, m.handle_p, m.kT)
        - wlc_gibbs(f_min, m.handle_length, m.handle_p, m.kT)
    )
    g_d = float(fjc_gibbs(f_min, m.hairpin_d, m.hairpin_d, m.kT))
    return g_ss, g_h, g_d


def _work(x, f, kind, sort_by_force, kT):
    """Return the force-ensemble work of one trajectory.

    Args:
        x (numpy.ndarray): Extension (nm).
        f (numpy.ndarray): Force (pN).
        kind (str): ``"u"`` (forward) or ``"f"`` (reverse).
        sort_by_force (bool): Integrate on the force-sorted curve.
        kT (float): Thermal energy (pN nm).

    Returns:
        float: Work in kT.
    """
    if sort_by_force:  # integrate on the force-sorted curve
        order = np.argsort(f, kind="stable")
        w = -np.trapezoid(x[order], f[order]) / kT
        return -w if kind == "f" else w  # a refolding path descends in force
    return -np.trapezoid(x, f) / kT


def works_from_directory(
    directory,
    f_range,
    kT=KT25,
    sort_by_force=True,
    f_col=0,
    x_col=1,
    names=None,
):
    """Return the force-ensemble works of the trajectories in a folder.

    Files ``u<N>.txt`` (forward) and ``f<N>.txt`` (reverse) with columns force
    and extension are read. Extensions are referenced to the mean folded-branch
    position at ``f_range[0]`` and curves are clipped to ``f_range``. Pass the
    works to :func:`zero_force_free_energy`, which applies the reference
    correction.

    Args:
        directory (str | os.PathLike): Folder with the trajectories.
        f_range (tuple[float, float]): Force range ``(f_min, f_max)`` (pN).
        kT (float): Thermal energy (pN nm).
        sort_by_force (bool): Integrate on the force-sorted curve instead of
            along the time order.
        f_col (int): Zero-based force column.
        x_col (int): Zero-based extension column.
        names (list[str] | None): Trajectory names to read; all files matching
            ``[uf]<N>.txt`` if None.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Forward and reverse works (kT)
        relative to ``x0``.
    """
    f0, f1 = f_range
    curves = {"u": [], "f": []}
    paths = (
        [Path(directory) / f"{n}.txt" for n in names]
        if names is not None
        else sorted(
            p
            for p in Path(directory).glob("*.txt")
            if TRAJECTORY.match(p.name)
        )
    )
    for path in paths:
        if path.name[0] not in curves:
            continue
        x, f = read_curve(path, x_col=x_col, f_col=f_col)
        if np.isnan(x).any() or np.isnan(f).any():
            continue
        curves[path.name[0]].append((x, f))
    x0 = []
    for x, f in curves["u"] + curves["f"]:
        o = np.argsort(f, kind="stable")
        if f[o][0] <= f0 <= f[o][-1]:
            x0.append(np.interp(f0, f[o], x[o]))
    x0 = float(np.mean(x0))
    out = {}
    for kind, items in curves.items():
        ws = []
        for x, f in items:
            keep = (f >= f0) & (f <= f1)
            if keep.sum() >= 2:
                ws.append(
                    _work(x[keep] - x0, f[keep], kind, sort_by_force, kT)
                )
        out[kind] = np.array(ws)
    return out["u"], out["f"]


def zero_force_free_energy(
    w_f, w_r, f_min, f_max, model: ForceModel | None = None, method="bennett"
):
    """Return the zero-force free energy of a force-controlled run.

    Args:
        w_f (array-like): Forward works (kT) from :func:`works_from_directory`.
        w_r (array-like): Reverse works (kT).
        f_min (float): Force of the folded state (pN).
        f_max (float): Force of the unfolded state (pN).
        model (ForceModel | None): Elastic parameters; defaults if None.
        method (str): Estimator, ``"bennett"`` or ``"mean"``.

    Returns:
        tuple[float, float]: ``DG0`` and its uncertainty (kT).
    """
    m = model or ForceModel()
    shift = float(folded_extension(f_min, m)) * (f_max - f_min) / m.kT
    fe = estimate_free_energy(np.asarray(w_f) - shift, np.asarray(w_r) + shift)
    g_ss, g_h, g_d = gibbs_terms(f_min, f_max, m)
    dg0 = fe.estimate(method) - g_ss - g_h + g_d
    err = float(
        np.hypot(fe.bennett_err, 0.5 * abs(fe.jarzynski_f - fe.jarzynski_r))
    )
    return dg0, err
