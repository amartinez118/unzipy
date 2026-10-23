"""Position-controlled pulling: free energy at zero force.

``force_ensemble`` is the force-controlled counterpart. Units: pN, nm, kT.

Model (energies in kT) for a hairpin with ``n_bases`` stem base pairs pulled
between the folded state at force ``f_min`` and the unfolded state at force
``f_max``:

    DG0 = DG - G_ssDNA(f_max) - G_handles + G_diameter(f_min)

where ``G_ssDNA`` is the WLC energy of the released single strands (``2 n_bases
+ loop_nt`` nucleotides), ``G_diameter`` the FJC energy of the hairpin diameter
and ``G_handles`` the energy absorbed by the beads and handles between
``f_min`` and ``f_max``: either a linear spring of stiffness ``k_eff``
(``handles="spring"``, the effective-stiffness method, suited to short handles)
or an explicit handle WLC (``handles="wlc"``, for long handles).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import KT25
from ..io import read_curve
from ..polymer import fjc_energy, wlc_compliance, wlc_energy
from .estimators import bennett, bennett_error, jarzynski, jarzynski_reverse
from .work import work_position


@dataclass(frozen=True)
class PullingModel:
    """Elastic parameters of a position-controlled construct.

    Set them for your molecule.

    Attributes:
        n_bases (int): Base pairs of the stem (the released strand has ``2
            n_bases + loop_nt`` nucleotides).
        loop_nt (int): Nucleotides of the hairpin loop.
        ss_p (float): Single-strand persistence length (nm).
        ss_d0 (float): Contour length per nucleotide (nm).
        hairpin_d (float): Hairpin diameter (nm).
        handle_p (float): Handle persistence length (nm).
        handle_d0 (float): Contour length per handle unit (nm).
        handle_n (int): Units in one handle.
        n_handles (int): Number of handles; their energies add.
        kT (float): Thermal energy (pN nm).
    """

    n_bases: int = 20  # stem base pairs (released: 2 n_bases + loop_nt nt)
    loop_nt: int = 4  # nucleotides in the hairpin loop
    ss_p: float = 0.75  # ssDNA persistence length (nm)
    ss_d0: float = 0.665  # interphosphate distance (nm)
    hairpin_d: float = 2.0  # hairpin diameter (nm)
    handle_p: float = 20.0  # handle persistence length (nm)
    handle_d0: float = 0.27  # handle contour length per unit (nm)
    handle_n: int = 500  # units in ONE handle
    n_handles: int = 2  # handles in the construct (their energies add)
    kT: float = KT25

    @property
    def ss_contour(self):
        """Return the contour length of the released strand.

        Returns:
            float: Contour length in nm.
        """
        return (2 * self.n_bases + self.loop_nt) * self.ss_d0

    @classmethod
    def from_sequence(cls, sequence, loop_nt=4, **kw):
        """Build the model from a hairpin sequence.

        The stem length is ``(len(sequence) - loop_nt) / 2``.

        Args:
            sequence (str): Full hairpin sequence (5' arm + loop + 3' arm).
            loop_nt (int): Nucleotides of the loop.
            **kw: Other model fields.

        Returns:
            PullingModel: The model.

        Raises:
            ValueError: If the length is not ``2 * stem + loop_nt``.
        """
        n, rest = divmod(len(sequence) - loop_nt, 2)
        if rest or n < 1:
            raise ValueError("sequence length must be 2 * stem + loop_nt")
        return cls(n_bases=n, loop_nt=loop_nt, **kw)

    @classmethod
    def dna_short(cls, n_bases=20, **kw):
        """Return the ssDNA model with short (29 bp) handles.

        Uses ``P = 1.35 nm`` and 0.59 nm per nucleotide (Severino et al.,
        J. Stat. Mech. (2019) 124001).

        Args:
            n_bases (int): Stem length; the CD4 hairpin has 20 bp plus a 4-nt
                loop, i.e. 44 nt.
            **kw: Other model fields.

        Returns:
            PullingModel: The model.
        """
        return cls(n_bases=n_bases, ss_p=1.35, ss_d0=0.59, **kw)

    @classmethod
    def rna_long(cls, n_bases=20, **kw):
        """Return the ssRNA model with long handles.

        Uses ``P = 0.75 nm`` and 0.665 nm per nucleotide (Severino et al.,
        J. Stat. Mech. (2019) 124001).

        Args:
            n_bases (int): Stem length.
            **kw: Other model fields.

        Returns:
            PullingModel: The model.
        """
        return cls(n_bases=n_bases, ss_p=0.75, ss_d0=0.665, **kw)


TRAJECTORY = re.compile(r"^[uf]\d+\.txt$")


def branch_force(x, slope, intercept):
    """Return the force on a fitted branch line, ``slope * x + intercept``.

    Use it to build ``f_min = k_F x0 + a_F`` and ``f_max = k_U x1 + a_U`` from
    the folded and unfolded branch fits rather than from the noisy measured
    extremes.

    Args:
        x (float | array-like): Extension (nm).
        slope (float): Branch stiffness (pN/nm).
        intercept (float): Branch intercept (pN).

    Returns:
        float | numpy.ndarray: Force in pN.
    """
    return slope * x + intercept


def works_from_directory(
    directory, x_range, kT=KT25, names=None, interpolate_edges=True, **read_kw
):
    """Integrate the trajectories of a folder over an extension range.

    Only files named ``u<N>.txt`` (forward, unfolding) and ``f<N>.txt``
    (reverse, refolding) are read, or exactly ``names`` (for example the
    trajectories listed in an index file). Curves with NaN are skipped. Works
    are signed trapezoid integrals of ``f dx`` in kT, so refolding (decreasing
    ``x``) comes out negative. The returned forces are the *measured* mean
    extremes inside the range, which noise biases outward; for ``DG0`` prefer
    :func:`branch_force` endpoints.

    Args:
        directory (str | os.PathLike): Folder with the trajectories.
        x_range (tuple[float, float]): Integration range ``(x0, x1)`` (nm).
        kT (float): Thermal energy (pN nm).
        names (list[str] | None): Trajectory names to read.
        interpolate_edges (bool): Add the curve values interpolated at exactly
            ``x0`` and ``x1`` to the integration (needs the curve to span the
            range).
        **read_kw: Passed to :func:`unzipy.io.read_curve`.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, float, float]: Forward works,
        reverse works (kT), and the measured mean minimum and maximum force
        (pN).
    """
    x0, x1 = x_range
    w = {"u": [], "f": []}
    fmin, fmax = [], []
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
        kind = path.name[0]
        if kind not in w:
            continue
        x, f = read_curve(path, **read_kw)
        if np.isnan(x).any() or np.isnan(f).any():
            continue
        sel = (x >= x0) & (x <= x1)
        if sel.sum() < 2:
            continue
        xs, fs = x[sel], f[sel]
        if (
            interpolate_edges
        ):  # integrate exactly between x0 and x1 (as simpleWork.m)
            order = np.argsort(x, kind="stable")
            if x[order][0] <= x0 and x[order][-1] >= x1:
                f0, f1 = np.interp([x0, x1], x[order], f[order])
                pts = (
                    [(x0, f0), (x1, f1)]
                    if xs[0] <= xs[-1]
                    else [(x1, f1), (x0, f0)]
                )
                xs = np.concatenate([[pts[0][0]], xs, [pts[1][0]]])
                fs = np.concatenate([[pts[0][1]], fs, [pts[1][1]]])
        w[kind].append(work_position(xs, fs, kT))
        fmin.append(f[sel].min())
        fmax.append(f[sel].max())
    return (
        np.array(w["u"]),
        np.array(w["f"]),
        float(np.mean(fmin)),
        float(np.mean(fmax)),
    )


@dataclass(frozen=True)
class FreeEnergy:
    """Free-energy estimates from forward and reverse works.

    Attributes:
        jarzynski_f (float): Jarzynski estimate from the forward works (kT).
        jarzynski_r (float): Jarzynski estimate from the reverse works (kT).
        bennett (float): Bennett acceptance ratio estimate (kT).
        bennett_err (float): Standard error of the Bennett estimate (kT).
        n_f (int): Number of forward trajectories.
        n_r (int): Number of reverse trajectories.
        wdis_f (float): ``<W_F> - dG`` (kT).
        wdis_r (float): ``<W_R> + dG`` (kT).
    """

    jarzynski_f: float
    jarzynski_r: float
    bennett: float
    bennett_err: float
    n_f: int
    n_r: int
    wdis_f: float  # <W_F> - DG
    wdis_r: float  # <W_R> + DG

    @property
    def wdis(self):
        """Return the average dissipated work per cycle.

        ``(<W_F> + <W_R>) / 2`` in kT, i.e. ``2 W_dis = <W>_F - <W>_R`` with
        ``<W>_R`` the mean of the sign-flipped reverse work.

        Returns:
            float: Dissipated work in kT.
        """
        return 0.5 * (self.wdis_f + self.wdis_r)

    def estimate(self, method="bennett"):
        """Return the combined free-energy estimate.

        Args:
            method (str): ``"bennett"`` (minimum variance) or ``"mean"``, one
                half of the mean Jarzynski estimate plus the Bennett estimate.

        Returns:
            float: Free-energy difference (kT).

        Raises:
            ValueError: If the method is unknown.
        """
        if method == "bennett":
            return self.bennett
        if method == "mean":
            return 0.5 * (
                0.5 * (self.jarzynski_f + self.jarzynski_r) + self.bennett
            )
        raise ValueError(method)


def estimate_free_energy(w_f, w_r):
    """Return all estimators of ``dG`` from forward and reverse works.

    Args:
        w_f (array-like): Forward works (kT).
        w_r (array-like): Reverse works (kT), negative on average.

    Returns:
        FreeEnergy: Jarzynski, Bennett and dissipation estimates.
    """
    w_f, w_r = np.asarray(w_f, float), np.asarray(w_r, float)
    dg = bennett(w_f, w_r)
    return FreeEnergy(
        jarzynski_f=float(jarzynski(w_f)),
        jarzynski_r=float(jarzynski_reverse(w_r)),
        bennett=float(dg),
        bennett_err=float(bennett_error(w_f, w_r, dg)),
        n_f=w_f.size,
        n_r=w_r.size,
        wdis_f=float(w_f.mean() - dg),
        wdis_r=float(w_r.mean() + dg),
    )


def contributions(f_min, f_max, model=None, k_eff=None, handles="spring"):
    """Return the free-energy terms separating ``dG`` from ``dG0``.

    ``DG0 = DG - g_ss - g_handles + g_diameter``.

    Args:
        f_min (float): Force of the folded state (pN).
        f_max (float): Force of the unfolded state (pN).
        model (PullingModel | None): Elastic parameters; defaults if None.
        k_eff (float | None): Effective stiffness (pN/nm) of trap and handles,
            required for ``handles="spring"``.
        handles (str): ``"spring"`` (effective stiffness) or ``"wlc"``.

    Returns:
        tuple[float, float, float]: Single strand, handles and diameter terms
        (kT).

    Raises:
        ValueError: If ``k_eff`` is missing or ``handles`` is unknown.
    """
    m = model or PullingModel()
    g_ss = float(wlc_energy(f_max, m.ss_contour, m.ss_p, m.kT))
    g_d = float(fjc_energy(f_min, m.hairpin_d, m.hairpin_d, m.kT))
    if handles == "spring":
        if k_eff is None:
            raise ValueError("handles='spring' needs k_eff (pN/nm)")
        g_h = 0.5 * (f_max**2 - f_min**2) / (m.kT * k_eff)
    elif handles == "wlc":
        Lc = (
            m.handle_n * m.handle_d0
        )  # one handle; the construct has ``n_handles`` of them
        g_h = m.n_handles * float(
            wlc_energy(f_max, Lc, m.handle_p, m.kT)
            - wlc_energy(f_min, Lc, m.handle_p, m.kT)
        )
    else:
        raise ValueError(handles)
    return g_ss, g_h, g_d


def zero_force_free_energy(
    w_f,
    w_r,
    f_min,
    f_max,
    model=None,
    k_eff=None,
    handles="spring",
    method="bennett",
    df_max=0.0,
):
    """Return the zero-force free energy of a position-controlled run.

    The uncertainty combines the Bennett error, the spread between the two
    Jarzynski estimates and, through the single-strand stretching term, the
    force scatter ``df_max``.

    Args:
        w_f (array-like): Forward works (kT).
        w_r (array-like): Reverse works (kT).
        f_min (float): Force of the folded state (pN).
        f_max (float): Force of the unfolded state (pN).
        model (PullingModel | None): Elastic parameters; defaults if None.
        k_eff (float | None): Effective stiffness (pN/nm).
        handles (str): ``"spring"`` or ``"wlc"``; see :func:`contributions`.
        method (str): Estimator, ``"bennett"`` or ``"mean"``.
        df_max (float): Standard deviation of the final force across
            trajectories (pN).

    Returns:
        tuple[float, float]: ``DG0`` and its uncertainty (kT).
    """
    model = model or PullingModel()
    fe = estimate_free_energy(w_f, w_r)
    g_ss, g_h, g_d = contributions(f_min, f_max, model, k_eff, handles)
    dg0 = fe.estimate(method) - g_ss - g_h + g_d
    sens = (
        f_max
        * model.ss_contour
        * float(wlc_compliance(f_max, model.ss_p, model.kT))
        / model.kT
    )
    jar_spread = abs(fe.jarzynski_f - fe.jarzynski_r)
    err = float(
        np.sqrt(
            fe.bennett_err**2 + (0.5 * jar_spread) ** 2 + (sens * df_max) ** 2
        )
    )
    return dg0, err
