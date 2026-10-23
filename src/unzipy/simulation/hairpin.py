"""Stochastic pulling of a DNA/RNA hairpin with optical tweezers.

The trap position ``lam`` is the controlled variable. For every ``lam`` the
energies of all intermediate states ``n = 0..N`` (``n`` open base pairs) are
computed once, so that the folded/unfolded switching probability of a time step
is

    p = k0 exp(-B) [(1 - sigma) + sigma exp(G_N)] dt,
    B = ln sum_{i>=k} exp(G_i - G_k)

(``sigma = 0`` folded, 1 unfolded; ``B`` is a Kramers-like barrier over the
full landscape). This obeys detailed balance: ``k_fold->unf / k_unf->fold =
exp(-G_N)``. Everything that depends only on ``lam`` is vectorised on a grid
and trajectories are simulated in parallel. Units: pN, nm, s, kT.

The finite stretch modulus of the diameter FJC (about 8e6 pN, a ~1e-6 effect)
is neglected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ..constants import kT as thermal_energy
from ..landscape.nn import kcal_to_kT
from ..polymer import fjc_energy, fjc_extension, wlc_energy, wlc_extension


@dataclass(frozen=True)
class HairpinSim:
    """Molecule and instrument of a simulated pulling experiment.

    State ``n`` keeps the energies ``gbp[n:]``; the number of base pairs is
    ``len(gbp)``. ``nt_open`` overrides the nucleotides released in each state,
    needed for structures with interior loops (see
    :func:`unzipy.io.read_mfold_input`).

    Attributes:
        gbp (tuple[float, ...]): Stack energies (kcal/mol) followed by the loop
            energy.
        nloop (int): Nucleotides in the loop.
        temperature (float): Temperature in degrees Celsius.
        handle_p (float): Handle persistence length (nm).
        handle_length (float): Contour length of the handles (nm).
        hairpin_d (float): Hairpin diameter (nm).
        k_trap (float): Trap stiffness (pN/nm).
        ss_p (float): Single-strand persistence length (nm).
        ss_d0 (float): Contour length per nucleotide (nm).
        k0 (float): Attempt rate (1/s).
        dt (float): Time step (s).
        n_meas (int): Measurements per trap position.
        dl (float): Trap step (nm).
        var_l (float): Variance of the extension noise (nm^2).
        f_min (float): Force below which a reverse run ends (pN).
        f_max (float): Force above which a forward run ends (pN).
        nt_open (tuple[int, ...] | None): Nucleotides released in each state.
    """

    gbp: tuple
    nloop: int = 4
    temperature: float = 25.0
    handle_p: float = 1.5  # nm
    handle_length: float = 29 * 2 * 0.34  # nm
    hairpin_d: float = 2.0  # nm
    k_trap: float = 0.068  # pN/nm
    ss_p: float = 1.35  # nm
    ss_d0: float = 0.59  # nm per nucleotide
    k0: float = 1e5  # attempt rate (1/s)
    dt: float = 1e-4  # time step (s)
    n_meas: int = 10  # measurements per trap position
    dl: float = 0.5  # trap step (nm)
    var_l: float = 0.02  # variance of the extension noise (nm^2)
    f_min: float = 3.0
    f_max: float = 25.0
    nt_open: tuple | None = (
        None  # released nucleotides per state (default 2n, + loop at n = N)
    )

    def __post_init__(self):
        """Validate the parameters.

        Raises:
            ValueError: If ``gbp`` is empty, ``f_min >= f_max`` or ``nt_open``
                has the wrong length.
        """
        if len(self.gbp) < 1:
            raise ValueError(
                "gbp needs at least one entry "
                "(stacks followed by the loop energy)"
            )
        if not self.f_min < self.f_max:
            raise ValueError("f_min must be smaller than f_max")
        if self.nt_open is not None and len(self.nt_open) != len(self.gbp) + 1:
            raise ValueError("nt_open needs len(gbp) + 1 entries")

    @classmethod
    def from_sequence(
        cls, sequence, loop_nt, loop_dg, temperature=25.0, na=1.0, **kw
    ):
        """Build a simulation of a perfect hairpin from its sequence.

        Stacks come from the SantaLucia parameters; the hairpin-loop energy is
        explicit because it depends on the loop.

        Args:
            sequence (str): Full sequence (5' arm + loop + 3' arm), any length.
            loop_nt (int): Nucleotides in the loop.
            loop_dg (float): Hairpin-loop free energy (kcal/mol).
            temperature (float): Temperature in degrees Celsius.
            na (float): Sodium concentration (mol/L).
            **kw: Other :class:`HairpinSim` fields.

        Returns:
            HairpinSim: The simulation.
        """
        from ..landscape.nn import hairpin_stem, stack_energies

        stem = hairpin_stem(sequence, loop_nt)
        gbp = tuple(np.append(stack_energies(stem, temperature, na), loop_dg))
        return cls(gbp, nloop=loop_nt, temperature=temperature, **kw)

    @property
    def n_bp(self):
        """Return the number of base pairs.

        Returns:
            int: Length of ``gbp``.
        """
        return len(self.gbp)

    @property
    def kT(self):
        """Return the thermal energy.

        Returns:
            float: ``k_B T`` in pN nm.
        """
        return thermal_energy(self.temperature)

    # --- elastic model ---------------------------------------------------
    def _contour(self, n):
        """Return the contour length of the released strand.

        Args:
            n (array-like): State (number of open pairs).

        Returns:
            numpy.ndarray: Contour length in nm.
        """
        n = np.asarray(n)
        if (
            self.nt_open is not None
        ):  # MFOLD-derived structures (interior loops, bulges)
            return np.asarray(self.nt_open, dtype=float)[n] * self.ss_d0
        return 2 * n * self.ss_d0 + np.where(
            n == self.n_bp, self.nloop * self.ss_d0, 0.0
        )

    def extension(self, f, n):
        """Return the trap position at a given force and state.

        Args:
            f (array-like): Force (pN).
            n (array-like): Number of open base pairs.

        Returns:
            numpy.ndarray: Trap position in nm.
        """
        f = np.asarray(f, dtype=float)
        n = np.asarray(n)
        x = (
            self.handle_length * wlc_extension(f, self.handle_p, self.kT)
            + f / self.k_trap
        )
        x = x + np.where(
            n < self.n_bp,
            fjc_extension(f, self.hairpin_d, self.hairpin_d, self.kT),
            0.0,
        )
        return x + self._contour(n) * wlc_extension(f, self.ss_p, self.kT)

    def force(self, lam, n, hi=100.0):
        """Return the force at a given trap position and state.

        Solved by vectorised bisection.

        Args:
            lam (array-like): Trap position (nm).
            n (array-like): Number of open base pairs.
            hi (float): Upper bound of the force search (pN).

        Returns:
            numpy.ndarray: Force in pN.
        """
        lam, n = np.broadcast_arrays(
            np.asarray(lam, dtype=float), np.asarray(n)
        )
        lo, up = np.zeros(lam.shape), np.full(lam.shape, hi)
        for _ in range(60):
            mid = 0.5 * (lo + up)
            below = self.extension(mid, n) < lam
            lo, up = np.where(below, mid, lo), np.where(below, up, mid)
        return 0.5 * (lo + up)

    def stiffness(self, f, sigma):
        """Return the molecular stiffness of the folded or unfolded branch.

        The trap compliance is in series with the molecule and is removed.

        Args:
            f (array-like): Force (pN).
            sigma (array-like): 0 for the folded and 1 for the unfolded branch.

        Returns:
            numpy.ndarray: Stiffness ``km`` in pN/nm.
        """
        f = np.asarray(f, dtype=float)
        n = np.where(np.asarray(sigma) == 0, 0, self.n_bp)
        h = 1e-4 * np.maximum(f, 1e-2)
        dxdf = (self.extension(f + h, n) - self.extension(f - h, n)) / (2 * h)
        return 1.0 / (
            dxdf - 1.0 / self.k_trap
        )  # trap compliance is in series with the molecule

    def landscape(self, lam):
        """Return the free energy of every state at given trap positions.

        Args:
            lam (array-like): Trap positions (nm).

        Returns:
            numpy.ndarray: Array ``G[n, ...]`` in kT, relative to ``n = 0``.
        """
        lam = np.asarray(lam, dtype=float)
        kcal = kcal_to_kT(self.temperature)
        g = np.asarray(self.gbp, dtype=float) * kcal
        n = np.arange(self.n_bp + 1).reshape(-1, *([1] * lam.ndim))
        f = self.force(lam, n)
        stack = np.append(np.cumsum(g[::-1])[::-1], 0.0).reshape(n.shape)
        e = stack + 0.5 * f**2 / (self.k_trap * self.kT)
        e = e + wlc_energy(f, self.handle_length, self.handle_p, self.kT)
        e = e + wlc_energy(
            f, np.maximum(self._contour(n), 1e-12), self.ss_p, self.kT
        ) * (n > 0)
        e = e + np.where(
            n < self.n_bp,
            fjc_energy(f, self.hairpin_d, self.hairpin_d, self.kT),
            0.0,
        )
        return e - e[0]

    def barrier(self, G):
        """Return the Kramers-like barrier of the landscape.

        ``B = ln sum_i sum_{k<=i} exp(G_i - G_k)`` along the first axis.

        Args:
            G (numpy.ndarray): Free energies from :meth:`landscape`.

        Returns:
            numpy.ndarray: Barrier at each trap position.
        """
        inner = np.logaddexp.accumulate(-G, axis=0)  # ln sum_{k<=i} exp(-G_k)
        return logsumexp(G + inner, axis=0)

    # --- simulation ------------------------------------------------------
    def simulate(
        self, n_traj=100, direction="forward", lam_start=None, rng=None
    ):
        """Simulate independent pulling or relaxing trajectories.

        The trap moves by ``dl`` per block of ``n_meas`` measurements.

        Args:
            n_traj (int): Number of trajectories.
            direction (str): ``"forward"`` (pulling, starting folded) or
                ``"reverse"`` (relaxing, starting unfolded).
            lam_start (float | None): Initial trap position (nm); derived from
                the molecule if None.
            rng (int | numpy.random.Generator | None): Seed or generator.

        Returns:
            list[dict]: One dict per trajectory with arrays ``lam``, ``f`` and
            ``sigma`` and the float ``lam_start``.

        Raises:
            ValueError: If ``direction`` is invalid.
        """
        gen = np.random.default_rng(rng)
        fwd = direction == "forward"
        if not fwd and direction != "reverse":
            raise ValueError(direction)
        if lam_start is None:
            # start below / above the folded / unfolded extensions, whatever
            # the molecule size
            if fwd:
                lam_start = (
                    np.floor(
                        min(50.0, float(self.extension(self.f_min, 0)))
                        / self.dl
                    )
                    * self.dl
                )
            else:
                lam_start = (
                    np.ceil(
                        max(
                            400.0,
                            float(self.extension(self.f_max, self.n_bp))
                            + 20.0,
                        )
                        / self.dl
                    )
                    * self.dl
                )  # grid-aligned
        # trap positions visited: forward ends when the unfolded force
        # exceeds f_max, reverse when the folded force drops below f_min
        end = (
            self.extension(self.f_max, self.n_bp)
            if fwd
            else self.extension(self.f_min, 0)
        )
        steps = int(np.ceil(abs(end - lam_start) / self.dl)) + 1
        lam = lam_start + (1 if fwd else -1) * self.dl * np.arange(steps)
        G = self.landscape(lam)
        B = self.barrier(G)
        fF, fU = self.force(lam, 0), self.force(lam, self.n_bp)
        kmF, kmU = self.stiffness(fF, 0), self.stiffness(fU, 1)
        pref = self.k0 * np.exp(-B) * self.dt
        gN = G[-1]

        sigma = np.full(n_traj, 0 if fwd else 1)
        f_rec = np.empty((steps, self.n_meas, n_traj))
        l_rec = np.empty_like(f_rec)
        s_rec = np.empty(f_rec.shape, dtype=np.int8)
        alive = np.ones(n_traj, dtype=bool)
        last = np.full(n_traj, steps - 1)
        for k in range(steps):
            f = np.where(sigma == 0, fF[k], fU[k])
            km = np.where(sigma == 0, kmF[k], kmU[k])
            for m in range(self.n_meas):
                sd = np.sqrt(self.kT * self.k_trap**2 / (self.k_trap + km))
                f_rec[k, m] = f + gen.normal(0, sd)
                l_rec[k, m] = lam[k] + gen.normal(
                    0, np.sqrt(self.var_l), n_traj
                )
                s_rec[k, m] = sigma
                p = np.minimum(
                    pref[k] * np.where(sigma == 0, 1.0, np.exp(gN[k])), 1.0
                )
                flip = gen.uniform(size=n_traj) <= p
                sigma = np.where(flip, 1 - sigma, sigma)
                f = np.where(flip, np.where(sigma == 0, fF[k], fU[k]), f)
                km = np.where(flip, np.where(sigma == 0, kmF[k], kmU[k]), km)
            done = (
                (f > self.f_max) if fwd else ((f < self.f_min) & (sigma == 0))
            )
            newly = alive & done
            last[newly] = k
            alive &= ~done
            if not alive.any():
                break
        out = []
        for j in range(n_traj):
            n_blocks = (last[j] + 1) * self.n_meas
            sl = (slice(0, last[j] + 1), slice(None), j)
            out.append(
                {
                    "lam": l_rec[sl].reshape(n_blocks),
                    "f": f_rec[sl].reshape(n_blocks),
                    "sigma": s_rec[sl].reshape(n_blocks),
                    "lam_start": float(lam_start),
                }
            )
        return out


def write_trajectories(trajectories, directory, prefix="u"):
    """Write simulated trajectories to text files.

    Files ``<prefix>1.txt``, ``<prefix>2.txt``, ... have the columns ``lam f
    sigma``.

    Args:
        trajectories (list[dict]): Trajectories from
            :meth:`HairpinSim.simulate`.
        directory (str | os.PathLike): Output folder, created if needed.
        prefix (str): File name prefix (``u`` forward, ``f`` reverse).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(trajectories, 1):
        np.savetxt(
            directory / f"{prefix}{i}.txt",
            np.column_stack([t["lam"], t["f"], t["sigma"]]),
        )
