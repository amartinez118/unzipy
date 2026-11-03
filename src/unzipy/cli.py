"""Command-line entry point: ``unzipy <command> ...``.

The molecule is always explicit: give ``--sequence`` (full hairpin, 5' arm +
loop + 3' arm) or ``--n-bp``, with ``--loop-nt``, and the single-strand
elasticity through ``--preset`` or ``--ss-p`` and ``--ss-d0``. Nothing falls
back to a particular construct. Also available as ``python -m unzipy``.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__

PRESETS = {
    "dna": (1.35, 0.59),
    "rna": (0.75, 0.665),
}  # persistence length (nm), nm per nucleotide


def _add_molecule(s, force=False):
    """Add the molecule options to a sub-parser.

    Args:
        s (argparse.ArgumentParser): Sub-parser to extend.
        force (bool): Also add the handle options of the force ensemble.
    """
    g = s.add_argument_group("molecule")
    g.add_argument(
        "--sequence",
        help="full hairpin sequence; sets the stem length from its length",
    )
    g.add_argument(
        "--n-bp", type=int, help="stem base pairs (if no --sequence)"
    )
    g.add_argument(
        "--loop-nt",
        type=int,
        required=True,
        help="nucleotides in the hairpin loop",
    )
    g.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help="ssDNA (P=1.35, 0.59 nm/nt) or ssRNA (0.75, 0.665)",
    )
    g.add_argument(
        "--ss-p", type=float, help="single-strand persistence length (nm)"
    )
    g.add_argument(
        "--ss-d0",
        type=float,
        help="interphosphate distance (nm per nucleotide)",
    )
    if force:
        g.add_argument(
            "--handle-length",
            type=float,
            default=0.34 * 29,
            help="contour length of one handle (nm)",
        )
        g.add_argument(
            "--handle-p",
            type=float,
            default=1.5,
            help="handle persistence length (nm)",
        )
        g.add_argument("--n-handles", type=int, default=2)


def _molecule(a):
    """Resolve the molecule options.

    Args:
        a (argparse.Namespace): Parsed arguments.

    Returns:
        tuple[int, float, float]: Stem base pairs, single-strand persistence
        length (nm) and nucleotide spacing (nm).

    Raises:
        ValueError: If the specification is incomplete or inconsistent.
    """
    if bool(a.sequence) == (a.n_bp is not None):
        raise ValueError("give exactly one of --sequence and --n-bp")
    if a.sequence:
        n_bp, rest = divmod(len(a.sequence) - a.loop_nt, 2)
        if rest or n_bp < 1:
            raise ValueError("sequence length must be 2 * stem + loop_nt")
    else:
        n_bp = a.n_bp
    p, d0 = PRESETS[a.preset] if a.preset else (None, None)
    p = a.ss_p if a.ss_p is not None else p
    d0 = a.ss_d0 if a.ss_d0 is not None else d0
    if p is None or d0 is None:
        raise ValueError("give --preset or both --ss-p and --ss-d0")
    return n_bp, p, d0


def _neqwork_position(a):
    """Run the position-controlled free-energy analysis.

    Args:
        a (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Results to print as JSON.

    Raises:
        ValueError: If ``--k-eff`` is given without ``--f-min`` and
            ``--f-max``, or the molecule parameters are incomplete.
    """
    from .fluctuation import estimate_free_energy
    from .fluctuation.position_ensemble import (
        PullingModel,
        works_from_directory,
        zero_force_free_energy,
    )

    if a.k_eff is not None and (a.f_min is None or a.f_max is None):
        raise ValueError(
            "--k-eff needs --f-min and --f-max (forces from the branch fits)"
        )
    n_bp, p, d0 = _molecule(a)
    w_f, w_r, f_min_meas, f_max_meas = works_from_directory(
        a.directory, a.x_range
    )
    fe = estimate_free_energy(w_f, w_r)
    out = {
        "n_forward": len(w_f),
        "n_reverse": len(w_r),
        "dG_bennett": fe.bennett,
        "dG_bennett_err": fe.bennett_err,
        "wdis": fe.wdis,
        "jarzynski_f": fe.jarzynski_f,
        # informational: noise biases the measured extremes outward
        "measured_f_min": f_min_meas,
        "measured_f_max": f_max_meas,
    }
    if a.k_eff is not None:
        model = PullingModel(n_bases=n_bp, loop_nt=a.loop_nt, ss_p=p, ss_d0=d0)
        dg0, err = zero_force_free_energy(
            w_f, w_r, a.f_min, a.f_max, model, k_eff=a.k_eff
        )
        out.update(f_min=a.f_min, f_max=a.f_max, dG0=dg0, dG0_err=err)
    return out


def _neqwork_force(a):
    """Run the force-controlled free-energy analysis.

    Args:
        a (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Results to print as JSON.
    """
    from .fluctuation import force_ensemble as fe_mod

    n_bp, p, d0 = _molecule(a)
    model = fe_mod.ForceModel(
        n_bp=n_bp,
        loop_nt=a.loop_nt,
        ss_p=p,
        ss_d0=d0,
        handle_length=a.handle_length,
        handle_p=a.handle_p,
        n_handles=a.n_handles,
    )
    w_f, w_r = fe_mod.works_from_directory(a.directory, a.f_range)
    f_min, f_max = a.f_range
    dg0, err = fe_mod.zero_force_free_energy(w_f, w_r, f_min, f_max, model)
    return {
        "n_forward": len(w_f),
        "n_reverse": len(w_r),
        "released_nt": model.ss_nt,
        "dG0": dg0,
        "dG0_err": err,
    }


def _split(a):
    """Split a cyclic trace into trajectories.

    Args:
        a (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Number of segments written.
    """
    from .unzipping import read_tweezers_trace, split_cycles, write_segments

    t, x, f = read_tweezers_trace(a.file, t_range=a.t_range)
    segments = split_cycles(t, f, a.f_min, a.f_max)
    names = write_segments(t, x, f, segments, a.output)
    return {
        "segments": len(names),
        "unfolding": sum(n[0] == "u" for n in names),
    }


def _simulate(a):
    """Simulate pulling trajectories of a hairpin and write them.

    Args:
        a (argparse.Namespace): Parsed arguments.

    Returns:
        dict: Summary of the simulation.
    """
    from .simulation import HairpinSim, write_trajectories

    sim = HairpinSim.from_sequence(
        a.sequence, a.loop_nt, a.loop_dg, a.temperature
    )
    for direction, prefix in (("forward", "u"), ("reverse", "f")):
        write_trajectories(
            sim.simulate(a.n, direction, rng=a.seed), a.output, prefix
        )
    return {"output": a.output, "n_per_direction": a.n, "base_pairs": sim.n_bp}


def build_parser():
    """Build the command-line parser.

    Returns:
        argparse.ArgumentParser: Parser with one sub-command per workflow.
    """
    p = argparse.ArgumentParser(prog="unzipy", description=__doc__)
    p.add_argument(
        "--version", action="version", version=f"unzipy {__version__}"
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "neqwork-position",
        help="free energy from position-controlled pulling curves",
    )
    s.add_argument(
        "directory", help="folder with u<N>.txt / f<N>.txt trajectories"
    )
    s.add_argument(
        "--x-range", nargs=2, type=float, required=True, metavar=("X0", "X1")
    )
    s.add_argument(
        "--k-eff",
        type=float,
        help="effective stiffness (pN/nm): also report dG0",
    )
    s.add_argument(
        "--f-min",
        type=float,
        help="force at x0 from the folded-branch fit (needed with --k-eff)",
    )
    s.add_argument(
        "--f-max",
        type=float,
        help="force at x1 from the unfolded-branch fit (needed with --k-eff)",
    )
    _add_molecule(s)
    s.set_defaults(func=_neqwork_position)

    s = sub.add_parser(
        "neqwork-force",
        help="free energy from force-controlled pulling curves",
    )
    s.add_argument("directory")
    s.add_argument(
        "--f-range", nargs=2, type=float, required=True, metavar=("F0", "F1")
    )
    _add_molecule(s, force=True)
    s.set_defaults(func=_neqwork_force)

    s = sub.add_parser(
        "split", help="split a cyclic tweezers trace into u*/f* trajectories"
    )
    s.add_argument("file")
    s.add_argument("--f-min", type=float, required=True)
    s.add_argument("--f-max", type=float, required=True)
    s.add_argument("--t-range", nargs=2, type=float)
    s.add_argument("-o", "--output", default="events_raw")
    s.set_defaults(func=_split)

    s = sub.add_parser(
        "simulate", help="simulate pulling of a perfect hairpin"
    )
    s.add_argument(
        "sequence", help="hairpin sequence (5' arm + loop + 3' arm)"
    )
    s.add_argument("--loop-nt", type=int, required=True)
    s.add_argument(
        "--loop-dg",
        type=float,
        required=True,
        help="hairpin-loop free energy (kcal/mol)",
    )
    s.add_argument("--temperature", type=float, default=25.0)
    s.add_argument("-n", type=int, default=100)
    s.add_argument("--seed", type=int)
    s.add_argument("-o", "--output", default="simulation")
    s.set_defaults(func=_simulate)
    return p


def main(argv=None):
    """Run the command-line interface.

    Args:
        argv (list[str] | None): Arguments; ``sys.argv[1:]`` if None.

    Returns:
        int: Exit status (0 on success, 2 on invalid input).
    """
    args = build_parser().parse_args(argv)
    try:
        result = args.func(args)
    except ValueError as err:  # bad user input: message, not a traceback
        print(f"unzipy: error: {err}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
