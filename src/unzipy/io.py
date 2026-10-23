"""Reading experiment files and output of external programs."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_table(path, decimal=".", comments="#", **kwargs):
    """Read a numeric table, optionally with decimal commas.

    Args:
        path (str | os.PathLike): File to read.
        decimal (str): Decimal separator of the file. Commas are replaced by
            dots when ``decimal=","``.
        comments (str): Prefix of comment lines.
        **kwargs: Passed to :func:`numpy.loadtxt`.

    Returns:
        numpy.ndarray: The table.
    """
    text = Path(path).read_text()
    if decimal != ".":
        text = text.replace(decimal, ".")
    return np.loadtxt(text.splitlines(), comments=comments, **kwargs)


def read_curve(path, x_col=1, f_col=3):
    """Read one force-extension trajectory.

    The defaults match trajectory files with the columns ``index, x, t, f``.

    Args:
        path (str | os.PathLike): File to read; ``#`` and ``%`` start comments.
        x_col (int): Zero-based column of the extension.
        f_col (int): Zero-based column of the force.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Extension ``x`` and force ``f``.
    """
    data = np.loadtxt(
        path, comments=("#", "%"), usecols=(x_col, f_col), ndmin=2
    )
    return data[:, 0], data[:, 1]


def read_mfold_input(path):
    """Read an MFOLD-derived structure file.

    The file lists, for each molecule, the energy and the number of released
    nucleotides of every opening state. Blocks start with ``<id> <dG> <count>
    molecule`` followed by ``count + 1`` lines ``<nt_open> <ddG> <index>`` (the
    last ``ddG`` is 0).

    Args:
        path (str | os.PathLike): File to read.

    Returns:
        list[dict]: One dict per molecule with ``id``, ``dg`` (kcal/mol),
        ``gbp`` (the ``count`` energies) and ``nt_open`` (``count + 1``
        nucleotide counts). Pass the last two to
        :class:`unzipy.simulation.HairpinSim`.

    Raises:
        ValueError: If a block is truncated or a header is malformed.
    """
    lines = [
        ln.split() for ln in Path(path).read_text().splitlines() if ln.strip()
    ]
    out, i = [], 0
    while i < len(lines):
        head = lines[i]
        if head[-1] != "molecule":
            raise ValueError(
                f"line {i + 1}: expected a '... molecule' header, got {head}"
            )
        count = int(head[2])
        block = np.array(
            [[float(v) for v in ln[:2]] for ln in lines[i + 1 : i + count + 2]]
        )
        if block.shape[0] != count + 1:
            raise ValueError(f"molecule {head[0]}: truncated block")
        out.append(
            {
                "id": head[0],
                "dg": float(head[1]),
                "gbp": tuple(block[:-1, 1]),
                "nt_open": tuple(block[:, 0].astype(int)),
            }
        )
        i += count + 2
    return out


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_mc_log(path, n_params, skip_lines=0, drop=("T 0",)):
    """Parse the log of a Monte Carlo fit into an array.

    Lines may mix numbers with words such as ``rejected``, ``accepted
    (downhill)``, ``chi2`` or ``E1`` and with punctuation. The header lines are
    dropped, the strings in ``drop`` are removed, label tokens (``E1``,
    ``E_AT``) are discarded, and every number is read in order and arranged as
    ``n_params`` columns per Monte Carlo step.

    Args:
        path (str | os.PathLike): Log file.
        n_params (int): Numbers written per step.
        skip_lines (int): Header lines to skip.
        drop (tuple[str, ...]): Literal strings to delete before parsing.

    Returns:
        numpy.ndarray: Array of shape ``(n_steps, n_params)``. An incomplete
        last step is discarded.
    """
    lines = Path(path).read_text().splitlines()[skip_lines:]
    values = []
    for ln in lines:
        for d in drop:
            ln = ln.replace(d, " ")
        ln = re.sub(
            r"\b[A-Za-z_]\w*", " ", ln
        )  # words / labels such as E1, E_AT, chi2, rejected
        values.extend(float(t) for t in _NUMBER.findall(ln))
    n_steps = len(values) // n_params
    return np.array(values[: n_steps * n_params]).reshape(n_steps, n_params)
