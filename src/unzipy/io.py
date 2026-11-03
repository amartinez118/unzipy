"""Reading experiment files and output of external programs."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
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


@dataclass
class Trace:
    """A raw pulling trace read by column name.

    Attributes:
        t (numpy.ndarray): Time (s).
        f (numpy.ndarray): Force (pN), absolute value of the chosen column.
        x (numpy.ndarray): Trap position (nm), the mean of the position
            columns.
        force (numpy.ndarray): Force with its original sign (pN).
        count (numpy.ndarray): Sample counter; NaN where the files have none.
        status (numpy.ndarray): Status code; NaN where the files have none.
        files (list[str]): Files read, in order.
    """

    t: np.ndarray
    f: np.ndarray
    x: np.ndarray
    force: np.ndarray
    count: np.ndarray
    status: np.ndarray
    files: list


def _read_named_columns(path, names, header_lines):
    """Read the named columns of one file.

    Args:
        path (str | os.PathLike): File to read.
        names (list[str]): Wanted column names; those absent from the file
            are returned as None.
        header_lines (int): Number of leading lines searched for the header.

    Returns:
        dict[str, numpy.ndarray | None]: Column by name.

    Raises:
        ValueError: If no leading line contains the force column or the
            file has no complete data row.
    """
    header = None
    with open(path, errors="replace") as fh:
        for _ in range(header_lines):
            cols = fh.readline().lstrip("#").split()
            if names[0] in cols:
                header = cols
                break
    if header is None:
        raise ValueError(f"no header with the force column in {path}")
    use = [header.index(n) for n in names if n in header]
    present = [n for n in names if n in header]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # empty file
            data = np.loadtxt(path, comments="#", usecols=use, ndmin=2)
    except ValueError:  # ragged rows: keep only the complete ones
        rows = []
        with open(path, errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if line.startswith("#") or len(parts) != len(header):
                    continue
                try:
                    rows.append([float(parts[c]) for c in use])
                except ValueError:
                    continue
        data = np.array(rows, dtype=float).reshape(-1, len(use))
    if data.shape[0] == 0:
        raise ValueError(f"no complete data rows in {path}")
    out = dict.fromkeys(names)
    for k, n in enumerate(present):
        out[n] = data[:, k]
    return out


def read_trace(
    paths,
    force="Y_force",
    positions=("A_dist-Y", "B_dist-Y"),
    time="time(sec)",
    count="CycleCount/n",
    status="Status",
    header_lines=6,
):
    """Read raw trace files by column name.

    The layout of instrument files varies between sessions (which columns are
    recorded, their order, whether a sample counter or a status column exists,
    where the header line sits), so columns are found from the header. Several
    files, for example the continuation files of one long run, are
    concatenated. Rows with a different number of fields than the header
    (truncated writes) are dropped, and ``#`` lines are comments.

    Args:
        paths (str | os.PathLike | list): File or files, in time order.
        force (str): Name of the force column.
        positions (tuple[str, ...]): Names of the trap position columns; the
            position is their mean.
        time (str): Name of the time column.
        count (str): Name of the sample counter column (optional).
        status (str): Name of the status column (optional).
        header_lines (int): Number of leading lines searched for the header.

    Returns:
        Trace: Time, force, position and the optional columns.

    Raises:
        ValueError: If a file has no header with the required columns or no
            complete data row.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    names = [force, time, *positions, count, status]
    parts = []
    for path in paths:
        cols = _read_named_columns(path, names, header_lines)
        missing = [n for n in (force, time, *positions) if cols[n] is None]
        if missing:
            raise ValueError(f"{path}: columns not found: {missing}")
        parts.append(cols)

    def cat(name):
        return np.concatenate(
            [
                p[name]
                if p[name] is not None
                else np.full(len(p[force]), np.nan)
                for p in parts
            ]
        )

    signed = cat(force)
    pos = np.mean([cat(n) for n in positions], axis=0)
    return Trace(
        t=cat(time),
        f=np.abs(signed),
        x=pos,
        force=signed,
        count=cat(count),
        status=cat(status),
        files=[str(p) for p in paths],
    )
