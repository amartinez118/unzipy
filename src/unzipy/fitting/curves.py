"""Force-extension curves as inputs of batch and global fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_curve


@dataclass
class Curve:
    """One force-extension curve.

    Attributes:
        f (numpy.ndarray): Force (pN).
        x (numpy.ndarray): Extension (nm).
        sigma (numpy.ndarray | float | None): Standard deviation of the
            fitted variable; unit weights if None.
        label (str | None): Name of the curve (molecule, file, ...).
    """

    f: np.ndarray
    x: np.ndarray
    sigma: object = None
    label: str | None = None


def as_curves(items):
    """Normalise curves given in several forms.

    Args:
        items (list): Each item is a :class:`Curve`, a ``(f, x)`` or
            ``(f, x, sigma)`` tuple, or a dict with keys ``f``, ``x`` and
            optionally ``sigma`` and ``label``.

    Returns:
        list[Curve]: Curves with array data and unique labels
        (``curve0``, ``curve1``, ... for unlabelled ones).

    Raises:
        ValueError: If there are no curves, an item has an unsupported
            form, or labels repeat.
    """
    out = []
    for i, it in enumerate(items):
        if isinstance(it, Curve):
            c = it
        elif isinstance(it, dict):
            c = Curve(it["f"], it["x"], it.get("sigma"), it.get("label"))
        elif isinstance(it, (tuple, list)) and len(it) in (2, 3):
            c = Curve(*it)
        else:
            raise ValueError(f"curve {i}: unsupported form {type(it)}")
        out.append(
            Curve(
                np.asarray(c.f, dtype=float).ravel(),
                np.asarray(c.x, dtype=float).ravel(),
                c.sigma,
                c.label if c.label is not None else f"curve{i}",
            )
        )
    if not out:
        raise ValueError("no curves given")
    labels = [c.label for c in out]
    if len(set(labels)) != len(labels):
        raise ValueError(f"curve labels must be unique: {labels}")
    return out


def load_curves(
    directory, pattern="*.txt", names=None, f_col=3, x_col=1, sigma=None
):
    """Read curves from the files of a folder.

    Args:
        directory (str | os.PathLike): Folder with the curve files.
        pattern (str): Glob pattern of the files read (ignored if
            ``names`` is given).
        names (list[str] | None): File stems to read, in this order.
        f_col (int): Zero-based force column.
        x_col (int): Zero-based extension column.
        sigma (float | None): Standard deviation assigned to every curve.

    Returns:
        list[Curve]: One curve per file, labelled by the file stem.

    Raises:
        ValueError: If no file is found.
    """
    root = Path(directory)
    paths = (
        [root / f"{n}.txt" for n in names]
        if names is not None
        else sorted(root.glob(pattern))
    )
    if not paths:
        raise ValueError(f"no curve files found in {root}")
    curves = []
    for p in paths:
        x, f = read_curve(p, x_col=x_col, f_col=f_col)
        curves.append(Curve(f, x, sigma, p.stem))
    return as_curves(curves)
