"""Work from force-extension curves, in kT."""

from __future__ import annotations

import numpy as np

from ..constants import KT25


def work_position(x, f, kT=KT25):
    """Return the work done at controlled position, ``int f dx / kT``.

    The integral uses the trapezoid rule along the order of the samples, so a
    path with decreasing ``x`` gives a negative work.

    Args:
        x (array-like): Extension (nm) along the trajectory.
        f (array-like): Force (pN) along the trajectory.
        kT (float): Thermal energy (pN nm).

    Returns:
        float: Work in kT.
    """
    return np.trapezoid(f, x) / kT


def work_force(x, f, kT=KT25):
    """Return the work at controlled force, ``-int x df / kT``.

    The integral uses the trapezoid rule along the order of the samples.

    Args:
        x (array-like): Extension (nm) along the trajectory.
        f (array-like): Force (pN) along the trajectory.
        kT (float): Thermal energy (pN nm).

    Returns:
        float: Work in kT.
    """
    return -np.trapezoid(x, f) / kT
