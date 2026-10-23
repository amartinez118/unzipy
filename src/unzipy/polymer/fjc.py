"""Freely jointed chain (FJC) with Kuhn length ``b`` and contour ``L``.

Lengths in nm, forces in pN, energies in kT. The functions are stable for ``f
-> 0`` (series expansion) and large ``f`` (log-sinh form), and are even or odd
in the force as the physics requires.
"""

from __future__ import annotations

import numpy as np

from ..constants import KT25


def _langevin(z):
    """Return ``coth(z) - 1/z`` (odd in ``z``).

    Args:
        z (array-like): Dimensionless force ``f b / kT``.

    Returns:
        numpy.ndarray: Langevin function.
    """
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-3
    zs = np.where(small, 1.0, z)
    return np.where(small, z / 3.0 - z**3 / 45.0, 1.0 / np.tanh(zs) - 1.0 / zs)


def _logsinhc(z):
    """Return ``log(sinh(z) / z)`` (even in ``z``), stable at both ends.

    Args:
        z (array-like): Dimensionless force ``f b / kT``.

    Returns:
        numpy.ndarray: The logarithm.
    """
    z = np.abs(np.asarray(z, dtype=float))
    small = z < 1e-3
    zs = np.where(small, 1.0, z)
    big = zs + np.log1p(-np.exp(-2.0 * zs)) - np.log(2.0) - np.log(zs)
    return np.where(small, z * z / 6.0 - z**4 / 180.0, big)


def fjc_extension(f, b, L, kT=KT25):
    """Return the extension of an FJC at a given force.

    Args:
        f (array-like): Force (pN).
        b (float): Kuhn length (nm).
        L (float): Contour length (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Extension ``x`` in nm.
    """
    return L * _langevin(np.asarray(f, dtype=float) * b / kT)


def fjc_compliance(f, b, L, kT=KT25):
    """Return ``dx/df`` of an FJC.

    Args:
        f (array-like): Force (pN).
        b (float): Kuhn length (nm).
        L (float): Contour length (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Compliance in nm/pN.
    """
    z = np.abs(np.asarray(f, dtype=float) * b / kT)  # dx/df is even in f
    small = z < 1e-3
    zs = np.where(small, 1.0, z)
    core = np.where(
        small, 1.0 / 3.0 - z * z / 15.0, 1.0 / zs**2 - 1.0 / np.sinh(zs) ** 2
    )
    return L * b / kT * core


def fjc_gibbs(f, b, L, kT=KT25):
    """Return the force-ensemble free energy of an FJC.

    ``G = -(L / b) ln(sinh z / z)`` with ``z = f b / kT``, i.e. ``-integral of
    x df`` divided by ``kT``.

    Args:
        f (array-like): Force (pN).
        b (float): Kuhn length (nm).
        L (float): Contour length (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Free energy in kT (negative).
    """
    z = np.asarray(f, dtype=float) * b / kT
    return -L / b * _logsinhc(z)


def fjc_energy(f, b, L, kT=KT25):
    """Return the elastic energy of an FJC at a given force.

    This is ``integral of f dx`` divided by ``kT``, equal to ``G + f x / kT``.

    Args:
        f (array-like): Force (pN).
        b (float): Kuhn length (nm).
        L (float): Contour length (nm).
        kT (float): Thermal energy (pN nm).

    Returns:
        numpy.ndarray: Energy in kT.
    """
    f = np.asarray(f, dtype=float)
    return (
        fjc_gibbs(f, b, L, kT) + f * fjc_extension(f, b, L, kT) / kT
    )  # even in f
