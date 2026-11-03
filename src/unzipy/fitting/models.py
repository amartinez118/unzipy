"""Elastic elements and series models for fitting force-extension data.

An :class:`Element` is one elastic component (a WLC, an extensible FJC, the
optical trap, ...) with named parameters. A :class:`Model` puts elements in
series: at a given force their extensions add, optionally plus a constant
``shift`` (the arbitrary offset of the measured extension). This covers a
single strand, handles in series with a molecule, and the full trap, handle
and molecule chain of an optical-tweezers experiment.

Units: forces in pN, lengths in nm.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from ..constants import KT25
from ..polymer import (
    ewlc_extension,
    ewlc_scaled_extension,
    fjc_extension,
    wlc_extension,
)

_TINY = 1e-9


@dataclass(frozen=True)
class ParamSpec:
    """Specification of one model parameter.

    Attributes:
        name (str): Parameter name.
        value (float | None): Default starting value; None means the user
            must give one if the parameter is free.
        lower (float): Lower bound.
        upper (float): Upper bound.
        fixed (bool): Fixed at ``value`` unless freed explicitly.
    """

    name: str
    value: float | None = None
    lower: float = -np.inf
    upper: float = np.inf
    fixed: bool = False


@dataclass(frozen=True)
class Element:
    """One elastic component.

    Attributes:
        kind (str): Kind of element, e.g. ``"wlc"``.
        label (str): Label used to prefix the parameter names in a
            multi-element model.
        params (tuple[ParamSpec, ...]): Parameters of the element.
        func (Callable): ``func(f, kT, **values)`` returning the extension
            (nm) at force ``f`` (pN).
    """

    kind: str
    label: str
    params: tuple
    func: Callable


def wlc(label="wlc", bouchiat=False):
    """Return an inextensible worm-like chain element.

    Args:
        label (str): Element label.
        bouchiat (bool): Include the Bouchiat polynomial correction.

    Returns:
        Element: Element with parameters ``L`` (contour, nm) and ``p``
        (persistence length, nm).
    """

    def func(f, kT, L, p):
        return L * wlc_extension(f, p, kT, bouchiat)

    return Element(
        "wlc",
        label,
        (ParamSpec("L", lower=_TINY), ParamSpec("p", lower=_TINY)),
        func,
    )


def ewlc(label="ewlc", form="scaled"):
    """Return an extensible worm-like chain element.

    Args:
        label (str): Element label.
        form (str): ``"scaled"`` for ``Lc -> Lc (1 + f/Y)`` or
            ``"additive"`` for ``x/Lc = l_WLC(f) + f/S``.

    Returns:
        Element: Element with parameters ``L`` (nm), ``p`` (nm) and the
        modulus ``Y`` (scaled) or ``S`` (additive), in pN.

    Raises:
        ValueError: If ``form`` is unknown.
    """
    if form == "scaled":

        def func(f, kT, L, p, Y):  # pyright: ignore[reportRedeclaration]
            return L * ewlc_scaled_extension(f, p, Y, kT)

        modulus = "Y"
    elif form == "additive":

        def func(f, kT, L, p, S):
            return L * ewlc_extension(f, p, S, kT)

        modulus = "S"
    else:
        raise ValueError("form must be 'scaled' or 'additive'")
    return Element(
        "ewlc",
        label,
        (
            ParamSpec("L", lower=_TINY),
            ParamSpec("p", lower=_TINY),
            ParamSpec(modulus, lower=_TINY),
        ),
        func,
    )


def fjc(label="fjc"):
    """Return a freely jointed chain element.

    Args:
        label (str): Element label.

    Returns:
        Element: Element with parameters ``L`` (contour, nm) and ``b``
        (Kuhn length, nm).
    """

    def func(f, kT, L, b):
        return fjc_extension(f, b, L, kT)

    return Element(
        "fjc",
        label,
        (ParamSpec("L", lower=_TINY), ParamSpec("b", lower=_TINY)),
        func,
    )


def efjc(label="efjc"):
    """Return an extensible freely jointed chain element.

    The extension is ``(1 + f/Y)`` times that of the FJC.

    Args:
        label (str): Element label.

    Returns:
        Element: Element with parameters ``L`` (nm), ``b`` (nm) and the
        modulus ``Y`` (pN).
    """

    def func(f, kT, L, b, Y):
        return (1.0 + f / Y) * fjc_extension(f, b, L, kT)

    return Element(
        "efjc",
        label,
        (
            ParamSpec("L", lower=_TINY),
            ParamSpec("b", lower=_TINY),
            ParamSpec("Y", lower=_TINY),
        ),
        func,
    )


def dipole(label="dipole"):
    """Return the dipole (hairpin diameter) element of the folded state.

    Args:
        label (str): Element label.

    Returns:
        Element: Element with the single parameter ``d`` (diameter, nm).
    """

    def func(f, kT, d):
        return fjc_extension(f, d, d, kT)

    return Element("dipole", label, (ParamSpec("d", lower=_TINY),), func)


def trap(label="trap"):
    """Return an optical trap whose stiffness depends linearly on force.

    ``k_b(f) = k0 + alpha f`` gives the bead displacement
    ``ln(1 + alpha f / k0) / alpha`` (``f / k0`` when ``alpha = 0``).

    Args:
        label (str): Element label.

    Returns:
        Element: Element with parameters ``k0`` (pN/nm) and ``alpha``
        (1/nm, fixed at 0 unless freed).
    """

    def func(f, kT, k0, alpha):
        if abs(alpha) < 1e-12:
            return f / k0
        return np.log1p(alpha * f / k0) / alpha

    return Element(
        "trap",
        label,
        (
            ParamSpec("k0", lower=_TINY),
            ParamSpec("alpha", value=0.0, fixed=True),
        ),
        func,
    )


def ss_wlc(label="ss", bouchiat=False):
    """Return a single strand as a WLC parametrised by its nucleotides.

    Args:
        label (str): Element label.
        bouchiat (bool): Include the Bouchiat polynomial correction.

    Returns:
        Element: Element with parameters ``n`` (nucleotides), ``d0``
        (contour per nucleotide, nm) and ``p`` (persistence length, nm).
    """

    def func(f, kT, n, d0, p):
        return n * d0 * wlc_extension(f, p, kT, bouchiat)

    return Element(
        "ss_wlc",
        label,
        (
            ParamSpec("n", lower=_TINY),
            ParamSpec("d0", lower=_TINY),
            ParamSpec("p", lower=_TINY),
        ),
        func,
    )


def ss_fjc(label="ss"):
    """Return a single strand as an FJC parametrised by its nucleotides.

    Args:
        label (str): Element label.

    Returns:
        Element: Element with parameters ``n`` (nucleotides), ``d0``
        (contour per nucleotide, nm) and ``b`` (Kuhn length, nm).
    """

    def func(f, kT, n, d0, b):
        return fjc_extension(f, b, n * d0, kT)

    return Element(
        "ss_fjc",
        label,
        (
            ParamSpec("n", lower=_TINY),
            ParamSpec("d0", lower=_TINY),
            ParamSpec("b", lower=_TINY),
        ),
        func,
    )


class Model:
    """Elastic elements in series, with an optional constant offset.

    A single-element model uses the bare parameter names of the element
    (``L``, ``p``, ...). With several elements the names are prefixed by the
    element label (``handles.L``, ``ss.p``, ...).

    Attributes:
        elements (tuple[Element, ...]): The elements in series.
        offset (bool): Whether a constant ``shift`` is added.
        kT (float): Thermal energy (pN nm).
        specs (dict[str, ParamSpec]): Parameter specifications by full
            name.
    """

    def __init__(self, *elements, offset=True, kT=KT25):
        """Build a series model.

        Args:
            *elements (Element): Elements in series.
            offset (bool): Add a constant ``shift`` parameter (default 0).
            kT (float): Thermal energy (pN nm).

        Raises:
            ValueError: If there are no elements or labels repeat.
        """
        if not elements:
            raise ValueError("a model needs at least one element")
        labels = [e.label for e in elements]
        if len(set(labels)) != len(labels):
            raise ValueError(f"element labels must be unique: {labels}")
        self.elements = tuple(elements)
        self.offset = offset
        self.kT = kT
        single = len(elements) == 1
        self.specs = {}
        self._map = []
        for el in elements:
            local = {}
            for ps in el.params:
                full = ps.name if single else f"{el.label}.{ps.name}"
                self.specs[full] = replace(ps, name=full)
                local[ps.name] = full
            self._map.append((el, local))
        if offset:
            self.specs["shift"] = ParamSpec("shift", value=0.0)

    @property
    def names(self):
        """Return the parameter names.

        Returns:
            list[str]: Full parameter names.
        """
        return list(self.specs)

    def extension(self, f, values):
        """Return the extension at a given force.

        Args:
            f (array-like): Force (pN), non-negative.
            values (dict[str, float]): Value of every parameter.

        Returns:
            numpy.ndarray: Extension in nm.
        """
        f = np.asarray(f, dtype=float)
        x = 0.0
        for el, local in self._map:
            kw = {loc: values[full] for loc, full in local.items()}
            x = x + el.func(f, self.kT, **kw)
        if self.offset:
            x = x + values["shift"]
        return x

    def force(self, x, values, f_max=500.0, iters=70):
        """Return the force at a given extension.

        The extension is increasing in force, so the inversion is a
        vectorised bisection. Extensions beyond the reach of the model at
        ``f_max`` return ``f_max``; those below the offset return 0.

        Args:
            x (array-like): Extension (nm).
            values (dict[str, float]): Value of every parameter.
            f_max (float): Upper bound of the force (pN).
            iters (int): Bisection steps.

        Returns:
            numpy.ndarray: Force in pN.
        """
        x = np.asarray(x, dtype=float)
        lo = np.zeros_like(x)
        hi = np.full_like(x, f_max)
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            below = self.extension(mid, values) < x
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi)
