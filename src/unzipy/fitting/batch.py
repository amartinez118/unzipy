"""Batch and global fits over many force-extension curves.

:func:`fit_batch` fits every curve independently and tabulates the results,
which suits one molecule per curve. :func:`fit_global` fits all curves at
once with some parameters shared among them (e.g. the elasticity of the same
handles) and the others free per curve (e.g. the contour length or offset of
each molecule), which constrains the shared parameters far better than the
individual fits.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import issparse, lil_matrix

from .curves import as_curves
from .fit import VARIABLES, _parameters, fit


@dataclass
class BatchReport:
    """Independent fits of many curves.

    Attributes:
        labels (list[str]): Curve labels.
        reports (list[FitReport | None]): Fit of each curve; None if it
            failed (see ``errors``).
        errors (dict[str, str]): Error message by label for failed curves.
        variable (str): ``"x(f)"`` or ``"f(x)"``.
    """

    labels: list
    reports: list
    errors: dict
    variable: str

    @property
    def ok(self):
        """Return the successful fits.

        Returns:
            dict[str, FitReport]: Fits by label, converged ones only.
        """
        return {
            lab: r
            for lab, r in zip(self.labels, self.reports, strict=True)
            if r is not None and r.success
        }

    @property
    def n(self):
        """Return the total number of data points of the successful fits.

        Returns:
            int: Sum of the sizes of the curves.
        """
        return sum(r.n for r in self.ok.values())

    @property
    def n_free(self):
        """Return the total number of fitted parameters.

        Returns:
            int: Sum over the successful fits.
        """
        return sum(r.n_free for r in self.ok.values())

    @property
    def chi2(self):
        """Return the summed chi-square of the successful fits.

        Returns:
            float: Total chi-square.
        """
        return sum(r.chi2 for r in self.ok.values())

    @property
    def aic(self):
        """Return the Akaike criterion of all fits taken together.

        Returns:
            float: ``n ln(chi2 / n) + 2 k`` with the totals.
        """
        return self.n * np.log(self.chi2 / self.n) + 2 * self.n_free

    @property
    def bic(self):
        """Return the Bayesian criterion of all fits taken together.

        Returns:
            float: ``n ln(chi2 / n) + k ln n`` with the totals.
        """
        return self.n * np.log(self.chi2 / self.n) + self.n_free * np.log(
            self.n
        )

    def values(self, name):
        """Return one parameter across the successful curves.

        Args:
            name (str): Parameter name.

        Returns:
            numpy.ndarray: Fitted values in curve order.
        """
        return np.array([r.values[name] for r in self.ok.values()])

    def table(self):
        """Return the results as one row per curve.

        Returns:
            list[dict]: Rows with ``label``, ``success``, ``n``, ``rms``,
            ``aic`` and, for every parameter, its value and ``<name>_err``;
            failed curves have only ``label``, ``success`` and ``error``.
        """
        rows = []
        for lab, r in zip(self.labels, self.reports, strict=True):
            if r is None:
                rows.append(
                    {"label": lab, "success": False, "error": self.errors[lab]}
                )
                continue
            row = {
                "label": lab,
                "success": r.success,
                "n": r.n,
                "rms": r.rms,
                "aic": r.aic,
            }
            for k, v in r.values.items():
                row[k] = v
                row[f"{k}_err"] = r.errors[k]
            rows.append(row)
        return rows

    def summary(self):
        """Summarise the fitted parameters across molecules.

        Returns:
            dict[str, dict]: For each fitted parameter the ``mean``,
            ``std`` (between curves), ``sem`` and number ``n`` of curves.
        """
        ok = list(self.ok.values())
        if not ok:
            return {}
        out = {}
        for k in ok[0].free:
            v = np.array([r.values[k] for r in ok])
            std = v.std(ddof=1) if v.size > 1 else np.nan
            out[k] = {
                "mean": float(v.mean()),
                "std": float(std),
                "sem": float(std / np.sqrt(v.size)),
                "n": int(v.size),
            }
        return out

    def to_csv(self, path):
        """Write the table to a CSV file.

        Args:
            path (str | os.PathLike): Output file.
        """
        rows = self.table()
        cols = list(dict.fromkeys(k for row in rows for k in row))
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            writer.writerows(rows)


def fit_batch(model, curves, on_error="skip", **kwargs):
    """Fit many curves independently with the same model.

    Args:
        model (Model): Elastic model.
        curves (list): Curves in any form accepted by
            :func:`unzipy.fitting.curves.as_curves`.
        on_error (str): ``"skip"`` records a failing curve in
            ``BatchReport.errors``; ``"raise"`` propagates the error.
        **kwargs: Passed to :func:`unzipy.fitting.fit` (``start``,
            ``fixed``, ``free``, ``bounds``, ``variable``, ...). A curve's
            own ``sigma`` is used unless ``sigma`` is given here; ``start``
            may also be a list of dicts, one per curve.

    Returns:
        BatchReport: The fits and their table.

    Raises:
        ValueError: If ``on_error`` is invalid, or a fit fails with
            ``on_error="raise"``.
    """
    if on_error not in ("skip", "raise"):
        raise ValueError("on_error must be 'skip' or 'raise'")
    curves = as_curves(curves)
    start = kwargs.pop("start", None)
    starts = (
        start if isinstance(start, (list, tuple)) else [start] * len(curves)
    )
    if len(starts) != len(curves):
        raise ValueError("start must be a dict or one dict per curve")
    variable = kwargs.get("variable", "x(f)")
    reports, errors = [], {}
    for c, st in zip(curves, starts, strict=True):
        opts = dict(kwargs)
        opts.setdefault("sigma", c.sigma)
        try:
            reports.append(fit(model, c.f, c.x, start=st, **opts))
        except (ValueError, np.linalg.LinAlgError) as err:
            if on_error == "raise":
                raise
            reports.append(None)
            errors[c.label] = str(err)
    return BatchReport([c.label for c in curves], reports, errors, variable)


@dataclass
class GlobalReport:
    """Global fit of many curves with shared and per-curve parameters.

    Attributes:
        model (Model): The fitted model.
        variable (str): ``"x(f)"`` or ``"f(x)"``.
        labels (list[str]): Curve labels.
        shared (dict[str, float]): Fitted shared parameters.
        shared_errors (dict[str, float]): Their standard errors.
        per_curve (list[dict]): Fitted per-curve parameters of each curve.
        per_curve_errors (list[dict]): Their standard errors.
        fixed (dict[str, float]): Fixed parameters (common to all curves).
        free (tuple[str, ...]): Names of every fitted parameter: the shared
            ones, then ``name[label]`` for the per-curve ones.
        cov (numpy.ndarray): Covariance matrix of the fitted parameters.
        n (int): Total number of data points.
        dof (int): Degrees of freedom.
        chi2 (float): Total chi-square.
        rms (float): Root-mean-square residual in data units.
        aic (float): Akaike information criterion.
        bic (float): Bayesian information criterion.
        success (bool): Whether the optimiser converged.
        message (str): Optimiser message.
    """

    model: object
    variable: str
    labels: list
    shared: dict
    shared_errors: dict
    per_curve: list
    per_curve_errors: list
    fixed: dict
    free: tuple
    cov: np.ndarray
    n: int
    dof: int
    chi2: float
    rms: float
    aic: float
    bic: float
    success: bool
    message: str

    @property
    def n_free(self):
        """Return the number of fitted parameters.

        Returns:
            int: Length of ``free``.
        """
        return len(self.free)

    def values_for(self, label):
        """Return every parameter of one curve.

        Args:
            label (str): Curve label.

        Returns:
            dict[str, float]: Fixed, shared and per-curve values.

        Raises:
            ValueError: If the label is unknown.
        """
        if label not in self.labels:
            raise ValueError(f"unknown curve {label!r}")
        i = self.labels.index(label)
        return {**self.fixed, **self.shared, **self.per_curve[i]}

    def predict(self, label, grid, f_max=500.0):
        """Evaluate the fitted curve of one molecule.

        Args:
            label (str): Curve label.
            grid (array-like): Forces (pN) for an ``x(f)`` fit, extensions
                (nm) for an ``f(x)`` fit.
            f_max (float): Upper force bound when inverting (pN).

        Returns:
            numpy.ndarray: Extension (nm) or force (pN) of the model.
        """
        v = self.values_for(label)
        grid = np.asarray(grid, dtype=float)
        if self.variable == "x(f)":
            return self.model.extension(np.clip(grid, 0, None), v)
        return self.model.force(grid, v, f_max)

    def table(self):
        """Return the per-curve parameters as one row per curve.

        Returns:
            list[dict]: Rows with ``label`` and, for every per-curve
            parameter, its value and ``<name>_err``.
        """
        rows = []
        for lab, vals, errs in zip(
            self.labels, self.per_curve, self.per_curve_errors, strict=True
        ):
            row = {"label": lab}
            for k, v in vals.items():
                row[k] = v
                row[f"{k}_err"] = errs[k]
            rows.append(row)
        return rows

    def summary(self):
        """Return a text summary of the shared parameters and fit quality.

        Returns:
            str: Human-readable summary.
        """
        lines = [
            f"global fit of {self.variable}: {len(self.labels)} curves, "
            f"n={self.n}, dof={self.dof}"
        ]
        for k, v in self.shared.items():
            lines.append(
                f"  shared {k} = {v:.6g} +/- {self.shared_errors[k]:.2g}"
            )
        for k, v in self.fixed.items():
            lines.append(f"  fixed  {k} = {v:.6g}")
        lines.append(
            f"  rms={self.rms:.4g}  chi2={self.chi2:.4g}  "
            f"AIC={self.aic:.2f}  BIC={self.bic:.2f}"
        )
        return "\n".join(lines)


def fit_global(
    model,
    curves,
    shared,
    *,
    variable="x(f)",
    start=None,
    fixed=None,
    free=None,
    bounds=None,
    absolute_sigma=False,
    f_max=500.0,
    max_nfev=500,
):
    """Fit many curves at once with some parameters shared among them.

    Every fitted parameter listed in ``shared`` takes a single value for all
    curves; every other fitted parameter (``shift``, contour lengths, ...)
    is fitted separately for each curve. Fixed parameters are common to all
    curves.

    Args:
        model (Model): Elastic model.
        curves (list): Curves in any form accepted by
            :func:`unzipy.fitting.curves.as_curves`; their own ``sigma`` is
            used.
        shared (list[str]): Names of the parameters shared by all curves.
        variable (str): ``"x(f)"`` or ``"f(x)"``.
        start (dict[str, float] | None): Starting values by name, used for
            every curve; a name may instead map to a list with one value per
            curve for the per-curve parameters.
        fixed (dict[str, float | None] | None): Parameters held fixed.
        free (list[str] | None): Default-fixed parameters to fit.
        bounds (dict[str, tuple[float, float]] | None): Bounds overriding
            the defaults.
        absolute_sigma (bool): Treat the curves' ``sigma`` as absolute.
        f_max (float): Upper force bound when inverting for ``f(x)`` (pN).
        max_nfev (int): Maximum number of function evaluations.

    Returns:
        GlobalReport: Shared and per-curve parameters with errors and
        criteria.

    Raises:
        ValueError: If fewer than two curves are given, ``shared`` names a
            parameter that is not fitted, nothing is shared or per curve, or
            there are not enough data points.
    """
    if variable not in VARIABLES:
        raise ValueError(f"variable must be one of {VARIABLES}")
    curves = as_curves(curves)
    if len(curves) < 2:
        raise ValueError("a global fit needs at least two curves")
    start = dict(start or {})
    per_curve_start = {
        k: v
        for k, v in start.items()
        if isinstance(v, (list, tuple, np.ndarray))
    }
    base_start = {k: v for k, v in start.items() if k not in per_curve_start}
    for k, v in per_curve_start.items():
        if len(v) != len(curves):
            raise ValueError(f"start[{k!r}] needs one value per curve")
        base_start[k] = float(v[0])
    values, held, free_names, lower, upper = _parameters(
        model, base_start, fixed, free, bounds
    )
    shared = list(shared)
    bad = [k for k in shared if k not in free_names]
    if bad:
        raise ValueError(f"shared parameters that are not fitted: {bad}")
    shared_names = tuple(k for k in free_names if k in shared)
    curve_names = tuple(k for k in free_names if k not in shared)
    if not shared_names or not curve_names:
        raise ValueError(
            "need both shared and per-curve fitted parameters "
            "(use fit_batch for fully independent fits)"
        )
    ns, nc, ncurve = len(shared_names), len(curve_names), len(curves)
    limit = dict(zip(free_names, zip(lower, upper, strict=True), strict=True))
    sig = []
    for c in curves:
        if c.f.size != c.x.size:
            raise ValueError(f"curve {c.label}: f and x differ in length")
        s = (
            np.ones(c.f.size)
            if c.sigma is None
            else np.broadcast_to(
                np.asarray(c.sigma, dtype=float), c.f.shape
            ).copy()
        )
        if np.any(s <= 0):
            raise ValueError(f"curve {c.label}: sigma must be positive")
        sig.append(s)
    n_total = sum(c.f.size for c in curves)
    npar = ns + nc * ncurve
    if n_total <= npar:
        raise ValueError(f"need more than {npar} points for {npar} parameters")

    theta0 = np.empty(npar)
    lo = np.empty(npar)
    hi = np.empty(npar)
    for j, k in enumerate(shared_names):
        theta0[j], lo[j], hi[j] = values[k], *limit[k]
    for i in range(ncurve):
        for j, k in enumerate(curve_names):
            col = ns + i * nc + j
            v0 = per_curve_start[k][i] if k in per_curve_start else values[k]
            theta0[col], lo[col], hi[col] = v0, *limit[k]
    theta0 = np.clip(theta0, lo, hi)

    sizes = [c.f.size for c in curves]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    sparsity = lil_matrix((n_total, npar), dtype=int)
    for i in range(ncurve):
        rows = slice(offsets[i], offsets[i + 1])
        sparsity[rows, :ns] = 1
        sparsity[rows, ns + i * nc : ns + (i + 1) * nc] = 1

    def curve_values(theta, i):
        v = dict(values)
        v.update(zip(shared_names, theta[:ns], strict=True))
        v.update(
            zip(
                curve_names,
                theta[ns + i * nc : ns + (i + 1) * nc],
                strict=True,
            )
        )
        return v

    def resid(theta):
        parts = []
        for i, c in enumerate(curves):
            v = curve_values(theta, i)
            if variable == "x(f)":
                r = model.extension(np.clip(c.f, 0.0, None), v) - c.x
            else:
                r = model.force(c.x, v, f_max) - c.f
            parts.append(r / sig[i])
        return np.concatenate(parts)

    res = least_squares(
        resid,
        theta0,
        bounds=(lo, hi),
        x_scale="jac",
        jac_sparsity=sparsity,
        max_nfev=max_nfev,
    )
    jac = res.jac.toarray() if issparse(res.jac) else res.jac
    dof = n_total - npar
    chi2 = float(np.sum(res.fun**2))
    cov = np.linalg.pinv(jac.T @ jac)
    absolute = absolute_sigma and any(c.sigma is not None for c in curves)
    if not absolute:
        cov = cov * chi2 / dof
    err = np.sqrt(np.diag(cov))
    if absolute:
        aic = chi2 + 2 * npar
        bic = chi2 + npar * np.log(n_total)
    else:
        aic = n_total * np.log(chi2 / n_total) + 2 * npar
        bic = n_total * np.log(chi2 / n_total) + npar * np.log(n_total)
    labels = [c.label for c in curves]
    free_labels = list(shared_names) + [
        f"{k}[{lab}]" for lab in labels for k in curve_names
    ]
    weights = np.concatenate(sig)
    return GlobalReport(
        model=model,
        variable=variable,
        labels=labels,
        shared=dict(zip(shared_names, res.x[:ns], strict=True)),
        shared_errors=dict(zip(shared_names, err[:ns], strict=True)),
        per_curve=[
            dict(
                zip(
                    curve_names,
                    res.x[ns + i * nc : ns + (i + 1) * nc],
                    strict=True,
                )
            )
            for i in range(ncurve)
        ],
        per_curve_errors=[
            dict(
                zip(
                    curve_names,
                    err[ns + i * nc : ns + (i + 1) * nc],
                    strict=True,
                )
            )
            for i in range(ncurve)
        ],
        fixed={k: values[k] for k in held},
        free=tuple(free_labels),
        cov=cov,
        n=n_total,
        dof=dof,
        chi2=chi2,
        rms=float(np.sqrt(np.mean((res.fun * weights) ** 2))),
        aic=float(aic),
        bic=float(bic),
        success=bool(res.success),
        message=str(res.message),
    )
