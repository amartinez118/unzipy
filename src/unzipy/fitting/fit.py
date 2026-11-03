"""Least-squares fits of force-extension data to elastic models.

Two variables can be fitted: the extension as a function of force,
``x(f)`` (noise on the extension; closed-form for every model), or the
force as a function of extension, ``f(x)`` (noise on the force, as in
position-controlled experiments; the model is inverted numerically).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .models import Model

VARIABLES = ("x(f)", "f(x)")


@dataclass
class FitReport:
    """Result of fitting a model to one force-extension curve.

    Attributes:
        model (Model): The fitted model.
        variable (str): ``"x(f)"`` or ``"f(x)"``.
        values (dict[str, float]): Value of every parameter.
        errors (dict[str, float]): Standard error of every parameter
            (NaN for fixed ones).
        free (tuple[str, ...]): Names of the fitted parameters.
        cov (numpy.ndarray): Covariance matrix of the free parameters.
        residuals (numpy.ndarray): Weighted residuals (data units divided
            by ``sigma``), model minus data.
        n (int): Number of data points.
        dof (int): Degrees of freedom, ``n`` minus the free parameters.
        chi2 (float): Sum of squared weighted residuals.
        rms (float): Root-mean-square residual in data units.
        aic (float): Akaike information criterion.
        bic (float): Bayesian information criterion.
        success (bool): Whether the optimiser converged.
        message (str): Optimiser message.
    """

    model: Model
    variable: str
    values: dict
    errors: dict
    free: tuple
    cov: np.ndarray
    residuals: np.ndarray
    n: int
    dof: int
    chi2: float
    rms: float
    aic: float
    bic: float
    success: bool
    message: str
    _data: tuple = field(repr=False, default=())
    _options: dict = field(repr=False, default_factory=dict)

    @property
    def n_free(self):
        """Return the number of fitted parameters.

        Returns:
            int: Length of ``free``.
        """
        return len(self.free)

    @property
    def reduced_chi2(self):
        """Return ``chi2 / dof``.

        Returns:
            float: Reduced chi-square (about 1 for correct ``sigma``).
        """
        return self.chi2 / self.dof

    @property
    def correlation(self):
        """Return the correlation matrix of the free parameters.

        Returns:
            numpy.ndarray: Correlation matrix.
        """
        sd = np.sqrt(np.diag(self.cov))
        return self.cov / np.outer(sd, sd)

    def predict(self, grid):
        """Evaluate the fitted curve.

        Args:
            grid (array-like): Forces (pN) if the fit was ``x(f)``,
                extensions (nm) if it was ``f(x)``.

        Returns:
            numpy.ndarray: Extension (nm) or force (pN) of the model.
        """
        grid = np.asarray(grid, dtype=float)
        if self.variable == "x(f)":
            return self.model.extension(np.clip(grid, 0, None), self.values)
        f_max = self._options.get("f_max", 500.0)
        return self.model.force(grid, self.values, f_max)

    def summary(self):
        """Return a text table of the parameters and fit quality.

        Returns:
            str: Human-readable summary.
        """
        lines = [f"fit of {self.variable}: n={self.n}, dof={self.dof}"]
        width = max(len(k) for k in self.values)
        for k, v in self.values.items():
            if k in self.free:
                lines.append(
                    f"  {k:<{width}} = {v:.6g} +/- {self.errors[k]:.2g}"
                )
            else:
                lines.append(f"  {k:<{width}} = {v:.6g} (fixed)")
        lines.append(
            f"  rms={self.rms:.4g}  chi2={self.chi2:.4g}  "
            f"AIC={self.aic:.2f}  BIC={self.bic:.2f}"
        )
        return "\n".join(lines)

    def bootstrap(self, n_boot=500, level=0.95, rng=None):
        """Estimate confidence intervals by resampling the data points.

        Pairs ``(f, x)`` are resampled with replacement and the model is
        refitted from the best-fit values; failed refits are skipped.

        Args:
            n_boot (int): Number of resamples.
            level (float): Confidence level of the intervals.
            rng (int | numpy.random.Generator | None): Seed or generator.

        Returns:
            dict: ``samples`` (array of shape ``(n_ok, n_free)``), ``names``
            (the free parameters) and ``ci`` (name -> ``(low, high)``).
        """
        gen = np.random.default_rng(rng)
        f, x, sigma = self._data
        rows = []
        for _ in range(n_boot):
            idx = gen.integers(0, self.n, self.n)
            try:
                r = fit(
                    self.model,
                    f[idx],
                    x[idx],
                    variable=self.variable,
                    sigma=sigma[idx],
                    start=self.values,
                    **self._options,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            if r.success:
                rows.append([r.values[k] for k in self.free])
        samples = np.array(rows)
        q = [100 * (1 - level) / 2, 100 * (1 + level) / 2]
        ci = {
            k: tuple(np.percentile(samples[:, i], q))
            for i, k in enumerate(self.free)
        }
        return {"samples": samples, "names": self.free, "ci": ci}


def _parameters(model, start, fixed, free, bounds):
    """Resolve which parameters are free and their values and limits.

    Args:
        model (Model): Elastic model.
        start (dict | None): Starting values by name.
        fixed (dict | None): Parameters held fixed (value or None).
        free (list[str] | None): Default-fixed parameters to fit.
        bounds (dict | None): Bounds overriding the defaults.

    Returns:
        tuple: ``values`` (every parameter), ``held`` (set of fixed
        names), ``free_names`` (tuple), and ``lower`` and ``upper`` bound
        arrays of the free parameters.

    Raises:
        ValueError: If names are unknown, nothing is free, or a free
            parameter has no starting value.
    """
    start, fixed, bounds = start or {}, fixed or {}, bounds or {}
    free = list(free or [])
    for group, label in (
        (start, "start"),
        (fixed, "fixed"),
        (bounds, "bounds"),
    ):
        unknown = set(group) - set(model.names)
        if unknown:
            raise ValueError(
                f"unknown parameters in {label}: {sorted(unknown)}"
            )
    unknown = set(free) - set(model.names)
    if unknown:
        raise ValueError(f"unknown parameters in free: {sorted(unknown)}")
    values = {k: s.value for k, s in model.specs.items()}
    values.update(start)
    held = {
        k for k, s in model.specs.items() if s.fixed and k not in free
    } | set(fixed)
    for k, v in fixed.items():
        if v is not None:
            values[k] = v
    free_names = tuple(k for k in model.names if k not in held)
    if not free_names:
        raise ValueError("no free parameters to fit")
    missing = [k for k, v in values.items() if v is None]
    if missing:
        raise ValueError(f"give starting values (start=...) for {missing}")
    limits = [
        bounds.get(k, (model.specs[k].lower, model.specs[k].upper))
        for k in free_names
    ]
    lower = np.array([lim[0] for lim in limits], dtype=float)
    upper = np.array([lim[1] for lim in limits], dtype=float)
    return values, held, free_names, lower, upper


def fit(
    model,
    f,
    x,
    *,
    variable="x(f)",
    sigma=None,
    start=None,
    fixed=None,
    free=None,
    bounds=None,
    absolute_sigma=False,
    f_max=500.0,
    max_nfev=2000,
):
    """Fit a model to a force-extension curve by non-linear least squares.

    Negative forces in the data (noise around zero) are treated as zero
    when evaluating ``x(f)``.

    Args:
        model (Model): Elastic model.
        f (array-like): Force (pN).
        x (array-like): Extension (nm).
        variable (str): ``"x(f)"`` (noise on the extension) or ``"f(x)"``
            (noise on the force; the model is inverted numerically).
        sigma (float | array-like | None): Standard deviation of the
            fitted variable; the same for all points if a scalar. Unit
            weights if None.
        start (dict[str, float] | None): Starting values by parameter name.
            Parameters without a default must be given here.
        fixed (dict[str, float | None] | None): Parameters held fixed, at
            the given value or, if None, at their starting value.
        free (list[str] | None): Parameters that are fixed by default
            (such as ``trap.alpha``) but should be fitted.
        bounds (dict[str, tuple[float, float]] | None): Bounds overriding
            the defaults.
        absolute_sigma (bool): Treat ``sigma`` as absolute, so the
            covariance is not rescaled by the reduced chi-square.
        f_max (float): Upper bound of the force when inverting the model
            for ``f(x)`` fits (pN).
        max_nfev (int): Maximum number of function evaluations.

    Returns:
        FitReport: Fitted values, errors, residuals and criteria.

    Raises:
        ValueError: If the data or the parameters are inconsistent: wrong
            shapes, non-finite values, unknown names, missing start values,
            no free parameter or too few points.
    """
    if variable not in VARIABLES:
        raise ValueError(f"variable must be one of {VARIABLES}")
    f = np.asarray(f, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if f.size != x.size:
        raise ValueError("f and x must have the same length")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(x))):
        raise ValueError("f and x must be finite")
    sig = (
        np.ones(f.size)
        if sigma is None
        else np.broadcast_to(np.asarray(sigma, dtype=float), f.shape).copy()
    )
    if np.any(sig <= 0):
        raise ValueError("sigma must be positive")

    values, held, free_names, lower, upper = _parameters(
        model, start, fixed, free, bounds
    )
    if f.size <= len(free_names):
        raise ValueError(
            f"need more than {len(free_names)} points for "
            f"{len(free_names)} free parameters"
        )
    theta0 = np.clip([values[k] for k in free_names], lower, upper)
    f_use = np.clip(f, 0.0, None)

    def vals(theta):
        out = dict(values)
        out.update(zip(free_names, theta, strict=True))
        return out

    def resid(theta):
        v = vals(theta)
        if variable == "x(f)":
            return (model.extension(f_use, v) - x) / sig
        return (model.force(x, v, f_max) - f) / sig

    res = least_squares(
        resid,
        theta0,
        bounds=(lower, upper),
        x_scale="jac",
        max_nfev=max_nfev,
    )
    n, k = f.size, len(free_names)
    dof = n - k
    chi2 = float(np.sum(res.fun**2))
    cov = np.linalg.pinv(res.jac.T @ res.jac)
    if not (absolute_sigma and sigma is not None):
        cov = cov * chi2 / dof
    errors = {name: np.nan for name in model.names}
    errors.update(zip(free_names, np.sqrt(np.diag(cov)), strict=True))
    if absolute_sigma and sigma is not None:
        aic, bic = chi2 + 2 * k, chi2 + k * np.log(n)
    else:
        aic = n * np.log(chi2 / n) + 2 * k
        bic = n * np.log(chi2 / n) + k * np.log(n)
    final = vals(res.x)
    return FitReport(
        model=model,
        variable=variable,
        values=final,
        errors=errors,
        free=free_names,
        cov=cov,
        residuals=res.fun,
        n=n,
        dof=dof,
        chi2=chi2,
        rms=float(np.sqrt(np.mean((res.fun * sig) ** 2))),
        aic=float(aic),
        bic=float(bic),
        success=bool(res.success),
        message=str(res.message),
        _data=(f, x, sig),
        _options={
            "fixed": {k: final[k] for k in held},
            "bounds": bounds,
            "absolute_sigma": absolute_sigma and sigma is not None,
            "f_max": f_max,
            "max_nfev": max_nfev,
        },
    )


def compare(**reports):
    """Compare fits of the same data with information criteria.

    Args:
        **reports (FitReport): Fits by label; they must share the data size
            and the fitted variable.

    Returns:
        list[dict]: One row per fit sorted by AIC, with ``label``, ``k``
        (free parameters), ``aic``, ``delta_aic``, ``weight`` (Akaike
        weight), ``bic`` and ``delta_bic``.

    Raises:
        ValueError: If fewer than two fits are given or they are not
            comparable.
    """
    if len(reports) < 2:
        raise ValueError("give at least two fits to compare")
    first = next(iter(reports.values()))
    for r in reports.values():
        if r.n != first.n or r.variable != first.variable:
            raise ValueError("fits must use the same data and variable")
    aic_min = min(r.aic for r in reports.values())
    bic_min = min(r.bic for r in reports.values())
    raw = {k: np.exp(-0.5 * (r.aic - aic_min)) for k, r in reports.items()}
    total = sum(raw.values())
    rows = [
        {
            "label": k,
            "k": r.n_free,
            "aic": r.aic,
            "delta_aic": r.aic - aic_min,
            "weight": raw[k] / total,
            "bic": r.bic,
            "delta_bic": r.bic - bic_min,
        }
        for k, r in reports.items()
    ]
    return sorted(rows, key=lambda row: row["aic"])
