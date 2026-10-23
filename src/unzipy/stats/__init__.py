"""Statistics: bootstrap, KDE, outliers and Gaussian mixtures."""

from .bootstrap import bin_with_errors, bootstrap, bootstrap_pdf
from .kde import kde_pdf, optimal_bandwidth
from .mixture import GaussianMixture, fit_gmm
from .outliers import grubbs_outliers, iqr_outliers

__all__ = [
    "GaussianMixture",
    "bin_with_errors",
    "bootstrap",
    "bootstrap_pdf",
    "fit_gmm",
    "grubbs_outliers",
    "iqr_outliers",
    "kde_pdf",
    "optimal_bandwidth",
]
