# Changelog

## Unreleased

- `unzipy.fitting`: general fitters of elastic models (WLC, eWLC, FJC, eFJC,
  dipole, trap, series chains) in `x(f)` and `f(x)`, with uncertainties,
  bootstrap intervals and model comparison.
- `fit_batch`, `fit_global` and `load_curves`: fits over many curves, with
  tables, summaries and parameters shared across molecules.
- Property-based tests (hypothesis) of the WLC/FJC, Jarzynski/Bennett and
  bias invariants.

## 0.1.0 (unreleased)

First public version: polymer elasticity; Jarzynski (with bias correction), Bennett and Crooks
estimators; position- and force-ensemble free-energy pipelines; stretching-model fit; unzipping
analysis (alignment, rupture rates, events, equilibrium FDC, splitting); nearest-neighbour energies
and landscapes; hairpin pulling simulation; command-line interface.
