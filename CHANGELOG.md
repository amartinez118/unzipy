# Changelog

## Unreleased

## 0.1.0 - 2026-11-03

- Release workflow: publishing a GitHub release runs lint and tests, then publishes to PyPI.
- `SECURITY.md`; README "Citing" section.
- `CITATION.cff`: citation metadata for the software.
- `unzipy.fitting`: general fitters of elastic models (WLC, eWLC, FJC, eFJC,
  dipole, trap, series chains) in `x(f)` and `f(x)`, with uncertainties,
  bootstrap intervals and model comparison.
- `fit_batch`, `fit_global` and `load_curves`: fits over many curves, with
  tables, summaries and parameters shared across molecules.
- Property-based tests (hypothesis) of the WLC/FJC, Jarzynski/Bennett and
  bias invariants.
- `split_cycles_adaptive`: splits traces without force limits.
- `read_trace`: reads raw files by column name (any layout, continuation files, truncated rows).
- `stiffness_before_rip`: effective stiffness of the folded branch just before the rip.
- Fixed the Jarzynski bias for small dissipation: the power law of eq. 7 grew with the number of
  work values for `<D> <= ln 2` (values up to 1e6 kT); `bias` now falls back to eq. 4 and is bounded.

## 0.1.0 (unreleased)

First public version: polymer elasticity; Jarzynski (with bias correction), Bennett and Crooks
estimators; position- and force-ensemble free-energy pipelines; stretching-model fit; unzipping
analysis (alignment, rupture rates, events, equilibrium FDC, splitting); nearest-neighbour energies
and landscapes; hairpin pulling simulation; command-line interface.
