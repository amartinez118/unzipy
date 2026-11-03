# unzipy

[![CI](https://github.com/amartinez118/unzipy/actions/workflows/ci.yml/badge.svg)](https://github.com/amartinez118/unzipy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Single-molecule force spectroscopy and nonequilibrium statistical physics in Python.**

`unzipy` collects the analysis methods used for optical- and magnetic-tweezers pulling experiments
on nucleic-acid hairpins, as a tested, vectorised, importable library:

- polymer elasticity (worm-like chain with Bouchiat correction, extensible WLC, freely jointed chain);
- free-energy estimation from work (Jarzynski with bias correction, Bennett acceptance ratio,
  Crooks fluctuation theorem, dissipated work), for position-controlled and force-controlled pulling;
- recovery of folding free energies at zero force by subtracting trap, handle and molecule stretching
  (effective-stiffness method and a full elastic-model fit);
- unzipping analysis: force–distance-curve alignment, rupture-force rates, event detection,
  equilibrium force–distance curves from sequence energies;
- nearest-neighbour hairpin energies, free-energy landscapes and a stochastic pulling simulation.

Status: **alpha** — the API
may still change before 1.0.

## Install

```
pip install git+https://github.com/amartinez118/unzipy      # or, from a clone:  pip install -e ".[dev]"
```

Requires Python ≥ 3.10, numpy ≥ 2.0 and scipy ≥ 1.13. Optional extras: `unzipy[fast]` (numba, faster
curve alignment), `unzipy[plot]` (matplotlib).

## Quick start

```python
import numpy as np
from unzipy.polymer import wlc_extension
from unzipy.fluctuation import bennett, bennett_error

x = wlc_extension(5.0, p=50.0)  # relative extension of a WLC at 5 pN
rng = np.random.default_rng(0)
w_f, w_r = (
    rng.normal(7, 2, 500),
    rng.normal(-3, 2, 500),
)  # work values (kT): dG = 5 kT
print(
    bennett(w_f, w_r), "+/-", bennett_error(w_f, w_r)
)  # free energy from forward + reverse work
```

Conventions: forces in **pN**, lengths in **nm**, energies in **kT** (kcal/mol where stated), temperature
25 °C unless passed. Work is the work done *on* the system. Molecules are always parametrised by their
stem length and loop length (or their sequence): nothing assumes a particular hairpin.

| subpackage | contents |
|---|---|
| `unzipy.polymer` | WLC, eWLC, FJC: force↔extension, energies, compliances |
| `unzipy.fluctuation` | Jarzynski (+ bias model and corrected estimator), Bennett (+ error), Crooks, matching, dissipated work; `position_ensemble` and `force_ensemble` pipelines |
| `unzipy.stats` | bootstrap, KDE with optimal bandwidth, outlier tests, Gaussian mixtures |
| `unzipy.fitting` | fits of force-extension data to elastic models (WLC, eWLC, FJC, eFJC, dipole, trap, series chains) in `x(f)` or `f(x)`, with uncertainties, bootstrap intervals and AIC/BIC model comparison |
| `unzipy.unzipping` | curve alignment, rupture rates, event detection, elastic fits and stretching model, equilibrium FDC, trace splitting |
| `unzipy.landscape` | nearest-neighbour stacks (SantaLucia 1998, Huguet 2010), sequential-opening landscape, mutant libraries |
| `unzipy.simulation` | two-state hairpin pulling simulation |

A command-line interface covers the common workflows (`unzipy --help`). See
[docs/guide.md](docs/guide.md) for workflows and [docs/methods.md](docs/methods.md) for the methods,
references and how each is validated; `examples/quickstart.py` is a runnable tour.

## Validation

Each module is tested against an independent reference wherever one exists: analytic limits and
numerical integrals (elasticity), exact Gaussian-work results (Jarzynski, Bennett), brute-force
simulation of the Jarzynski bias, Bell–Evans kinetics (rupture rates), equilibrium free energies from
a model landscape (simulation through Crooks/Bennett), MFOLD stacking energies, a quasi-static
force-ensemble protocol, and published numbers (Table 1 of J. Stat. Mech. (2019) 124001; the ensemble rate ratio of
Phys. Rev. E 98, 032146). See [docs/methods.md](docs/methods.md#validation) for the list and its limits.

## Contributing, license, citation

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Released under the
[MIT license](LICENSE). 

## Citing

If you use `unzipy` in your work, please cite it using [CITATION.cff](CITATION.cff)
(GitHub's "Cite this repository" button gives BibTeX and APA).
