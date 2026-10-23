# Methods, references and validation

Units: pN, nm, kT unless stated. `W` is the work done on the system; for the forward process
`<W_F> >= dG >= -<W_R>`.

## Polymer elasticity — `unzipy.polymer`

| model | relation | notes |
|---|---|---|
| WLC | Marko–Siggia interpolation `f = kT/(4P) [(1-x/L)^-2 - 1 + 4x/L]` | inverted in closed form (cubic in `1 - x/L`, trigonometric/Cardano branch, Newton-polished); optional Bouchiat polynomial correction (Bouchiat et al., Biophys. J. 76, 409 (1999)) |
| eWLC | `Lc -> Lc (1 + f/Y)` (`ewlc_scaled_*`) or the additive `l + f/S` (`ewlc_*`) | |
| FJC | `x = L (coth z - 1/z)`, `z = f b / kT` | series at small `z`, log-sinh form at large `z`; even/odd in `f` |

Energies come in two flavours: `*_energy` is `∫ f dx / kT` (position ensemble) and `*_gibbs` is
`-∫ x df / kT` (force ensemble); they differ by `f x / kT`.

## Free energies from work — `unzipy.fluctuation`

- **Jarzynski** (PRL 78, 2690 (1997)) via log-sum-exp; the reverse estimator is `ln<exp(-W_R)>`.
- **Bennett acceptance ratio** (Bennett 1976) solved with Brent's method; asymptotic error from
  Shirts et al., PRL 91, 140601 (2003).
- **Crooks** (PRE 60, 2721 (1999)): `ln[P_F(W)/P_R(-W)] = W - dG`, estimated with shared bin edges and
  bootstrap errors; `crooks_crossing` fits the slope (theory: 1) and the crossing point.
- **Matching** for strongly irreversible pulling: Gaussian continuation of the two distributions
  (cf. the matching method in Severino et al.).
- **Jarzynski bias** (`bias`, `jarzynski_corrected`): Palassini & Ritort, Phys. Rev. Lett. 107, 060601 (2011). Eqs. 3–6 are implemented as published; for `λ = (δ-1) ln N / μ` above 1.3 the
  closed expression lives in the paper's supplement and is *not* implemented — `bias()` falls back to
  the empirical Gaussian power law (eq. 7) and never returns a negative bias.
- **Dissipated work** of a two-state system at constant loading rate (eq. A9 of Phys. Rev. E 98,
  032146 (2018)) and the force/extension-ensemble rate ratio `Ω` (its eq. 7).

### Position-controlled pulling — `fluctuation.position_ensemble`

Following Severino, Monge, Rissone, Ritort (J. Stat. Mech. (2019) 124001):
`dG0 = dG - dW_m - dW_b - dW_h`, where `dW_m` is the WLC energy of the released strand at `f_max`
minus the dipole (FJC) energy of the folded hairpin at `f_min`, and `dW_b + dW_h` is the work to
stretch bead and handles. For short handles / locally linear force–distance curves this is the
**effective-stiffness** estimate `(f_max² - f_min²) / (2 k_eff)`; otherwise use
`unzipping.fit_stretching_model` (their section 5: extensible-WLC handles, trap stiffness
`k_b0 + α f`, both branches fitted). The released strand has `2 n_bp + loop_nt` nucleotides.

### Force-controlled pulling — `fluctuation.force_ensemble`

Work is `W = -∫ x df / kT` (Monge, Manosas, Ritort, PRE 98, 032146). Extensions are measured
relative to the folded branch at `f_min`; the model restores the reference before subtracting the
Gibbs energies of strand, handles and dipole.

## Statistics — `unzipy.stats`

Uniform bootstrap with chunked resampling; bootstrap densities on fixed bin edges; Gaussian KDE with
the Shimazaki–Shinomoto bandwidth (J. Comput. Neurosci. 29, 171 (2010)); Grubbs and IQR outlier
tests; Gaussian mixtures by EM (histogram-weighted EM with Sheppard's correction for very large
samples).

## Unzipping — `unzipy.unzipping`

- **Alignment**: force levels discretised into letters and aligned globally with Needleman–Wunsch
  (J. Mol. Biol. 48, 443 (1970)), score `10 - (a-b)²/10`, linear gap penalty; the extension shift of
  matched points is smoothed with a spline. Optional last-rip step.
- **Rupture rates**: `k(F) = ∓ r (dS/dF)/S` from a bootstrapped KDE of rupture forces
  (Bell–Evans framework, Evans & Ritchie, Biophys. J. 72, 1541 (1997)).
- **Equilibrium FDC** from pair energies: states `n = 0..N` open pairs, trap in series with two
  released strands; the mean force `<f> = Σ p_n f_n` equals `dF/dλ` exactly.
- Event detection, effective stiffness, ssDNA elastic fits, trace splitting.

## Landscape and simulation — `unzipy.landscape`, `unzipy.simulation`

Nearest-neighbour stacking free energies `dG = dH - T dS - m ln[Na+]` for the SantaLucia (PNAS 95,
1460 (1998)) and Huguet et al. (PNAS 107, 15431 (2010)) parameter sets. **Hairpin-loop energies are
inputs** (kcal/mol): loops, mismatches and bulges need a structure-prediction tool (e.g. MFOLD); read
its per-state output with `io.read_mfold_input`.

`HairpinSim` simulates pulling with two-state kinetics on the full landscape: for each trap position
the energies of all intermediate states are computed once, switching probability
`p = k0 exp(-B) [(1-σ) + σ exp(G_N)] dt` with `B = ln Σ_i Σ_{k≤i} exp(G_i - G_k)` (detailed balance),
and measurement noise from the trap–molecule series stiffness.

## Validation

| component | reference |
|---|---|
| WLC/FJC | analytic limits (low/high force), force↔extension round trips, energies = numerical integrals of `f dx` and `-x df`, finite-difference compliances, finite for forces up to 10⁸ pN |
| Jarzynski / Bennett | exact results for Gaussian work (`dG = μ - σ²/2`), BAR error calibrated against replicate scatter |
| Jarzynski bias | brute-force simulation at `λ` from 0.17 to 3.5; the paper's `ln 2` at `λ = 1` |
| Crooks, bootstrap | Gaussian work (slope 1, crossing at `dG`); uniformity of resampling; normalised densities |
| force ensemble | independent numerical construction of a quasi-static two-state protocol |
| position ensemble | Table 1 of J. Stat. Mech. (2019) 124001 (`dW_m` 17.3 vs 17.4 kT, `dW_h + dW_b` 224 vs 225 kT); *the zero-force bookkeeping is otherwise only checked for self-consistency* |
| dissipated work | small-rate limit and the published `Ω = 0.55` example; the full integral is not independently checked |
| rupture rates | Bell–Evans samples (unfolding and refolding directions) |
| equilibrium FDC | `<f> = dF/dλ` by finite differences |
| landscape | MFOLD stacking energies to within 0.015 kcal/mol |
| simulation | equilibrium free energy between two trap positions recovered by Bennett from simulated forward and reverse work |

## Not implemented

Structure prediction (MFOLD-like folding with mismatches and loops); locally adaptive KDE bandwidth;
the supplement's `λ > 1` Jarzynski-bias expression; hairpin-loop free-energy tables.
