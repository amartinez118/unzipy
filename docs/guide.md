# unzipy guide

Conventions: forces in pN, lengths in nm, energies in kT (kcal/mol where stated),
temperature 25 °C unless passed. A *work* is the work done **on** the system, so for the forward
(unfolding) process `<W_F> >= dG >= -<W_R>` where `W_R` is the (negative) reverse work.

## Molecule size is always a parameter

Nothing assumes a 20-bp hairpin. A hairpin is described by the number of stem base pairs
`n_bp` and the loop length `loop_nt`; it releases `2 * n_bp + loop_nt` nucleotides
(a 44-nt hairpin = 20 bp + 4-nt loop). Every model can be built from the sequence:

```python
from unzipy.fluctuation import ForceModel, PullingModel
from unzipy.simulation import HairpinSim
from unzipy.landscape import landscape_from_sequence

seq = (
    "GCGAGCCATAATTTCATCTG" + "TTTT" + "CAGATGAAATTATGGCTCGC"
)  # any length: 2*stem + loop
PullingModel.from_sequence(seq, loop_nt=4, ss_p=1.35, ss_d0=0.59)
ForceModel.from_sequence(seq, loop_nt=4)
HairpinSim.from_sequence(
    seq, loop_nt=4, loop_dg=1.14
)  # loop free energy (kcal/mol): an input
n, G = landscape_from_sequence(seq, force=12.0, loop_dg=1.14)
```

The construct has **two handles**; their energies are counted twice (`n_handles=2`).

## Free energy from position-controlled pulling (optical tweezers, trap position control)

1. `position_ensemble.works_from_directory(dir, x_range)` integrates `u<N>.txt` (unfolding) / `f<N>.txt` (refolding)
   curves (`W = int f dx / kT`).
2. `estimate_free_energy(w_f, w_r)`: Jarzynski, Bennett (+ error), dissipated work.
3. `position_ensemble.zero_force_free_energy(...)` subtracts the elastic contributions (Severino et al., J. Stat. Mech. (2019) 124001):
   `dG0 = dG - dW_m - dW_b - dW_h`, with the effective-stiffness estimate
   `dW_b + dW_h = (f_max^2 - f_min^2) / (2 k_eff)` for short handles (`handles="spring"`) or an explicit
   handle WLC (`handles="wlc"`). For strongly irreversible pulling or force-dependent stiffness fit the
   curves with `unzipping.fit_stretching_model` and use `StretchModel.zero_force_free_energy`.
   Use fitted-branch forces (`branch_force`) for `f_min`, `f_max` rather than noisy measured extremes.

## Force-controlled pulling (magnetic tweezers, force feedback)

`fluctuation.force_ensemble` (same function names as `position_ensemble`): work is `W = -int x df / kT` (Monge, Manosas, Ritort, PRE 98, 032146);
extensions are referenced to the folded branch at `f_min` and the reference is restored with the model.

## Fitting elastic models

```python
from unzipy.fitting import Model, fit, ss_wlc, ewlc, trap

model = Model(ewlc(label="handles"), ss_wlc(), trap())
report = fit(
    model,
    f,
    x,  # force (pN), extension (nm)
    start={"handles.L": 250.0, "ss.n": 40.0, "trap.k0": 0.07},
    fixed={
        "handles.p": 20.0,
        "handles.Y": 200.0,
        "ss.d0": 0.665,
        "ss.p": 0.75,
    },
)
print(report.summary())
ci = report.bootstrap(n_boot=500, rng=0)["ci"]
```

Contour lengths, the offset `shift` and the nucleotide count are strongly
correlated in series chains: check `report.correlation` and fix what you know
(or fit `shift` only when the extension has an arbitrary zero).

Use `variable="f(x)"` for position-controlled data, `free=["trap.alpha"]` to
fit a force-dependent trap stiffness, and `compare(wlc=fit_a, fjc=fit_b)` to
rank models by AIC/BIC.

## Unzipping

`unzipping`: `split_cycles` (raw trace -> trajectories), `align_curves`, `equilibrium_fdc`
(theoretical force-distance curve from pair energies), `rupture_rates` (Bell-Evans k(F)), `find_event`
(rip detection), `fit_ssdna_wlc`, `effective_stiffness`.

## Landscape and simulation

`landscape`: SantaLucia/Huguet nearest-neighbour stacks (`stack_energies(stem, T, na, params)`),
`sequential_landscape`, `coexistence_force`. Hairpin-loop energies are inputs (kcal/mol).
`simulation.HairpinSim` simulates pulling of any hairpin; for structures with interior loops
read MFOLD-derived energies with `io.read_mfold_input` and pass `gbp`, `nt_open`.

## Command line

```
unzipy neqwork-position DIR --x-range X0 X1 --sequence SEQ --loop-nt 4 --preset dna \
                        [--k-eff K --f-min F0 --f-max F1]
unzipy neqwork-force DIR --f-range F0 F1 --n-bp 20 --loop-nt 4 --preset dna [--n-handles 2 ...]
unzipy split TRACE.txt --f-min 5 --f-max 20 -o events_raw
unzipy simulate SEQ --loop-nt 4 --loop-dg 1.14 -n 100 -o sim

The molecule is always explicit (`--sequence` or `--n-bp`, `--loop-nt`, and `--preset dna|rna` or
`--ss-p/--ss-d0`); `--k-eff` requires `--f-min/--f-max` from the branch fits.
```
