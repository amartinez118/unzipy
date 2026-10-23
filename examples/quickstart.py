"""Quick tour of unzipy.

Shows polymer elasticity, free energy from work values, a coexistence force
from a sequence, and a simulated pulling experiment.
"""

import numpy as np

from unzipy.fluctuation import bennett, bennett_error, jarzynski
from unzipy.landscape import coexistence_force, kcal_to_kT, stack_energies
from unzipy.polymer import wlc_extension
from unzipy.simulation import HairpinSim

# 1. Elasticity: relative extension of a 50 nm persistence-length chain at 5 pN
print("x/L at 5 pN:", float(wlc_extension(5.0, p=50.0)))

# 2. Free energy from work values (kT): forward and reverse works of a
#    Gaussian toy process
rng = np.random.default_rng(0)
w_f, w_r = (
    rng.normal(7, 2, 500),
    rng.normal(-3, 2, 500),
)  # Delta G = 5 kT, <W_diss> = 2 kT
print(
    "Jarzynski:",
    float(jarzynski(w_f)),
    " Bennett:",
    bennett(w_f, w_r),
    "+/-",
    bennett_error(w_f, w_r),
)

# 3. Sequence -> coexistence force of a hairpin (SantaLucia stacks, 25 C)
stacks = stack_energies("GCGAGCCATAATTTCATCTG")
dg0 = -(stacks.sum() + 1.14) * kcal_to_kT()
fc = coexistence_force(dg0, stacks.size + 1)
print(f"dG0 = {dg0:.1f} kT, coexistence force = {fc:.1f} pN")

# 4. Simulated pulling experiment (two-state kinetics on the full landscape)
sim = HairpinSim(tuple(np.append(stacks, 1.14)), nloop=4)
pulls = sim.simulate(20, "forward", rng=1)
print(
    "mean rupture force: {:.1f} pN".format(
        np.mean([t["f"][np.argmax(t["sigma"] == 1)] for t in pulls])
    )
)
