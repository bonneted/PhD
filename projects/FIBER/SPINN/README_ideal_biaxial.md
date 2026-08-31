# Ideal planar biaxial test — SPINN identification

Notebook: `01_ideal_biaxial_identification.ipynb` · Runs: `results/biaxial_test/{NH,GOH}_inverse`

## Data

| | |
|---|---|
| Source | FEniCS 2019, `src/phd/fem/biaxial_test.py` |
| Geometry | 7 × 7 mm square, thickness 1 mm, plane stress |
| Protocol | 16 states — ratios 1:1, 0.5:1, 1:0.5, "custom" × 4 stretch levels (thesis Table 3.1), λ up to 1.36 |
| Measurements | displacement field (10×10 per state) + the two edge forces |
| Ground truth | NH: C10 = 0.4 MPa · GOH: C10 = 0.019, k1 = 5.15 MPa, k2 = 8.64, κ = 0.24, α = 38.8° |

## Assumptions

- **Incompressible plane stress.** λ₃ = 1/det F₂ substituted into Ψ, so σ₃₃ = 0 follows by construction and no pressure unknown is needed.
- **Affine displacement on the whole boundary** → the deformation is *homogeneous*. Consequence: the displacement field is identical whatever the parameters are, so it carries **no** parameter information. The edge force is the only informative observable.
- **GOH fibres at ±α** (Gasser et al. 2006, thesis Eq. 3.10–3.11, and the Abaqus `holzapfel` orientation). The reference `projects/FIBER/GOH.py` puts both families at +α, which is equivalent to one family with doubled k1; reproduce it with the `goh_ref` law.
- SPINN, mixed formulation (6 outputs `[Ux, Uy, Pxx, Pxy, Pyx, Pyy]`), loading state as a third network input.
- Stress output and residuals scaled **per loading state** from that state's measured force.

## Results

| | truth | identified | error |
|---|---|---|---|
| **NH** C10 | 0.4 | 0.4004 | **0.10 %** |
| **GOH** k1 | 5.15 | 4.918 | 4.5 % |
| **GOH** k2 | 8.64 | 8.344 | 3.4 % |
| **GOH** κ | 0.240 | 0.2336 | 2.7 % |
| **GOH** α | 38.8° | 39.17° | 0.9 % |
| **GOH** C10 | 0.019 | 0.0055 | **71 %** |

Force reproduction: NH 0.58 % / 0.95 %, GOH 0.75 % / 0.59 %. GOH MPE = 16.5 %.

## Interpretation

**NH is solved.** 0.1 % on C10 with sub-percent forces validates the whole chain — energy formulation, force operator, FEniCS reference and JAX model all agree (cross-checked to 1e-6).

**GOH: four of five parameters are good; C10 is not, and that is the experiment, not the fit.**
The sensitivity of the edge force to C10 is ~1 N against 1442 N for κ and 635 N for k2 — three orders of magnitude weaker. At these stretch levels the fibre term is already active at λ = 1.03, so the matrix term never dominates anywhere in the protocol. Refitting under 1 % force noise gives C10 a 65 % coefficient of variation while the others stay under 12 %. The thesis sees the same thing: Table 3.6 reports a fitted C10 of 0.0727 against 0.019 truth (283 % error) with the rest within a few percent. **Our 71 % is better than the thesis's classic fit.**

**The data is not the limitation.** A direct least-squares fit through the constitutive model recovers all five parameters exactly (10/10 random starts, MPE 0.00 %). What is hard is the coupled PINN problem, where parameters only feel the force data through the network's stress output.

**What made GOH work was per-state scaling.** The fibre term is exponential, so protocol stresses span 290×. With a single global output scale the low-stretch states need network outputs near 3e-3 — three orders below its natural range, where no relative accuracy exists. Scaling per state moved MPE from 47 % → 16.5 % and force error from ~10 % → 0.3 %. NH spans only 5.3× and is insensitive to the choice, which is why the problem stayed hidden.

**Caveat.** The idealised BC removes exactly the boundary-layer inhomogeneity the thesis identifies as the main error source in a real rake test. See the rakes notebook for that case.
