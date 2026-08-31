# Planar biaxial test with rakes — SPINN identification

Notebook: `02_rakes_biaxial_identification.ipynb` · Runs: `results/biaxial_rakes/{NH,GOH}_rakes`

## Data

| | |
|---|---|
| Source | Abaqus, `projects/FIBER/Abaqus/*_rakes_ideal_Results`, converted by `src/phd/fem/abaqus_rakes.py` |
| Geometry | 7 × 7 mm membrane (M3D3/M3D4), thickness 1.8 mm, `nlgeom=YES` |
| Grips | 20 rigid tungsten rakes, 5 per side, through r = 0.15 mm holes at bottom y = 0.78, top y = 6.22, left x = 0.57, right x = 6.43 |
| Loading | **one equibiaxial ramp**, 0 → 0.5 mm per side (35 frames NH, 45 GOH); nominal λ ≈ 1.17–1.18 |
| Measurements | full nodal displacement field per frame + per-rake reaction forces |
| Ground truth | NH: C10 = 0.25 MPa · GOH: C10 = 0.013, k1 = 1.1 MPa, k2 = 3.7, κ = 0.24, α = ±29.98° |

## Assumptions

- **No Dirichlet boundary.** The outer edges are traction-free and load enters at 20 interior points, so the measured displacement field is used as a dense soft constraint in place of a hard BC (it also fixes rigid-body modes).
- **Section-force integrals set the stress level:** `H·∫P_xx dY = Right_Fx`, `H·∫P_yy dX = Top_Fy`, enforced on 5 parallel cuts between opposite rake rows. Verified against the Abaqus rake forces: top/bottom rakes carry a net |Fx| of 0.00066 N against 5.03 N from left/right, so **the balance is exact to 0.013 %**.
- **Hole masking.** Equilibrium, constitutive residual and displacement data are all zeroed within r = 0.15 mm of each rake hole, where there is no material and the solution is singular.
- Same incompressible plane-stress NH/GOH energies and mixed SPINN formulation as the ideal test; per-state scaling of stress, displacement and residuals.

## Results

| | truth | identified | error |
|---|---|---|---|
| **NH** C10 | 0.25 | 0.2290 | **8.4 %** |
| **GOH** C10 | 0.013 | 0.0271 | 109 % |
| **GOH** k1 | 1.10 | 1.922 | 75 % |
| **GOH** k2 | 3.70 | 0.015 | 100 % |
| **GOH** κ | 0.240 | 0.292 | 22 % |
| **GOH** α | 29.98° | 3.84° | 87 % |

Section forces: NH 0.49 % / 0.57 %, GOH 0.25 % / 0.37 % NRMSE.
Displacement field: NH 5.4 % / 7.9 %, GOH 7.5 % / 6.9 % NRMSE.

## Interpretation

**NH works** (8.4 % on C10), with section forces to ~0.5 %. The BC formulation is sound.

**GOH does not — treat those numbers as a failure, not a result.** α collapses to 3.8° and k2 to 0.015, which is a degenerate, nearly-isotropic material.

**The cause is the displacement fit, not the forces or the protocol.** Three pieces of evidence:

1. The section forces are already matched to 0.25–0.37 %, so the force constraint is satisfied — the parameters are wrong anyway.
2. An equibiaxial-only protocol *is* identifiable: from exact forces, 8/8 random least-squares starts recover the truth to 0.00 %. Its conditioning is worse than the full protocol (κ(J) = 9.8e4 vs 6.2e3), but under 0.25 % force noise — the error this run actually achieves — a direct fit still gives α to 1.1 % and k1 to 5.0 %. So neither the single ramp nor the force accuracy can explain an 87 % error in α.
3. What remains is the displacement field, fitted only to ~7 % NRMSE. The constitutive residual ties P to F(u); if u is 7 % off, F is off, and the parameters absorb the discrepancy.

**Why the field fit stalls.** The reference field is heterogeneous — 21 % deviation from the best-fit affine field — precisely because of 20 localised stress concentrations at the holes. That is exactly the kind of field a separable, low-rank tensor decomposition represents badly: SPINN approximates u(x,y,s) as a sum of rank-R products of 1-D functions, and 20 localised features are not low-rank in that basis.

**Next steps, in order of expected value:**
1. Raise SPINN rank/width, or restrict the domain to the gauge region away from the holes (the section-force cuts stay valid there).
2. Weight the displacement data more heavily relative to the residuals.
3. If neither closes the gap, this is a genuine limit of the separable architecture for this geometry and a standard (non-separable) PINN is the comparison worth making.
