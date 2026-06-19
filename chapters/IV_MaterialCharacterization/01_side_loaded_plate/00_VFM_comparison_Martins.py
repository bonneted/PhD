"""VFM identification on the exact same data as the SPINN Martins comparison.

This script applies the Virtual Fields Method to the side-loaded plate
benchmark of Martins et al. (2018) using the *converged* FEM reference
solution (100x100 elements, 3x3 mm domain) and the exact same measurement
setup as the SPINN run in `01_comparison_Martins.ipynb` /
`results/side_loaded_plate/comparison_Martins`:

- 4 x 4 grid of strain values sampled from the FEM solution,
- i.i.d. Gaussian noise of standard deviation 1e-6 (1 micro-strain) added
  to each strain component,
- 10 noise realizations.

Two constant virtual strain fields are used, as in `00_VFM.ipynb`:

1. u* = (x, 0)  ->  eps* = (1, 0, 0): Q11 mean(exx) L^2 + Q12 mean(eyy) L^2 = F L
2. u* = (0, y)  ->  eps* = (0, 1, 0): Q11 mean(eyy) L^2 + Q12 mean(exx) L^2 = 0

The identified stiffness components (Q11, Q12) are converted to (E, nu).

Purpose: show that, when supplied with the same converged reference data and
the same sparse 4x4 sampling, the VFM error is dominated by the spatial
undersampling of the integrals and the strain noise, providing a fairer
context for Table 4.1 of the thesis (the published values of Martins et al.
were obtained with an unconverged FE model and cannot be replicated exactly).
"""

from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phd.io import load_side_loaded_plate_reference_raw

# --- Problem definition (Martins et al. benchmark, converged reference) ---
FEM_DATASET = "3x3mm.dat"
L = 3.0      # mm
m = 10.0     # N/mm
b = 50.0     # N
F = m * L**2 / 2 + b * L  # resultant of the linear traction profile

E_REF = 210.0e3  # MPa
NU_REF = 0.3

N_GRID = 4              # 4 x 4 measurement grid (as in Martins et al.)
NOISE_STD = 1e-6        # 1 micro-strain Gaussian noise on strains
N_TRIALS = 10
SEED = 0

# --- Load converged FEM reference and build strain interpolators ---
raw = load_side_loaded_plate_reference_raw(FEM_DATASET)
X_val = raw[:, :2]
strain_val = raw[:, 4:7]

n_mesh = int(np.sqrt(X_val.shape[0]))
x_grid = np.linspace(0, L, n_mesh)
y_grid = np.linspace(0, L, n_mesh)

interps = [
    RegularGridInterpolator(
        (x_grid, y_grid), strain_val[:, i].reshape(n_mesh, n_mesh).T
    )
    for i in range(3)
]


def strain_at(points):
    return np.array([itp((points[:, 0], points[:, 1])) for itp in interps]).T


def vfm_identify(exx, eyy):
    """VFM with two constant virtual strain fields -> (Q11, Q12, E, nu)."""
    A = np.array(
        [
            [np.mean(exx) * L**2, np.mean(eyy) * L**2],
            [np.mean(eyy) * L**2, np.mean(exx) * L**2],
        ]
    )
    B = np.array([F * L, 0.0])
    Q11, Q12 = np.linalg.solve(A, B)

    mu = (Q11 - Q12) / 2
    lam = Q12
    nu = lam / (2 * (mu + lam))
    E = mu * (3 * lam + 2 * mu) / (lam + mu)
    return Q11, Q12, E, nu


# --- Measurement grid (same as SPINN run: 4x4 over the full domain) ---
xs = np.linspace(0, L, N_GRID)
Xg, Yg = np.meshgrid(xs, xs)
pts = np.column_stack([Xg.ravel(), Yg.ravel()])
eps_clean = strain_at(pts)

# Sanity check without noise: quantifies the pure 4x4 sampling bias
Q11_0, Q12_0, E_0, nu_0 = vfm_identify(eps_clean[:, 0], eps_clean[:, 1])
print("--- No noise (pure 4x4 spatial-sampling bias) ---")
print(f"E  = {E_0 / 1e3:8.2f} GPa  (error {abs(E_0 - E_REF) / E_REF * 100:6.2f} %)")
print(f"nu = {nu_0:8.4f}      (error {abs(nu_0 - NU_REF) / NU_REF * 100:6.2f} %)")

# Full-field check (mean over all FEM points, no noise): should be ~exact
Q11_f, Q12_f, E_f, nu_f = vfm_identify(strain_val[:, 0], strain_val[:, 1])
print("--- No noise, full-field integration (sanity) ---")
print(f"E  = {E_f / 1e3:8.2f} GPa  (error {abs(E_f - E_REF) / E_REF * 100:6.2f} %)")
print(f"nu = {nu_f:8.4f}      (error {abs(nu_f - NU_REF) / NU_REF * 100:6.2f} %)")

# --- Noise trials (same noise model as the SPINN comparison run) ---
rng = np.random.default_rng(SEED)
E_list, nu_list = [], []
for _ in range(N_TRIALS):
    eps_noisy = eps_clean + rng.normal(0, NOISE_STD, eps_clean.shape)
    _, _, E_id, nu_id = vfm_identify(eps_noisy[:, 0], eps_noisy[:, 1])
    E_list.append(E_id)
    nu_list.append(nu_id)

E_arr = np.array(E_list) / 1e3  # GPa
nu_arr = np.array(nu_list)
E_err = np.abs(E_arr - E_REF / 1e3) / (E_REF / 1e3) * 100
nu_err = np.abs(nu_arr - NU_REF) / NU_REF * 100

print(f"--- {N_TRIALS} noise realizations (sigma = {NOISE_STD:g} on strains) ---")
print(f"E  = {E_arr.mean():8.2f} +/- {E_arr.std():.2f} GPa "
      f"(error {E_err.mean():.2f} +/- {E_err.std():.2f} %)")
print(f"nu = {nu_arr.mean():8.4f} +/- {nu_arr.std():.4f}  "
      f"(error {nu_err.mean():.2f} +/- {nu_err.std():.2f} %)")
i_worst = int(np.argmax(E_err + nu_err))
print(f"Worst run: E = {E_arr[i_worst]:.2f} GPa ({E_err[i_worst]:.2f} %), "
      f"nu = {nu_arr[i_worst]:.4f} ({nu_err[i_worst]:.2f} %)")
