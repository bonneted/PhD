"""
Ideal planar biaxial test of a soft-tissue sample - FEniCS reference solution.

Run with the legacy FEniCS environment, from the repository root:

    conda run -n fenics python src/phd/fem/biaxial_test.py --law goh
    conda run -n fenics python src/phd/fem/biaxial_test.py --law nh

Problem
-------
A square sample of side ``L`` (7 x 7 mm, as in Fehervary's thesis) and
thickness ``H`` is stretched biaxially. In this idealised version the full
boundary is displacement-controlled with the affine field

    u = ((lambda11 - 1) X, (lambda22 - 1) Y)

so the reference solution is a homogeneous deformation. That is deliberate: it
is the limit case a rake-based test approximates, and it isolates the parameter
identification problem from the boundary-layer effects studied in the thesis.
Because the deformation is homogeneous, the displacement field alone carries no
information about the material parameters -- the identifiable observable is the
edge force, which is what a real biaxial rig measures.

A loading *protocol* of several (lambda11, lambda22) states is solved (Table 3.1
of the thesis: ratios 1:1, 0.5:1, 1:0.5 and 'custom', at four stretch levels),
because a single state gives only two force values and cannot identify the five
GOH parameters.

Material
--------
Neo-Hookean and Gasser-Ogden-Holzapfel, incompressible, reduced to plane stress
by substituting lambda_3 = 1/det(F2). Identical formulation to
``phd.physics.hyperelasticity`` -- see that module for the credit note on the
original KU Leuven implementation (H. Fehervary, J. Vastmans, 2020).

Units: mm, N, MPa.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from fenics import (
    Constant, DirichletBC, Expression, Function, FunctionSpace, Identity,
    Measure, MeshFunction, Point, RectangleMesh, TensorFunctionSpace,
    TestFunction, TrialFunction, VectorFunctionSpace, as_vector, assemble,
    conditional, derivative, det, diff, dot, dx, exp, grad, gt, inner,
    project, solve, tr, variable,
)


# --- Loading protocol (thesis Table 3.1) -------------------------------------
PROTOCOL = {
    "1:1":    [(1.06, 1.06), (1.12, 1.12), (1.24, 1.24), (1.36, 1.36)],
    "0.5:1":  [(1.03, 1.06), (1.06, 1.12), (1.12, 1.24), (1.18, 1.36)],
    "1:0.5":  [(1.06, 1.03), (1.12, 1.06), (1.24, 1.12), (1.36, 1.18)],
    "custom": [(1.06, 1.08), (1.12, 1.08), (1.24, 1.08), (1.36, 1.08)],
}

# --- Default material parameters (thesis Tables 3.2 and 3.3) -----------------
DEFAULT_PARAMS = {
    # GOH parameter set 1
    "goh": {"C10": 0.019, "k1": 5.15, "k2": 8.64, "kappa": 0.24, "alpha_deg": 38.8},
    # NH parameter set 1
    "nh": {"C10": 0.4},
}


def strain_energy(F2, law, params):
    """
    Plane-stress incompressible strain energy density [MPa].

    ``F2`` must be a UFL ``variable`` so that ``diff(psi, F2)`` gives the first
    Piola-Kirchhoff stress.
    """
    C2 = F2.T * F2
    lam3 = 1.0 / det(F2)
    I1 = tr(C2) + lam3 ** 2

    psi = Constant(params["C10"]) * (I1 - 3.0)

    if law == "goh":
        k1 = Constant(params["k1"])
        k2 = Constant(params["k2"])
        kappa = Constant(params["kappa"])
        alpha = np.deg2rad(params["alpha_deg"])
        # Two fibre families at +/- alpha from the 1-direction (see the note in
        # phd.physics.hyperelasticity on the reference implementation).
        for sign in (+1.0, -1.0):
            M = as_vector([np.cos(alpha), sign * np.sin(alpha)])
            I4 = dot(M, C2 * M)
            E = kappa * I1 + (1.0 - 3.0 * kappa) * I4 - 1.0
            E = conditional(gt(E, 0.0), E, 0.0)  # fibres carry tension only
            psi = psi + k1 / (2.0 * k2) * (exp(k2 * E ** 2) - 1.0)

    return psi


def solve_state(mesh, V, law, params, lam11, lam22, H, L, u_init=None, n_steps=4):
    """
    Solve one (lambda11, lambda22) state; returns (u, P_expr, F1, F2).

    A simple load-continuation in ``n_steps`` increments keeps Newton inside its
    basin for the stiff exponential of the GOH model at high stretch.
    """
    u = Function(V)
    if u_init is not None:
        u.assign(u_init)
    v = TestFunction(V)
    du = TrialFunction(V)

    F2 = variable(Identity(2) + grad(u))
    psi = strain_energy(F2, law, params)
    P = diff(psi, F2)

    # Total potential energy of the sheet (thickness H, no external work: the
    # whole boundary is displacement controlled).
    Pi = Constant(H) * psi * dx
    residual = derivative(Pi, u, v)
    jacobian = derivative(residual, u, du)

    bc_expr = Expression(
        ("a*x[0]", "b*x[1]"), a=0.0, b=0.0, degree=1
    )
    bc = DirichletBC(V, bc_expr, "on_boundary")

    for step in range(1, n_steps + 1):
        t = step / n_steps
        bc_expr.a = t * (lam11 - 1.0)
        bc_expr.b = t * (lam22 - 1.0)
        solve(
            residual == 0, u, bc, J=jacobian,
            solver_parameters={
                "newton_solver": {
                    "relative_tolerance": 1e-10,
                    "absolute_tolerance": 1e-12,
                    "maximum_iterations": 50,
                    "linear_solver": "mumps",
                }
            },
        )

    # Edge forces in the reference configuration: F_i = H * int P[i,i] dS.
    # The traction on X = L is P.N with N = e_1, so its 1-component is P[0,0].
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    class _Right(object):
        pass

    from fenics import AutoSubDomain, near
    AutoSubDomain(lambda x, on_b: on_b and near(x[0], L, 1e-10)).mark(boundaries, 1)
    AutoSubDomain(lambda x, on_b: on_b and near(x[1], L, 1e-10)).mark(boundaries, 2)
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    force_1 = assemble(Constant(H) * P[0, 0] * ds(1))
    force_2 = assemble(Constant(H) * P[1, 1] * ds(2))

    return u, P, force_1, force_2


def sample_on_grid(u, P, mesh, L, n_grid):
    """Sample displacement and 1st PK stress on a regular n_grid x n_grid grid."""
    W = TensorFunctionSpace(mesh, "P", 1, shape=(2, 2))
    P_fun = project(P, W)

    lin = np.linspace(0.0, L, n_grid)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    # Nudge points off the boundary: FEniCS point evaluation is fragile there.
    eps = 1e-9 * L
    pts_eval = np.clip(pts, eps, L - eps)

    u_val = np.array([u(p) for p in pts_eval])
    P_val = np.array([P_fun(p) for p in pts_eval])
    return pts, u_val, P_val


def run(law="goh", params=None, L=7.0, H=1.0, n_mesh=20, n_grid=100,
        output_dir=None, protocol=None):
    law = law.lower()
    params = dict(DEFAULT_PARAMS[law] if params is None else params)
    protocol = PROTOCOL if protocol is None else protocol

    mesh = RectangleMesh(Point(0.0, 0.0), Point(L, L), n_mesh, n_mesh)
    V = VectorFunctionSpace(mesh, "P", 2)

    states, coords, U, PK, forces = [], None, [], [], []

    for ratio, levels in protocol.items():
        u_prev = None
        for level, (lam11, lam22) in enumerate(levels, start=1):
            u, P, f1, f2 = solve_state(mesh, V, law, params, lam11, lam22, H, L, u_init=u_prev)
            u_prev = u
            pts, u_val, P_val = sample_on_grid(u, P, mesh, L, n_grid)
            coords = pts

            # Homogeneity check: the ideal test must give a uniform stress field.
            # Uniformity of the stress field, normalised by the overall stress
            # magnitude (per-component normalisation would blow up on the
            # near-zero shear components).
            inhomogeneity = float(np.max(np.std(P_val, axis=0)) / (np.max(np.abs(P_val)) + 1e-12))

            states.append((lam11, lam22))
            U.append(u_val)
            PK.append(P_val)
            forces.append((f1, f2))
            print(
                f"{ratio:>7s} level {level}  lam = ({lam11:.2f}, {lam22:.2f})  "
                f"F = ({f1: .4f}, {f2: .4f}) N   Pxx = {np.mean(P_val[:, 0]): .5f} MPa   "
                f"inhomogeneity = {inhomogeneity:.2e}"
            )

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "io" / "dataset" / "biaxial_test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = f"ideal_{int(L)}x{int(L)}mm_{law}"
    out = output_dir / f"{name}.npz"
    np.savez_compressed(
        out,
        coords=coords.astype(np.float64),          # (n_grid^2, 2) reference coordinates [mm]
        states=np.array(states),                   # (n_states, 2) [lambda11, lambda22]
        u=np.array(U),                             # (n_states, n_grid^2, 2) [mm]
        P=np.array(PK),                            # (n_states, n_grid^2, 4) [Pxx,Pxy,Pyx,Pyy] MPa
        force=np.array(forces),                    # (n_states, 2) edge forces [N]
        meta=json.dumps({
            "law": law, "params": params, "L": L, "H": H,
            "n_mesh": n_mesh, "n_grid": n_grid,
            "protocol": {k: [list(s) for s in v] for k, v in protocol.items()},
            "field_order": ["Pxx", "Pxy", "Pyx", "Pyy"],
            "units": {"length": "mm", "force": "N", "stress": "MPa"},
        }),
    )
    print(f"\nSaved FEM reference data to: {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--law", default="goh", choices=["goh", "nh"])
    ap.add_argument("--L", type=float, default=7.0, help="sample side [mm]")
    ap.add_argument("--H", type=float, default=1.0, help="sample thickness [mm]")
    ap.add_argument("--n-mesh", type=int, default=20)
    ap.add_argument("--n-grid", type=int, default=100)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    run(law=args.law, L=args.L, H=args.H, n_mesh=args.n_mesh,
        n_grid=args.n_grid, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
