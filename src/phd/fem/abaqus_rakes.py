"""
Convert Abaqus planar-biaxial-with-rakes results into a library dataset.

Run from the repository root:

    python src/phd/fem/abaqus_rakes.py --law nh
    python src/phd/fem/abaqus_rakes.py --law goh

Unlike src/phd/fem/biaxial_test.py this does not run a simulation; it reads the
CSVs produced by projects/FIBER/Abaqus/extraction_results_with_rakes_ideal.py
and repackages them the way phd.io expects. Plain numpy/scipy, so it runs in the
normal JAX environment -- no Abaqus and no FEniCS needed.

The model
---------
A 7 x 7 mm membrane sample (thickness 1.8 mm, plane stress, nlgeom) is gripped
by 20 rakes, five per side, each a rigid tungsten beam passing through a
circular hole of radius 0.15 mm in the sample:

    bottom  (1.5 .. 5.5, 0.78)      top    (1.5 .. 5.5, 6.22)
    left    (0.57, 1.5 .. 5.5)      right  (6.43, 1.5 .. 5.5)

The rakes are displaced outwards along a single equibiaxial ramp (35 frames,
0 -> 0.5 mm per side). Unlike the idealised test in biaxial_test.py, the
resulting deformation is *not* homogeneous -- it deviates from the best-fit
affine field by about 20% of the displacement range -- so the displacement field
carries information about the material parameters, not just the edge forces.

Ground truth (from the .inp files)
----------------------------------
NH   : C10 = 0.25 MPa
GOH  : C10 = 0.013 MPa, k1 = 1.1 MPa, k2 = 3.7, kappa = 0.24, alpha = +/-29.983 deg

Note the Abaqus ``*Anisotropic Hyperelastic, holzapfel`` orientation places the
two fibre families at +alpha and -alpha, which is the convention used by
phd.physics.hyperelasticity's "goh" law (and not the "+alpha twice" of the
reference projects/FIBER/GOH.py).
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import LinearNDInterpolator

# Geometry, in mm
SAMPLE_SIZE = 7.0
THICKNESS = 1.8
RAKE_RADIUS = 0.15

# Rake attachment points, read off the meshes (see module docstring)
RAKE_INSET = {"bottom": 0.78, "top": 6.22, "left": 0.57, "right": 6.43}
RAKE_SPACING = [1.5, 2.5, 3.5, 4.5, 5.5]

GROUND_TRUTH = {
    "nh": {"law": "nh", "C10": 0.25},
    "goh": {"law": "goh", "C10": 0.013, "k1": 1.1, "k2": 3.7,
            "kappa": 0.24, "alpha_deg": 29.983},
}


def rake_holes():
    """(20, 2) array of rake hole centres in the reference configuration."""
    pts = []
    for s in RAKE_SPACING:
        pts.append((s, RAKE_INSET["bottom"]))
        pts.append((s, RAKE_INSET["top"]))
        pts.append((RAKE_INSET["left"], s))
        pts.append((RAKE_INSET["right"], s))
    return np.array(pts)


def _read_initial_coords(results_dir):
    coords = {}
    with open(results_dir / "Initial_Coordinates.csv") as f:
        for row in csv.DictReader(f):
            if row["Instance"] == "SAMPLE-1":
                coords[int(row["NodeLabel"])] = (float(row["X0"]), float(row["Y0"]))
    node_ids = np.array(sorted(coords))
    return node_ids, np.array([coords[i] for i in node_ids])


def _read_frame(path, node_ids):
    disp = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["Instance"] == "SAMPLE-1":
                disp[int(row["NodeLabel"])] = (float(row["Ux"]), float(row["Uy"]))
    return np.array([disp[i] for i in node_ids])


def _read_forces(results_dir):
    """Per-frame (time, Bottom_Fy, Top_Fy, Left_Fx, Right_Fx)."""
    rows = []
    with open(results_dir / "Force_Summary.csv") as f:
        for row in csv.DictReader(f):
            rows.append((
                int(row["Frame"]), float(row["Time"]),
                float(row["Bottom_Fy"]), float(row["Top_Fy"]),
                float(row["Left_Fx"]), float(row["Right_Fx"]),
            ))
    rows.sort()
    return np.array(rows)


def _read_rake_motion(results_dir):
    """Applied rake displacement per frame, as (n_frames, 2) = (dx, dy) per side."""
    per_frame = defaultdict(dict)
    with open(results_dir / "Rake_Motion.csv") as f:
        for row in csv.DictReader(f):
            frame = int(row["Frame"])
            fam = row["Rake"].split("-")[1][0]     # B / T / L / R
            per_frame[frame][fam] = (float(row["Ux"]), float(row["Uy"]))

    motion = []
    for frame in sorted(per_frame):
        fams = per_frame[frame]
        # outward displacement magnitude per axis
        dx = 0.5 * (abs(fams.get("R", (0, 0))[0]) + abs(fams.get("L", (0, 0))[0]))
        dy = 0.5 * (abs(fams.get("T", (0, 0))[1]) + abs(fams.get("B", (0, 0))[1]))
        motion.append((dx, dy))
    return np.array(motion)


def nominal_stretches(motion):
    """
    Nominal stretch implied by the rake separation, used as the loading feature.

    Not the true stretch of the material -- the sample deforms non-uniformly and
    slips relative to the rakes -- but a smooth, monotonic parametrisation of the
    loading path, which is all the network needs on its loading axis.
    """
    gap_x = RAKE_INSET["right"] - RAKE_INSET["left"]
    gap_y = RAKE_INSET["top"] - RAKE_INSET["bottom"]
    return np.stack([1.0 + 2.0 * motion[:, 0] / gap_x,
                     1.0 + 2.0 * motion[:, 1] / gap_y], axis=1)


def interpolate_to_grid(coords, values, n_grid, size=SAMPLE_SIZE):
    """
    Interpolate an unstructured nodal field onto a regular n_grid x n_grid grid.

    Points inside a rake hole get NaN: there is no material there, and the
    surrounding solution is singular, so they must not be used as data.
    """
    lin = np.linspace(0.0, size, n_grid)
    X, Y = np.meshgrid(lin, lin, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel()])

    interp = LinearNDInterpolator(coords, values)
    out = interp(pts)

    holes = rake_holes()
    inside = np.zeros(len(pts), dtype=bool)
    for cx, cy in holes:
        inside |= np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < RAKE_RADIUS
    out[inside] = np.nan
    return pts, out, ~inside


def convert(law="nh", n_grid=100, abaqus_dir=None, output_dir=None):
    law = law.lower()
    if abaqus_dir is None:
        abaqus_dir = Path("projects/FIBER/Abaqus")
    abaqus_dir = Path(abaqus_dir)
    results_dir = abaqus_dir / f"{law.upper()}_rakes_ideal_Results"
    if not results_dir.exists():
        raise FileNotFoundError(f"Abaqus results not found: {results_dir}")

    node_ids, coords = _read_initial_coords(results_dir)
    forces = _read_forces(results_dir)
    motion = _read_rake_motion(results_dir)
    states = nominal_stretches(motion)

    frame_dir = results_dir / "Nodal_Displacements"
    frame_files = sorted(frame_dir.glob("Frame_*.csv"))
    print(f"{law}: {len(node_ids)} sample nodes, {len(frame_files)} frames")

    grid_pts = None
    U_grid, valid_mask = [], None
    for k, path in enumerate(frame_files):
        u_nodal = _read_frame(path, node_ids)
        pts, u_g, valid = interpolate_to_grid(coords, u_nodal, n_grid)
        grid_pts, valid_mask = pts, valid
        U_grid.append(u_g)
        if k % 10 == 0:
            print(f"  frame {k:3d}: |u|max = {np.nanmax(np.hypot(*u_g.T)):.4f} mm")
    U_grid = np.array(U_grid)

    # Align force rows with the frames that have displacement output
    n = min(len(frame_files), len(forces), len(states))
    forces, states, motion, U_grid = forces[:n], states[:n], motion[:n], U_grid[:n]

    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[1] / "io" / "dataset" / "biaxial_test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"rakes_ideal_7x7mm_{law}.npz"

    np.savez_compressed(
        out,
        coords=grid_pts,                    # (n_grid^2, 2) reference coords [mm]
        valid=valid_mask,                   # (n_grid^2,) False inside a rake hole
        states=states,                      # (n_frames, 2) nominal [lam11, lam22]
        motion=motion,                      # (n_frames, 2) applied rake motion [mm]
        u=U_grid,                           # (n_frames, n_grid^2, 2) displacement [mm]
        force=np.stack([forces[:, 5], forces[:, 3]], axis=1),   # (n_frames,2) [Right_Fx, Top_Fy]
        force_all=forces[:, 2:],            # Bottom_Fy, Top_Fy, Left_Fx, Right_Fx
        time=forces[:, 1],
        holes=rake_holes(),
        meta=json.dumps({
            "source": "abaqus_rakes_ideal",
            "law": law,
            "params": {k: v for k, v in GROUND_TRUTH[law].items() if k != "law"},
            "L": SAMPLE_SIZE, "H": THICKNESS,
            "rake_radius": RAKE_RADIUS, "rake_inset": RAKE_INSET,
            "n_grid": n_grid,
            "units": {"length": "mm", "force": "N", "stress": "MPa"},
        }),
    )
    print(f"Saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--law", default="nh", choices=["nh", "goh"])
    ap.add_argument("--n-grid", type=int, default=100)
    ap.add_argument("--abaqus-dir", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    convert(law=args.law, n_grid=args.n_grid,
            abaqus_dir=args.abaqus_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
