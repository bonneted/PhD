from pathlib import Path

import numpy as np
import pandas as pd


def _dataset_root() -> Path:
    return Path(__file__).parent


def get_side_loaded_plate_dataset_path(filename: str) -> Path:
    filepath = _dataset_root() / "side_loaded_plate" / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Side-loaded plate dataset '{filename}' not found at {filepath}")
    return filepath


def load_side_loaded_plate_reference_raw(filename: str) -> np.ndarray:
    filepath = get_side_loaded_plate_dataset_path(filename)
    return np.loadtxt(filepath)


def _read_dic_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"DIC file not found: {path}")
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    return df.dropna(axis=1).to_numpy()


def load_side_loaded_plate_dic_sample(dic_path: str, sample_id: int, measurement_type: str) -> dict:
    """
    Load one DIC sample using the legacy side_loaded_plate folder convention.

    Args:
        dic_path: Path relative to io/dataset root (e.g. "side_loaded_plate/dic/0.4MP/0noise")
                  or absolute path.
        sample_id: Sample index used in filenames, e.g. ux_0.csv
        measurement_type: "displacement" or "strain"

    Returns:
        dict with keys: x_values, y_values, data
    """
    measurement_type = str(measurement_type).lower()
    if measurement_type not in {"displacement", "strain"}:
        raise ValueError("measurement_type must be either 'displacement' or 'strain'.")

    base = Path(dic_path)
    if not base.is_absolute():
        base = _dataset_root() / dic_path
    if not base.exists():
        raise FileNotFoundError(f"DIC dataset folder not found: {base}")

    x_dic = _read_dic_csv(base / "x" / f"x_{sample_id}.csv")
    y_dic = _read_dic_csv(base / "y" / f"y_{sample_id}.csv")

    x_values = np.mean(x_dic, axis=0).reshape(-1, 1)
    y_values = np.mean(y_dic, axis=1).reshape(-1, 1)

    if measurement_type == "displacement":
        ux_dic = _read_dic_csv(base / "ux" / f"ux_{sample_id}.csv").T.reshape(-1, 1)
        uy_dic = _read_dic_csv(base / "uy" / f"uy_{sample_id}.csv").T.reshape(-1, 1)
        data = np.hstack([ux_dic, uy_dic])
    else:
        exx_dic = _read_dic_csv(base / "exx" / f"exx_{sample_id}.csv").T.reshape(-1, 1)
        eyy_dic = _read_dic_csv(base / "eyy" / f"eyy_{sample_id}.csv").T.reshape(-1, 1)
        exy_dic = _read_dic_csv(base / "exy" / f"exy_{sample_id}.csv").T.reshape(-1, 1)
        data = np.hstack([exx_dic, eyy_dic, exy_dic])

    return {
        "x_values": x_values,
        "y_values": y_values,
        "data": data,
    }


def get_biaxial_test_dataset_path(filename: str) -> Path:
    """Path of a biaxial-test FEM reference file (see src/phd/fem/biaxial_test.py)."""
    if not filename.endswith(".npz"):
        filename = f"{filename}.npz"
    filepath = _dataset_root() / "biaxial_test" / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Biaxial test dataset '{filename}' not found at {filepath}. "
            "Generate it with: conda run -n fenics python src/phd/fem/biaxial_test.py --law <nh|goh>"
        )
    return filepath


def load_biaxial_test_reference(filename: str) -> dict:
    """
    Load the FEM reference of an ideal planar biaxial test.

    Returns a dict with:
        coords: (n_grid^2, 2) reference coordinates [mm]
        states: (n_states, 2) prescribed [lambda11, lambda22]
        u:      (n_states, n_grid^2, 2) displacement [mm]
        P:      (n_states, n_grid^2, 4) 1st Piola-Kirchhoff [Pxx,Pxy,Pyx,Pyy] [MPa]
        force:  (n_states, 2) edge forces [N]
        meta:   dict with law, params, L, H, protocol, units
    """
    import json

    filepath = get_biaxial_test_dataset_path(filename)
    with np.load(filepath, allow_pickle=True) as data:
        return {
            "coords": data["coords"],
            "states": data["states"],
            "u": data["u"],
            "P": data["P"],
            "force": data["force"],
            "meta": json.loads(str(data["meta"])),
        }
