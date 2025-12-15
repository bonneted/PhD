"""
utils.py
--------
Utilities for saving and loading training results and datasets.
Provides a clean, modular interface for persisting experiment data.
Also contains callbacks for logging during training.

API:
    save_run_data(results, run_name=None, problem=None, base_dir=None)
    load_run(run_name, problem, base_dir=None, restore_model=False, train_fn=None)
    save_field(fields_dir, step, fields_dict)
    load_fields(fields_dir)
    create_interpolation_fn(x_grid, y_grid, data_array, transform_fn=None)
"""

import json
import time
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
import deepxde as dde
from omegaconf import OmegaConf


# =============================================================================
# Interpolation utilities for exact solutions from data
# =============================================================================

def create_interpolation_fn(x_grid, y_grid, data_array, transform_fn=None):
    """
    Create an interpolation function for field data on a regular grid.
    
    Args:
        x_grid: 1D array of x coordinates (first axis)
        y_grid: 1D array of y coordinates (second axis)  
        data_array: 2D array of shape (len(x_grid), len(y_grid)) or 
                    3D array of shape (len(x_grid), len(y_grid), n_components)
        transform_fn: Optional function to transform input coordinates.
                      Should take x and return transformed coordinates.
                      For SPINN inputs (list of arrays), use this to convert to 2D array.
    
    Returns:
        Function that interpolates data at given coordinates.
        Input: x array of shape (N, 2) or list [x_coords, y_coords] for SPINN
        Output: array of shape (N,) or (N, n_components)
    """
    # Ensure data is at least 3D for consistent handling
    if data_array.ndim == 2:
        data_array = data_array[:, :, np.newaxis]
    
    n_components = data_array.shape[2]
    interpolators = []
    
    for i in range(n_components):
        interp = RegularGridInterpolator(
            (x_grid, y_grid),
            data_array[:, :, i],
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        interpolators.append(interp)
    
    def interpolation_fn(x):
        # Handle SPINN list input
        if isinstance(x, (list, tuple)):
            if transform_fn is not None:
                x_in = transform_fn(x)
            else:
                # Default: create meshgrid from list inputs
                x0, x1 = np.atleast_1d(x[0].squeeze()), np.atleast_1d(x[1].squeeze())
                xx, yy = np.meshgrid(x0, x1, indexing='ij')
                x_in = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        else:
            x_in = x
            
        results = np.array([interp((x_in[:, 0], x_in[:, 1])) for interp in interpolators]).T
        return results.squeeze() if n_components == 1 else results
    
    return interpolation_fn


# =============================================================================
# Field Utilities - Generic save/load for field snapshots
# =============================================================================

def save_field(fields_dir, step, fields_dict):
    """
    Save a single field snapshot to disk.
    
    Args:
        fields_dir: Path to the fields directory
        step: Iteration/step number
        fields_dict: Dict mapping field names to numpy arrays
                     e.g., {"u_pred": array, "pde_loss": array, "pde_weights": array}
    
    Example:
        save_field("results/allen_cahn/run1/fields", 1000, {"u_pred": u, "pde_loss": f})
    """
    fields_dir = Path(fields_dir)
    fields_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the field data
    np.savez_compressed(fields_dir / f"fields_{step}.npz", **fields_dict)
    
    # Update steps index (append if exists)
    steps_file = fields_dir / "steps.txt"
    if steps_file.exists():
        existing_steps = np.loadtxt(steps_file, dtype=int, ndmin=1).tolist()
        if step not in existing_steps:
            existing_steps.append(step)
            np.savetxt(steps_file, np.array(sorted(existing_steps), dtype=int), fmt="%d")
    else:
        np.savetxt(steps_file, np.array([step], dtype=int), fmt="%d")


def load_fields(fields_dir):
    """
    Load all field snapshots from a directory.
    
    Args:
        fields_dir: Path to the fields directory
        
    Returns:
        dict mapping step -> {field_name: array}, or None if no fields exist
        
    Example:
        fields = load_fields("results/allen_cahn/run1/fields")
        for step, data in fields.items():
            print(f"Step {step}: u_pred shape = {data['u_pred'].shape}")
    """
    fields_dir = Path(fields_dir)
    
    if not fields_dir.exists():
        return None
    
    steps_file = fields_dir / "steps.txt"
    if not steps_file.exists():
        return None
    
    steps = np.loadtxt(steps_file, dtype=int, ndmin=1)
    
    fields = {}
    for step in steps:
        npz_file = fields_dir / f"fields_{step}.npz"
        if npz_file.exists():
            with np.load(npz_file) as f:
                fields[int(step)] = {k: f[k] for k in f.keys()}
    
    return fields if fields else None


def load_field(fields_dir, step):
    """
    Load a single field snapshot.
    
    Args:
        fields_dir: Path to the fields directory
        step: Iteration/step number to load
        
    Returns:
        dict mapping field_name -> array, or None if not found
    """
    fields_dir = Path(fields_dir)
    npz_file = fields_dir / f"fields_{step}.npz"
    
    if not npz_file.exists():
        return None
    
    with np.load(npz_file) as f:
        return {k: f[k] for k in f.keys()}


# =============================================================================
# Callbacks for Training
# =============================================================================

class FieldSaver(dde.callbacks.Callback):
    """Callback to save field predictions during training.
    
    Args:
        period (int): Interval (number of epochs) between saving fields.
        x_eval: Evaluation points for prediction.
        field_names (dict or list): Mapping from output index to field name (dict),
            or list of field names if using output_field_fn.
        results_manager: Optional ResultsManager for saving to disk.
        save_to_disk (bool): Whether to save snapshots to disk during training.
        output_field_fn: Optional function (x, f, field_name) -> array to compute
            derived fields like strain/stress. If provided, field_names should be
            a list of field names to compute. f is a tuple (output_values, net_fn).
    """
    def __init__(self, period, x_eval, field_names, results_manager=None, save_to_disk=True, output_field_fn=None):
        super().__init__()
        self.period = period
        self.x_eval = x_eval
        self.results_manager = results_manager
        self.save_to_disk = save_to_disk
        self.output_field_fn = output_field_fn
        self.steps = []
        self.history = []  # List of (step, data_dict)
        self._jax_op = None  # JIT-compiled operator for derived fields
        
        # Normalize field_names to list
        if isinstance(field_names, dict):
            self.field_names = list(field_names.values())
            self.field_indices = field_names  # Keep original for direct output access
        else:
            self.field_names = list(field_names)
            self.field_indices = None
        
        if self.save_to_disk and self.results_manager:
            # Create fields directory
            self.fields_dir = self.results_manager.run_dir / "fields"
            self.fields_dir.mkdir(exist_ok=True)

    def on_train_begin(self):
        """Initialize JAX operator for derived field computation."""
        if self.output_field_fn is not None:
            import jax
            
            def jax_op(inputs, params, field_name):
                """JIT-compiled operator to compute field from network."""
                y_fn = lambda _x: self.model.net.apply(params, _x)
                f = (y_fn(inputs), y_fn)
                return self.output_field_fn(inputs, f, field_name)
            
            self._jax_op = jax.jit(jax_op, static_argnums=(2,))

    def on_epoch_end(self):
        if self.model.train_state.epoch % self.period == 0:
            self._record(self.model.train_state.epoch)

    def _record(self, step):
        """Record and optionally save fields at given step."""
        if self.output_field_fn is not None:
            # Use JIT-compiled operator for derived fields
            data_dict = {}
            params = self.model.net.params
            for name in self.field_names:
                data_dict[name] = np.asarray(self._jax_op(self.x_eval, params, name))
        else:
            # Direct output access
            y_pred = self.model.predict(self.x_eval)
            
            # y_pred shape: (N, n_fields)
            # Handle list of arrays for SPINN if needed
            if isinstance(y_pred, list):
                y_pred = np.array(y_pred)
            
            if self.field_indices is not None:
                data_dict = {name: y_pred[:, i] for i, name in self.field_indices.items()}
            else:
                data_dict = {name: y_pred[:, i] for i, name in enumerate(self.field_names)}
        
        # Store in memory
        self.history.append((step, data_dict))
        self.steps.append(step)
        
        # Save to disk if requested using generic save_field
        if self.save_to_disk and self.results_manager:
            save_field(self.fields_dir, step, data_dict)


class VariableValue(dde.callbacks.Callback):
    """Callback to log scalar variable values during training.

    Args:
        var_list: A Variable or list of Variables to track.
        period (int): Interval (number of epochs) between checking values.
        filename (string): Output the values to the file `filename`.
            The file is kept open to allow instances to be re-used.
            If ``None``, output to the screen.
        precision (int): The precision of variables to display.
        scale_factors (list): Optional list of scaling factors to apply to each variable.
            This is useful when variables are trained with a scaling factor.
    """

    def __init__(self, var_list, period=1, filename=None, precision=2, scale_factors=None):
        super().__init__()
        self.var_list = var_list if isinstance(var_list, list) else [var_list]
        self.period = period
        self.precision = precision
        self.filename = filename
        self.scale_factors = scale_factors if scale_factors is not None else [1.0] * len(self.var_list)

        self.file = None
        if filename:
            self.file = open(filename, "w", buffering=1)
            
        self.value = None
        self.epochs_since_last = 0
        self.history = []

    def on_train_begin(self):
        if dde.backend.backend_name == "tensorflow.compat.v1":
            raw_values = self.model.sess.run(self.var_list)
        elif dde.backend.backend_name == "tensorflow":
            raw_values = [var.numpy() for var in self.var_list]
        elif dde.backend.backend_name in ["pytorch", "paddle"]:
            raw_values = [var.detach().item() for var in self.var_list]
        elif dde.backend.backend_name == "jax":
            raw_values = [var.value for var in self.var_list]

        # Convert to standard python types and apply scale factors
        self.value = []
        for v, scale in zip(raw_values, self.scale_factors):
            if hasattr(v, "item"):
                self.value.append(float(v.item()) * scale)
            elif hasattr(v, "__array__"):  # numpy or jax array
                val = np.array(v).item() if np.ndim(v) == 0 else np.array(v)
                self.value.append(float(val) * scale if np.isscalar(val) else val * scale)
            else:
                self.value.append(float(v) * scale)

        # Store in history
        self.history.append([self.model.train_state.epoch] + self.value)

        if self.file:
            print(
                self.model.train_state.epoch,
                dde.utils.list_to_str(self.value, precision=self.precision),
                file=self.file,
            )
            self.file.flush()

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self.on_train_begin()

    def on_train_end(self):
        if not self.epochs_since_last == 0:
            self.on_train_begin()
        if self.file:
            self.file.close()

    def get_value(self):
        """Return the variable values (already scaled)."""
        return self.value


class VariableArray(dde.callbacks.Callback):
    """Callback to log array-valued variables (e.g., Self-Attention weights).
    
    Supports multiple arrays with named keys, and saves history to npz format.
    Can handle shared weights (same array used for multiple purposes) or 
    separate weights (different arrays for PDE and material losses).

    Args:
        var_dict: A dictionary mapping names to Variables.
                  E.g., {"pde_weights": var1, "mat_weights": var2} for separate weights,
                  or {"pde_weights": var1, "mat_weights": var1} for shared weights.
                  Can also be a list [var1, var2, ...] which will be named "var_0", "var_1", etc.
        period (int): Interval (number of epochs) between checking values.
        results_manager: Optional ResultsManager for saving to disk.
        save_to_disk (bool): Whether to save snapshots to disk.
        precision (int): The precision of variables to display (for printing).
    """

    def __init__(self, var_dict, period=1, results_manager=None, save_to_disk=False, precision=2):
        super().__init__()
        
        # Convert list to dict with auto-naming
        if isinstance(var_dict, list):
            var_dict = {f"var_{i}": v for i, v in enumerate(var_dict)}
        
        self.var_dict = var_dict
        self.period = period
        self.precision = precision
        self.results_manager = results_manager
        self.save_to_disk = save_to_disk
        
        self.value = None
        self.epochs_since_last = 0
        self.history = []  # List of (epoch, {name: array, ...})
        self.steps = []
        
        if self.save_to_disk and self.results_manager:
            self.save_dir = self.results_manager.run_dir / "variables"
            self.save_dir.mkdir(exist_ok=True)

    def _get_raw_values(self):
        """Extract raw values from variables based on backend."""
        values = {}
        for name, var in self.var_dict.items():
            if dde.backend.backend_name == "tensorflow.compat.v1":
                val = self.model.sess.run(var)
            elif dde.backend.backend_name == "tensorflow":
                val = var.numpy()
            elif dde.backend.backend_name in ["pytorch", "paddle"]:
                val = var.detach().cpu().numpy()
            elif dde.backend.backend_name == "jax":
                val = np.array(var.value)
            else:
                val = np.array(var)
            values[name] = val
        return values

    def on_train_begin(self):
        self._record()

    def _record(self):
        """Record current values."""
        epoch = self.model.train_state.epoch
        self.value = self._get_raw_values()
        
        # Store in history
        self.history.append((epoch, {k: v.copy() for k, v in self.value.items()}))
        self.steps.append(epoch)
        
        # Save to disk if requested
        if self.save_to_disk and self.results_manager:
            filename = self.save_dir / f"variables_{epoch}.npz"
            np.savez_compressed(filename, epoch=epoch, **self.value)
            np.savetxt(self.save_dir / "steps.txt", np.array(self.steps), fmt="%d")

    def on_epoch_end(self):
        self.epochs_since_last += 1
        if self.epochs_since_last >= self.period:
            self.epochs_since_last = 0
            self._record()

    def on_train_end(self):
        if self.epochs_since_last != 0:
            self._record()

    def get_value(self):
        """Return the current variable values as a dict of arrays."""
        return self.value
    
    def save_all(self, path=None):
        """Save all history to a single npz file.
        
        Args:
            path: Path to save file. If None, uses results_manager or raises error.
        """
        if path is None:
            if self.results_manager:
                path = self.results_manager.get_path("variable_arrays.npz")
            else:
                raise ValueError("No path provided and no results_manager available")
        
        # Build arrays for each variable across all epochs
        steps = np.array(self.steps)
        data = {"steps": steps}
        
        if self.history:
            first_snapshot = self.history[0][1]
            for name in first_snapshot.keys():
                # Stack arrays across time: shape (n_epochs, *array_shape)
                data[name] = np.array([h[1][name] for h in self.history])
        
        np.savez_compressed(path, **data)
        return path


# =============================================================================
# Results Manager (internal use)
# =============================================================================

def _generate_experiment_name():
    """Generate a unique experiment name: timestamp_shorthash."""
    import hashlib
    timestamp = int(time.time())
    short_hash = hashlib.md5(str(timestamp).encode()).hexdigest()[:6]
    return f"{timestamp}_{short_hash}"


def _get_default_base_dir():
    """Get default results base directory (project_root/results)."""
    current_path = Path(__file__).resolve()
    project_root = current_path.parent.parent.parent.parent
    if not (project_root / "pyproject.toml").exists() and not (project_root / ".git").exists():
        project_root = Path.cwd()
    return project_root / "results"


class ResultsManager:
    """Internal class for managing paths. Used by save/load functions.
    
    Directory structure: {base_dir}/{problem}/{run_name}/
    """
    
    def __init__(self, problem, run_name, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else _get_default_base_dir()
        self.problem = problem
        self.run_name = run_name
        self.run_dir = self.base_dir / self.problem / self.run_name
        
    def ensure_dir(self):
        """Create the run directory if it doesn't exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def get_path(self, filename):
        """Get the full path for a file in the run directory."""
        return self.run_dir / filename


def _extract_problem_and_run_name(results):
    """Extract problem name and run_name from results config."""
    config = results.get("config", {})
    
    # Extract problem name
    if isinstance(config.get("problem"), dict):
        problem = config["problem"].get("name")
    else:
        problem = config.get("problem_name") or config.get("problem")
    
    # Extract run_name from results.experiment_name or generate one
    if isinstance(config.get("results"), dict):
        run_name = config["results"].get("experiment_name")
    else:
        run_name = config.get("experiment_name") or config.get("run_name")
    
    return problem, run_name


# =============================================================================
# Saving Functions
# =============================================================================

def save_run_data(results, run_name=None, problem=None, base_dir=None):
    """
    Save all run data to disk.
    
    Saves:
    - run_data.json: config, run_metrics, evaluation
    - loss_history.dat: training loss history
    - model_params.npz: network parameters
    - variables.dat: trainable variable history (inverse problems)
    - variable_arrays.npz: SA weight history
    - fields/: field snapshots
    
    Args:
        results: Dictionary returned by train()
        run_name: Name for this run (subfolder name). If None, extracted from 
                  results["config"]["results"]["experiment_name"] or auto-generated.
        problem: Problem name (e.g., "allen_cahn"). If None, extracted from
                 results["config"]["problem"]["name"].
        base_dir: Base directory for results. Defaults to project_root/results.
    
    Example:
        # Simple: auto-extract names from config
        save_run_data(results)
        
        # Override run_name
        save_run_data(results, run_name="my_experiment")
        
        # Full control
        save_run_data(results, run_name="baseline", problem="allen_cahn", 
                      base_dir="./my_results")
    """
    # Extract defaults from config if not provided
    config_problem, config_run_name = _extract_problem_and_run_name(results)
    
    problem = problem or config_problem or "default"
    run_name = run_name or config_run_name or _generate_experiment_name()
    
    rm = ResultsManager(problem=problem, run_name=run_name, base_dir=base_dir)
    rm.ensure_dir()
    
    # Update run_dir in results
    results["run_dir"] = str(rm.run_dir)
    
    # Save each component
    _save_run_metadata(results, rm)
    _save_loss_history(results, rm)
    _save_model_params(results, rm)
    _save_variable_history(results, rm)
    _save_variable_arrays(results, rm)
    _save_field_snapshots(results, rm)
    
    print(f"Data saved to {rm.run_dir}")
    return rm.run_dir


def _save_run_metadata(results, rm):
    """Save run_data.json with config, metrics, and evaluation."""
    run_data = {
        "config": OmegaConf.to_container(results.get("config", {}), resolve=True),
        "run_metrics": {
            "elapsed_time": results.get("runtime_metrics", {}).get("elapsed_time"),
            "iterations_per_sec": results.get("runtime_metrics", {}).get("iterations_per_sec"),
            "net_params_count": results.get("runtime_metrics", {}).get("net_params_count"),
            "run_dir": results.get("run_dir"),
        },
    }
    
    run_data_file = rm.get_path("run_data.json")
    with open(run_data_file, "w") as f:
        json.dump(run_data, f, indent=2, default=str)
    print(f"Saved run metadata to {run_data_file}")


def _save_loss_history(results, rm):
    """Save DeepXDE loss history."""
    if "losshistory" not in results or results["losshistory"] is None:
        return
    dde.utils.save_loss_history(results["losshistory"], str(rm.get_path("loss_history.dat")))


def _save_model_params(results, rm):
    """Save network parameters and external trainable variables."""
    if "model" not in results or results["model"] is None:
        return
    
    model = results["model"]
    params_file = rm.get_path("model_params.npz")
    save_dict = {"params": model.net.params}
    
    # Also save external trainable variables if they exist
    if hasattr(model, 'external_trainable_variables') and model.external_trainable_variables:
        external_vars = {f"var_{i}": v.value for i, v in enumerate(model.external_trainable_variables)}
        save_dict["external_vars"] = external_vars
    
    np.savez(params_file, **{k: np.array(v, dtype=object) for k, v in save_dict.items()})
    print(f"Saved model parameters to {params_file}")


def _save_variable_history(results, rm):
    """Save trainable variable history (for inverse problems)."""
    cb = results.get("callbacks", {}).get("variable_value")
    if cb is None or not cb.history:
        return
    
    # Save history
    var_file = rm.get_path("variables.dat")
    with open(var_file, "w") as f:
        for row in cb.history:
            flat_row = []
            for item in row:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat_row.extend(item)
                else:
                    flat_row.append(item)
            f.write(" ".join(map(str, flat_row)) + "\n")
    
    # Save metadata (scale_factors, period)
    meta_file = rm.get_path("variables_meta.json")
    meta = {
        "scale_factors": cb.scale_factors if hasattr(cb, 'scale_factors') else None,
        "period": cb.period if hasattr(cb, 'period') else 1,
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f)


def _save_variable_arrays(results, rm):
    """Save SA weight arrays if they exist."""
    cb = results.get("callbacks", {}).get("variable_array")
    if cb is None or not cb.history:
        return
    
    print(f"Saving {len(cb.history)} SA weight snapshots...")
    cb.save_all(rm.get_path("variable_arrays.npz"))


def _save_field_snapshots(results, rm):
    """Save field snapshots to disk."""
    saver = results.get("callbacks", {}).get("field_saver")
    if saver is None or not saver.history:
        return
    
    print(f"Saving {len(saver.history)} field snapshots...")
    
    fields_dir = rm.get_path("fields")
    fields_dir.mkdir(exist_ok=True)
    
    # Save steps index
    steps = [h[0] for h in saver.history]
    np.savetxt(fields_dir / "steps.txt", np.array(steps, dtype=int), fmt="%d")
    
    # Save each snapshot
    for step, fields in saver.history:
        np.savez_compressed(fields_dir / f"fields_{step}.npz", **fields)


# =============================================================================
# Loading Functions
# =============================================================================

def load_run(run_name, problem, base_dir=None, restore_model=False, train_fn=None):
    """
    Load a saved run from disk.
    
    Args:
        run_name: Name of the run (subfolder name, e.g., "SPINN_forward")
        problem: Problem name (e.g., "allen_cahn", "analytical_plate")
        base_dir: Base directory for results. Defaults to project_root/results.
        restore_model: If True, reconstruct the model using saved parameters
        train_fn: Training function (required if restore_model=True)
    
    Returns:
        dict matching train() output structure
        
    Example:
        results = load_run("SPINN_forward", "analytical_plate")
        results = load_run("baseline", "allen_cahn", base_dir="./my_results")
    """
    rm = ResultsManager(problem=problem, run_name=run_name, base_dir=base_dir)
    run_dir = rm.run_dir
    
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    
    # Load all components
    config, run_metrics = _load_run_metadata(run_dir, rm)
    losshistory = _load_loss_history(rm)
    variable_value_callback = _load_variable_history(run_dir)
    model_params, external_vars = _load_model_params(run_dir)
    field_saver = _load_field_snapshots(run_dir, rm)
    
    result = {
        "config": config,
        "run_metrics": run_metrics,
        "losshistory": losshistory,
        "run_dir": str(run_dir),
        "runtime_metrics": {
            "elapsed_time": run_metrics.get("elapsed_time"),
            "iterations_per_sec": run_metrics.get("iterations_per_sec"),
            "net_params_count": run_metrics.get("net_params_count"),
        },
        "callbacks": {
            "field_saver": field_saver,
            "variable_value": variable_value_callback,
            "variable_array": None,
        },
        "model_params": model_params,
        "external_vars": external_vars,
        "model": None,
    }
    
    # Optionally restore the model
    if restore_model and model_params is not None and config:
        if train_fn is None:
            raise ValueError("train_fn is required when restore_model=True")
        result = _restore_model(result, train_fn)
    
    return result


def _load_run_metadata(run_dir, rm):
    """Load config, run_metrics, and evaluation from run_data.json or legacy files."""
    config, run_metrics, evaluation = {}, {}, {}
    
    run_data_file = run_dir / "run_data.json"
    if run_data_file.exists():
        with open(run_data_file, "r") as f:
            run_data = json.load(f)
        config = OmegaConf.create(run_data.get("config", {}))
        run_metrics = run_data.get("run_metrics", {})
    else:
        # Fallback: try legacy config.json
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = OmegaConf.create(json.load(f))
        else:
            print(f"Warning: No config found in {run_dir}")
    
    return config, run_metrics


def _load_loss_history(rm):
    """Load loss history and create a mock LossHistory object."""
    try:
        data = np.loadtxt(rm.get_path("loss_history.dat"))
        steps = data[:, 0]
        loss_train = data[:, 1:]
        metrics_test = loss_train[:, -1:] if loss_train.shape[1] > 0 else []
        
        class MockLossHistory:
            def __init__(self, steps, loss_train, metrics_test):
                self.steps = steps.tolist()
                self.loss_train = loss_train.tolist()
                self.loss_test = loss_train.tolist()
                self.metrics_test = metrics_test.tolist() if hasattr(metrics_test, 'tolist') else metrics_test
        
        return MockLossHistory(steps, loss_train, metrics_test)
    except Exception as e:
        print(f"Warning: Could not load loss history: {e}")
        return None


def _load_variable_history(run_dir):
    """Load trainable variable history."""
    var_file = run_dir / "variables.dat"
    if not var_file.exists():
        return None
    
    try:
        var_data = np.loadtxt(var_file)
        if var_data.ndim == 1:
            var_data = var_data.reshape(1, -1)
        
        # Load metadata if available
        meta_file = run_dir / "variables_meta.json"
        scale_factors = None
        period = 1
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
                scale_factors = meta.get("scale_factors")
                period = meta.get("period", 1)
        
        # Create a mock callback that inherits from dde.callbacks.Callback
        class MockVariableCallback(dde.callbacks.Callback):
            """Mock callback for loaded variable history (read-only)."""
            def __init__(self, history, scale_factors=None, period=1):
                super().__init__()
                self.history = history
                self.scale_factors = scale_factors
                self.period = period
        
        return MockVariableCallback(var_data.tolist(), scale_factors, period)
    except Exception as e:
        print(f"Warning: Could not load variables: {e}")
        return None


def _load_model_params(run_dir):
    """Load model parameters and external variables."""
    params_file = run_dir / "model_params.npz"
    if not params_file.exists():
        return None, None
    
    try:
        with np.load(params_file, allow_pickle=True) as f:
            model_params = f["params"].item()
            external_vars = f["external_vars"].item() if "external_vars" in f else None
        return model_params, external_vars
    except Exception as e:
        print(f"Warning: Could not load model params: {e}")
        return None, None


def _load_field_snapshots(run_dir, rm):
    """Load field snapshots into a mock FieldSaver."""
    # Use the FieldSaver class defined in this module
    field_saver = FieldSaver(period=1, x_eval=None, results_manager=rm, field_names={}, save_to_disk=False)
    field_saver.history = []
    
    fields_dir = run_dir / "fields"
    if not fields_dir.exists():
        return field_saver
    
    steps_file = fields_dir / "steps.txt"
    if not steps_file.exists():
        return field_saver
    
    field_steps = np.loadtxt(steps_file, dtype=int)
    if field_steps.ndim == 0:
        field_steps = np.array([field_steps])
    
    for step in field_steps:
        file_path = fields_dir / f"fields_{step}.npz"
        if file_path.exists():
            with np.load(file_path) as f:
                fields = {k: f[k] for k in f.keys()}
                field_saver.history.append((int(step), fields))
    
    return field_saver


def _restore_model(result, train_fn):
    """Restore a model from saved parameters."""
    print("Restoring model from saved parameters...")
    config = result["config"]
    
    restore_config = {
        **config,
        "n_iter": 0,
        "restored_params": result["model_params"],
        "restored_external_vars": result["external_vars"],
        "save_on_disk": False,
    }
    
    restored = train_fn(restore_config)
    result["model"] = restored["model"]
    
    return result


# =============================================================================
# Continue Training
# =============================================================================

def _restore_loss_history(model, old_losshistory):
    """
    Restore loss history to model so training continues from last step.
    
    Args:
        model: DeepXDE model
        old_losshistory: Previous LossHistory object (or mock)
    """
    if old_losshistory is None:
        return
    
    # Restore step counter to continue iteration numbering
    if old_losshistory.steps:
        last_step = int(old_losshistory.steps[-1])
        model.train_state.step = last_step
        model.train_state.epoch = last_step
    
    # Copy old history to model's losshistory
    model.losshistory.steps = list(old_losshistory.steps)
    model.losshistory.loss_train = list(old_losshistory.loss_train)
    model.losshistory.loss_test = list(old_losshistory.loss_test)
    model.losshistory.metrics_test = list(old_losshistory.metrics_test)


def _recreate_callbacks(results, model):
    """
    Recreate functional callbacks from results, preserving history.
    
    Args:
        results: Results dictionary with callbacks and model info
        model: DeepXDE model with external_trainable_variables
    
    Returns:
        List of functional callbacks with history restored
    """
    # FieldSaver, VariableValue, VariableArray are defined in this module
    
    callbacks = []
    config = results.get("config", {})
    log_every = config.get("log_every", 100)
    
    # Recreate FieldSaver if we have field history
    old_field_saver = results.get("field_saver")
    if old_field_saver and old_field_saver.history:
        # Get x_eval from old saver or recreate from config
        x_eval = getattr(old_field_saver, 'x_eval', None)
        
        if x_eval is None:
            # Recreate x_eval from config
            net_type = config.get("net_type", "PINN")
            if net_type == "SPINN":
                x_eval = [np.linspace(0, 1, 100).reshape(-1, 1)] * 2
            else:
                x_lin = np.linspace(0, 1, 100, dtype=np.float32)
                X_mesh = np.meshgrid(x_lin, x_lin, indexing="ij")
                x_eval = np.stack((X_mesh[0].ravel(), X_mesh[1].ravel()), axis=1)
        
        field_names = getattr(old_field_saver, 'field_names', {})
        # Try to restore field_names from history if not available
        if not field_names and old_field_saver.history:
            first_snapshot = old_field_saver.history[0][1]
            field_names = {i: name for i, name in enumerate(first_snapshot.keys())}
        
        new_field_saver = FieldSaver(
            period=log_every,
            x_eval=x_eval,
            field_names=field_names,
            results_manager=None,
            save_to_disk=False
        )
        # Restore history
        new_field_saver.history = list(old_field_saver.history)
        new_field_saver.steps = [h[0] for h in old_field_saver.history]
        callbacks.append(new_field_saver)
        results["field_saver"] = new_field_saver
    
    # Recreate VariableValue callback for inverse problems
    old_var_cb = results.get("variable_value_callback")
    if old_var_cb and old_var_cb.history and model.external_trainable_variables:
        # Get scale_factors from old callback or compute from config
        scale_factors = getattr(old_var_cb, 'scale_factors', None)
        if scale_factors is None and config.get("task") == "inverse":
            # Compute from config like analytical_plate.py does
            variables_training_factor = list(config.get("variables_training_factors", [1.0, 1.0]))
            if config.get("normalize_parameters", False):
                variables_training_factor[0] *= config.get("lmbd_init", 1.0)
                variables_training_factor[1] *= config.get("mu_init", 1.0)
            scale_factors = variables_training_factor
        
        # For inverse problems, only use the material variables (last 2)
        # Skip SA weights at the beginning
        n_sa_vars = 0
        if config.get("SA", False):
            n_sa_vars = 1 if config.get("SA_share_weights", True) else 2
        material_vars = model.external_trainable_variables[n_sa_vars:]
        
        if material_vars:
            new_var_cb = VariableValue(
                var_list=material_vars,
                period=log_every,
                scale_factors=scale_factors
            )
            # Restore history
            new_var_cb.history = list(old_var_cb.history)
            callbacks.append(new_var_cb)
            results["variable_value_callback"] = new_var_cb
    
    # Recreate VariableArray callback if present
    old_var_array_cb = results.get("variable_array_callback")
    if old_var_array_cb and old_var_array_cb.history:
        var_dict = getattr(old_var_array_cb, 'var_dict', None)
        if var_dict:
            new_var_array_cb = VariableArray(
                var_dict=var_dict,
                period=log_every,
                save_to_disk=False
            )
            new_var_array_cb.history = list(old_var_array_cb.history)
            new_var_array_cb.steps = list(old_var_array_cb.steps)
            callbacks.append(new_var_array_cb)
            results["variable_array_callback"] = new_var_array_cb
    
    return callbacks


def continue_training(results, n_iter, callbacks=None, display_every=None, 
                      recreate_callbacks=True, restore_history=True):
    """
    Continue training from existing results with proper history continuation.
    
    Args:
        results: Dictionary returned by train() or load_run() - must contain 'model'
        n_iter: Number of additional iterations to train
        callbacks: Optional list of additional callbacks. If None and recreate_callbacks=True,
                   recreates callbacks from results with history preserved.
        display_every: Display frequency (default: use config's log_every)
        recreate_callbacks: If True, recreate functional callbacks from loaded history.
                           Set False if you want to use only user-provided callbacks.
        restore_history: If True, restore loss history so iteration numbers continue.
    
    Returns:
        Updated results dictionary with extended history
    """
    import time as time_module
    
    if results.get("model") is None:
        raise ValueError("Results must contain a trained model. Use load_run with restore_model=True.")
    
    model = results["model"]
    config = results.get("config", {})
    
    if display_every is None:
        display_every = config.get("log_every", 100)
    
    # Restore loss history to continue from last step
    if restore_history:
        old_losshistory = results.get("losshistory")
        _restore_loss_history(model, old_losshistory)
        start_step = model.train_state.step
        print(f"Restored state: starting from step {start_step}")
    
    # Build callback list
    all_callbacks = []
    
    # Recreate functional callbacks from loaded results
    if recreate_callbacks and callbacks is None:
        all_callbacks = _recreate_callbacks(results, model)
        if all_callbacks:
            print(f"Recreated {len(all_callbacks)} callbacks with history preserved")
    
    # Add user-provided callbacks
    if callbacks:
        all_callbacks.extend(callbacks)
    
    if not all_callbacks:
        print("Warning: No callbacks available. Training will continue without field/variable logging.")
    
    # Continue training
    print(f"Continuing training for {n_iter} iterations...")
    start_time = time_module.time()
    losshistory, train_state = model.train(
        iterations=n_iter, 
        callbacks=all_callbacks if all_callbacks else None, 
        display_every=display_every
    )
    elapsed = time_module.time() - start_time
    
    # Update results with extended loss history
    results["losshistory"] = losshistory
    
    # Accumulate elapsed time
    prev_elapsed = results.get("elapsed_time", 0) or 0
    results["elapsed_time"] = prev_elapsed + elapsed
    
    # Update iterations per second (overall average)
    total_iter = config.get("n_iter", 0) + n_iter
    if results["elapsed_time"] > 0:
        results["iterations_per_sec"] = total_iter / results["elapsed_time"]
    
    # Update config to reflect new iteration count
    results["config"]["n_iter"] = total_iter
    
    print(f"Additional training: {elapsed:.2f}s, {n_iter/elapsed:.2f} it/s")
    print(f"Total elapsed time: {results['elapsed_time']:.2f}s")
    print(f"Total iterations: {losshistory.steps[-1] if losshistory.steps else 0}")
    
    return results


# =============================================================================
# Dataset Functions
# =============================================================================

def get_dataset_path(filename):
    """
    Get the full path to a dataset file.
    
    Args:
        filename: Name of the dataset file (e.g., 'Allen_Cahn.mat')
    
    Returns:
        Path object pointing to the dataset file in the dataset directory
    """
    # Get the directory where this module is located (src/phd/io)
    io_dir = Path(__file__).parent
    # Dataset is in src/phd/io/dataset
    dataset_dir = io_dir / "dataset"
    filepath = dataset_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset '{filename}' not found in {dataset_dir}")
    
    return filepath
