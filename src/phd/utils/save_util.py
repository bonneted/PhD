"""
save_util.py
------------
Utilities for saving and loading training results.
Provides a clean, modular interface for persisting experiment data.
"""

import os
import json
import time
import numpy as np
from pathlib import Path
import deepxde as dde


class ResultsManager:
    """Manages paths and basic I/O for a training run directory."""
    
    def __init__(self, run_name=None, base_dir=None):
        """
        Initialize the ResultsManager.
        
        Args:
            run_name: Specific name for the run. If None, generated from timestamp.
            base_dir: Base directory for results. Defaults to 'results' in project root.
        """
        if base_dir is None:
            current_path = Path(__file__).resolve()
            project_root = current_path.parent.parent.parent.parent
            if not (project_root / "pyproject.toml").exists() and not (project_root / ".git").exists():
                project_root = Path.cwd()
            self.base_dir = project_root / "results"
        else:
            self.base_dir = Path(base_dir)
            
        if run_name is None:
            run_name = f"run_{int(time.time())}"
            
        self.run_dir = self.base_dir / run_name
        
    def ensure_dir(self):
        """Create the run directory if it doesn't exist."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
    def get_path(self, filename):
        """Get the full path for a file in the run directory."""
        return self.run_dir / filename


# =============================================================================
# Saving Functions
# =============================================================================

def save_run_data(results, results_manager=None):
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
        results_manager: ResultsManager instance. If None, creates from run_dir.
    """
    if results_manager is None:
        run_dir = Path(results["run_dir"])
        results_manager = ResultsManager(run_name=run_dir.name, base_dir=run_dir.parent)
    
    results_manager.ensure_dir()
    
    # Save each component
    _save_run_metadata(results, results_manager)
    _save_loss_history(results, results_manager)
    _save_model_params(results, results_manager)
    _save_variable_history(results, results_manager)
    _save_variable_arrays(results, results_manager)
    _save_field_snapshots(results, results_manager)
    
    print("Data saved successfully.")


def _save_run_metadata(results, rm):
    """Save run_data.json with config, metrics, and evaluation."""
    run_data = {
        "config": results.get("config", {}),
        "run_metrics": {
            "elapsed_time": results.get("elapsed_time"),
            "iterations_per_sec": results.get("iterations_per_sec"),
            "run_dir": results.get("run_dir"),
        },
        "evaluation": {},
    }
    
    # Add evaluation results (excluding numpy arrays)
    if results.get("evaluation"):
        eval_results = results["evaluation"]
        run_data["evaluation"] = {
            "l2_error": eval_results.get("l2_error"),
            "field_l2_errors": eval_results.get("field_l2_errors"),
            "ngrid": eval_results.get("ngrid"),
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
    cb = results.get("variable_value_callback")
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
    cb = results.get("variable_array_callback")
    if cb is None or not cb.history:
        return
    
    print(f"Saving {len(cb.history)} SA weight snapshots...")
    cb.save_all(rm.get_path("variable_arrays.npz"))


def _save_field_snapshots(results, rm):
    """Save field snapshots to disk."""
    saver = results.get("field_saver")
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

def load_run(run_dir, restore_model=False, train_fn=None, eval_fn=None):
    """
    Load a saved run from disk.
    
    Args:
        run_dir: Path to the run directory
        restore_model: If True, reconstruct the model using saved parameters
        train_fn: Training function (required if restore_model=True)
        eval_fn: Evaluation function (required if restore_model=True)
    
    Returns:
        dict matching train() output structure
    """
    run_dir = Path(run_dir)
    rm = ResultsManager(base_dir=run_dir.parent, run_name=run_dir.name)
    
    # Load all components
    config, run_metrics, evaluation = _load_run_metadata(run_dir, rm)
    losshistory = _load_loss_history(rm)
    variable_value_callback = _load_variable_history(run_dir)
    model_params, external_vars = _load_model_params(run_dir)
    field_saver = _load_field_snapshots(run_dir, rm)
    
    result = {
        "config": config,
        "losshistory": losshistory,
        "run_dir": run_metrics.get("run_dir", str(run_dir)),
        "elapsed_time": run_metrics.get("elapsed_time"),
        "iterations_per_sec": run_metrics.get("iterations_per_sec"),
        "evaluation": evaluation,
        "field_saver": field_saver,
        "variable_value_callback": variable_value_callback,
        "model_params": model_params,
        "external_vars": external_vars,
        "model": None,
        "variable_array_callback": None,
    }
    
    # Optionally restore the model
    if restore_model and model_params is not None and config:
        if train_fn is None:
            raise ValueError("train_fn is required when restore_model=True")
        result = _restore_model(result, train_fn, eval_fn)
    
    return result


def _load_run_metadata(run_dir, rm):
    """Load config, run_metrics, and evaluation from run_data.json or legacy files."""
    config, run_metrics, evaluation = {}, {}, {}
    
    run_data_file = run_dir / "run_data.json"
    if run_data_file.exists():
        with open(run_data_file, "r") as f:
            run_data = json.load(f)
        config = run_data.get("config", {})
        run_metrics = run_data.get("run_metrics", {})
        evaluation = run_data.get("evaluation", {})
    else:
        # Fallback: try legacy config.json
        config_file = run_dir / "config.json"
        if config_file.exists():
            with open(config_file, "r") as f:
                config = json.load(f)
        else:
            print(f"Warning: No config found in {run_dir}")
    
    return config, run_metrics, evaluation


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
    # Import here to avoid circular imports
    from phd.models.cm.utils import FieldSaver
    
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


def _restore_model(result, train_fn, eval_fn=None):
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
    
    # Re-evaluate with restored model
    if eval_fn is not None:
        result["evaluation"] = eval_fn(config, restored["model"])
    
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
    from phd.models.cm.utils import FieldSaver, VariableValue, VariableArray
    
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
