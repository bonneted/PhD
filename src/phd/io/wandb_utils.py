"""
Wandb sweep utilities for the PhD project.

This module provides utilities for running wandb sweeps with proper
results storage following the project's folder structure:
    results/{problem}/sweeps/{sweep_name}/

Example usage:
    from phd.io.wandb_utils import setup_wandb_sweep, run_sweep_agent

    # Setup sweep for Allen-Cahn problem
    sweep_id = setup_wandb_sweep(
        problem="allen_cahn",
        sweep_name="hyperparameter_search",
        sweep_config={
            "method": "bayes",
            "metric": {"name": "l2_relative_error", "goal": "minimize"},
            "parameters": {
                "training.n_iter": {"values": [10000, 20000, 50000]},
                "model.architecture.width": {"min": 16, "max": 64},
            }
        },
        project="Allen-Cahn"
    )
    
    # Or run an existing sweep
    run_sweep_agent(sweep_id, problem="allen_cahn")
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from datetime import datetime


def get_sweep_dir(problem: str, sweep_name: Optional[str] = None, base_dir: str = "results") -> Path:
    """
    Get the directory for a wandb sweep following project conventions.
    
    Args:
        problem: Problem name (e.g., "allen_cahn", "analytical_plate")
        sweep_name: Optional name for the sweep (defaults to timestamp)
        base_dir: Base results directory
    
    Returns:
        Path to sweep directory: results/{problem}/sweeps/{sweep_name}/
    """
    if sweep_name is None:
        sweep_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get project root (assuming this file is in src/phd/io/)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    sweep_dir = project_root / base_dir / problem / "sweeps" / sweep_name
    
    return sweep_dir


def setup_wandb_environment(problem: str, sweep_name: Optional[str] = None, base_dir: str = "results") -> Path:
    """
    Set up wandb environment variables for proper results storage.
    
    This sets WANDB_DIR and WANDB_DATA_DIR to store wandb files in:
        results/{problem}/sweeps/{sweep_name}/
        results/{problem}/sweeps/{sweep_name}/wandb_cache/
    
    Args:
        problem: Problem name
        sweep_name: Optional sweep name
        base_dir: Base results directory
    
    Returns:
        Path to sweep directory
    """
    sweep_dir = get_sweep_dir(problem, sweep_name, base_dir)
    
    # Create directories
    sweep_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = sweep_dir / "wandb_cache"
    cache_dir.mkdir(exist_ok=True)
    
    # Set environment variables
    os.environ['WANDB_DIR'] = str(sweep_dir)
    os.environ['WANDB_DATA_DIR'] = str(cache_dir)
    
    return sweep_dir


def setup_wandb_sweep(
    problem: str,
    sweep_config: Dict[str, Any],
    project: str,
    sweep_name: Optional[str] = None,
    base_dir: str = "results",
) -> str:
    """
    Create a new wandb sweep with proper results storage.
    
    Args:
        problem: Problem name (e.g., "allen_cahn")
        sweep_config: Wandb sweep configuration dict
        project: Wandb project name
        sweep_name: Optional name for sweep directory
        base_dir: Base results directory
    
    Returns:
        Sweep ID string
    """
    import wandb
    
    # Setup environment
    sweep_dir = setup_wandb_environment(problem, sweep_name, base_dir)
    print(f"Sweep results will be saved to: {sweep_dir}")
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=project)
    
    # Save sweep config for reference
    import yaml
    config_file = sweep_dir / "sweep_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(sweep_config, f, default_flow_style=False)
    
    # Save sweep info
    info_file = sweep_dir / "sweep_info.yaml"
    with open(info_file, 'w') as f:
        yaml.dump({
            "sweep_id": sweep_id,
            "project": project,
            "problem": problem,
            "created_at": datetime.now().isoformat(),
        }, f, default_flow_style=False)
    
    return sweep_id


def run_sweep_agent(
    sweep_id: str,
    problem: str,
    train_fn: Optional[Callable] = None,
    project: Optional[str] = None,
    sweep_name: Optional[str] = None,
    count: Optional[int] = None,
    base_dir: str = "results",
    log_history: bool = True,
):
    """
    Run a wandb sweep agent with proper environment setup and comprehensive logging.
    
    Args:
        sweep_id: Wandb sweep ID
        problem: Problem name
        train_fn: Training function to use (if None, uses default based on problem)
        project: Wandb project name
        sweep_name: Optional sweep name for results storage
        count: Number of runs to execute
        base_dir: Base results directory
        log_history: Whether to log full training history
    """
    import wandb
    
    # Setup environment
    sweep_dir = setup_wandb_environment(problem, sweep_name, base_dir)
    print(f"Sweep agent results: {sweep_dir}")
    
    # Get training function wrapper
    wrapped_fn = wandb_train_wrapper(problem, train_fn, log_history=log_history)
    
    # Run agent
    wandb.agent(sweep_id, function=wrapped_fn, project=project, count=count)


def _get_default_train_fn(problem: str) -> Callable:
    """Get the default training function for a problem."""
    if problem == "allen_cahn":
        from phd.models.allen_cahn import train
        return train
    elif problem == "analytical_plate":
        from phd.models.cm.analytical_plate import train
        return train
    else:
        raise ValueError(f"Unknown problem: {problem}. Provide train_fn explicitly.")


def log_training_results(results: Dict[str, Any], log_history: bool = True):
    """
    Log comprehensive training results to wandb.
    
    This logs:
    - Final metrics (l2_error, final_loss, elapsed_time)
    - Full loss history (training loss, test loss, test metric per step)
    - Parameter evolution (for inverse problems)
    - Per-field L2 errors
    - Config summary
    
    Args:
        results: Training results dict from train() function
        log_history: Whether to log full history (can be expensive)
    """
    import wandb
    import numpy as np
    
    losshistory = results.get('losshistory')
    evaluation = results.get('evaluation', {})
    config = results.get('config', {})
    
    # === Final Summary Metrics ===
    summary = {
        "elapsed_time_s": results.get('elapsed_time', 0),
        "iterations_per_sec": results.get('iterations_per_sec', 0),
        "l2_error": evaluation.get('l2_error', evaluation.get('l2_relative_error')),
    }
    
    # Per-field errors
    if 'field_l2_errors' in evaluation:
        for field, error in evaluation['field_l2_errors'].items():
            summary[f"l2_error_{field}"] = error
    
    # Final loss
    if losshistory is not None and hasattr(losshistory, 'loss_train'):
        summary["final_loss"] = float(np.sum(losshistory.loss_train[-1]))
        summary["final_loss_train"] = float(np.sum(losshistory.loss_train[-1]))
        if hasattr(losshistory, 'loss_test') and len(losshistory.loss_test) > 0:
            summary["final_loss_test"] = float(np.sum(losshistory.loss_test[-1]))
    
    # Parameter values (for inverse problems)
    variable_callback = results.get('variable_value_callback')
    if variable_callback is not None and hasattr(variable_callback, 'value_history'):
        final_values = {name: vals[-1] for name, vals in variable_callback.value_history.items()}
        summary.update({f"final_{k}": v for k, v in final_values.items()})
    
    # Log summary
    wandb.log(summary)
    
    # === Training History ===
    if log_history and losshistory is not None:
        steps = getattr(losshistory, 'steps', list(range(len(losshistory.loss_train))))
        
        for i, step in enumerate(steps):
            history_log = {"step": step}
            
            # Training loss (can be multi-component)
            if hasattr(losshistory, 'loss_train') and i < len(losshistory.loss_train):
                train_loss = losshistory.loss_train[i]
                if hasattr(train_loss, '__iter__'):
                    history_log["loss_train_total"] = float(np.sum(train_loss))
                    for j, loss in enumerate(train_loss):
                        history_log[f"loss_train_{j}"] = float(loss)
                else:
                    history_log["loss_train_total"] = float(train_loss)
            
            # Test loss
            if hasattr(losshistory, 'loss_test') and i < len(losshistory.loss_test):
                test_loss = losshistory.loss_test[i]
                if hasattr(test_loss, '__iter__'):
                    history_log["loss_test_total"] = float(np.sum(test_loss))
                else:
                    history_log["loss_test_total"] = float(test_loss)
            
            # Test metric (L2 error)
            if hasattr(losshistory, 'metrics_test') and i < len(losshistory.metrics_test):
                metric = losshistory.metrics_test[i]
                if hasattr(metric, '__iter__'):
                    history_log["l2_error_step"] = float(metric[0])
                else:
                    history_log["l2_error_step"] = float(metric)
            
            wandb.log(history_log, step=step)
        
        # Log parameter evolution
        if variable_callback is not None and hasattr(variable_callback, 'value_history'):
            param_steps = variable_callback.steps
            for i, step in enumerate(param_steps):
                param_log = {}
                for name, vals in variable_callback.value_history.items():
                    if i < len(vals):
                        param_log[name] = vals[i]
                if param_log:
                    wandb.log(param_log, step=step)


def wandb_train_wrapper(
    problem: str,
    train_fn: Optional[Callable] = None,
    log_history: bool = True,
):
    """
    Create a wrapped training function for wandb sweeps with comprehensive logging.
    
    Args:
        problem: Problem name for loading config
        train_fn: Training function (defaults to problem-specific)
        log_history: Whether to log full training history
    
    Returns:
        Wrapped function suitable for wandb.agent()
    """
    import wandb
    from phd.config import load_config
    
    if train_fn is None:
        train_fn = _get_default_train_fn(problem)
    
    def wrapped():
        wandb.init()
        
        # Convert wandb config to Hydra overrides
        overrides = [f"{k}={v}" for k, v in wandb.config.items()]
        cfg = load_config(f"problem/{problem}", overrides=overrides)
        
        # Run training
        results = train_fn(cfg)
        
        # Log comprehensive results
        log_training_results(results, log_history=log_history)
        
        wandb.finish()
    
    return wrapped


def get_sweep_commands(
    sweep_id: str,
    problem: str,
    project: str,
    entity: Optional[str] = None,
    sweep_name: Optional[str] = None,
    base_dir: str = "results",
) -> str:
    """
    Generate shell commands to run a sweep agent.
    
    Args:
        sweep_id: Wandb sweep ID
        problem: Problem name
        project: Wandb project name
        entity: Wandb entity/username
        sweep_name: Optional sweep name
        base_dir: Base results directory
    
    Returns:
        Shell commands string
    """
    sweep_dir = get_sweep_dir(problem, sweep_name, base_dir)
    cache_dir = sweep_dir / "wandb_cache"
    
    entity_str = f"{entity}/" if entity else ""
    
    commands = f"""# Wandb Sweep Commands for {problem}
# Sweep ID: {sweep_id}

# Set up environment
export WANDB_DIR={sweep_dir}
export WANDB_DATA_DIR={cache_dir}

# Run sweep agent
wandb agent {entity_str}{project}/{sweep_id}
"""
    return commands
