"""
IO package - handles data loading, saving, and dataset management.

API:
    save_run_data(results, run_name=None, problem=None, base_dir=None)
    load_run(run_name, problem, base_dir=None, restore_model=False, train_fn=None)
    save_field(fields_dir, step, fields_dict)
    load_fields(fields_dir)
    create_interpolation_fn(x_grid, y_grid, data_array, transform_fn=None)
"""

from .utils import (
    save_run_data,
    load_run,
    continue_training,
    FieldSaver,
    VariableValue,
    VariableArray,
    get_dataset_path,
    save_field,
    load_fields,
    load_field,
    create_interpolation_fn,
    save_df_to_latex,
    get_side_loaded_plate_dataset_path,
    load_side_loaded_plate_reference_raw,
    load_side_loaded_plate_dic_sample,
)

from .wandb_utils import (
    get_sweep_dir,
    setup_wandb_environment,
    setup_wandb_sweep,
    run_sweep_agent,
    get_sweep_commands,
    log_training_results,
    wandb_train_wrapper,
)

__all__ = [
    # Core utilities
    "save_run_data",
    "load_run",
    "continue_training",
    "FieldSaver",
    "VariableValue",
    "VariableArray",
    "get_dataset_path",
    # Field utilities
    "save_field",
    "load_fields",
    "load_field",
    "create_interpolation_fn",
    "get_side_loaded_plate_dataset_path",
    "load_side_loaded_plate_reference_raw",
    "load_side_loaded_plate_dic_sample",
    # Wandb utilities
    "get_sweep_dir",
    "setup_wandb_environment",
    "setup_wandb_sweep",
    "run_sweep_agent",
    "get_sweep_commands",
    "log_training_results",
    "wandb_train_wrapper",
    # LaTeX export
    "save_df_to_latex",
]
