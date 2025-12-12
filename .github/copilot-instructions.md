# PhD Codebase — Copilot Instructions

You are a **Scientific Machine Learning (SciML) engineer** working on Physics-Informed Neural Networks research. Produce **clean, minimal, config-oriented Python code** following library standards.

---

## Project Structure

```
configs/                 # Flat self-contained YAML configs (one per problem)
├── allen_cahn.yaml      # Complete Allen-Cahn config
└── analytical_plate.yaml # Complete analytical plate config

src/phd/
├── config/              # Hydra config loading
│   └── __init__.py      # load_config(), copy_config(), config_to_dict()
├── io/                  # Data loading, saving, results management
│   ├── utils.py         # ResultsManager, save/load functions, callbacks
│   ├── wandb_utils.py   # WandB sweep utilities
│   └── dataset/         # .mat, .npz data files
├── models/              # Problem-specific implementations
│   ├── allen_cahn.py
│   └── cm/              # Continuum mechanics problems
│       ├── analytical_plate.py
│       └── utils.py
├── plot/                # Plotting utilities
│   ├── config.py        # PlottingConfig, KUL_COLORS
│   ├── plot_util.py     # General plotting functions
│   └── plot_cm.py       # CM-specific visualizations
└── geo/                 # Geometry utilities

chapters/                # Thesis chapters (one folder per chapter)
└── {ChapterName}/
    ├── {ChapterName}.ipynb
    ├── images/          # .pgf, .png exports
    └── tables/          # .tex tables

results/                 # Training run outputs
└── {problem}/           # e.g., allen_cahn, analytical_plate
    ├── {experiment}/    # Auto-named: {timestamp}_{hash} or manual
    │   ├── run_data.json
    │   ├── loss_history.dat
    │   ├── model_params.npz
    │   └── fields/
    └── sweeps/          # WandB sweep results
        └── {sweep_name}/
            ├── sweep_config.yaml
            ├── sweep_info.yaml
            └── wandb/
```

---

## Configuration

### Simple, Flat Config Structure

Each problem has ONE self-contained YAML config file with ALL settings:

```yaml
# configs/allen_cahn.yaml
problem:
  name: allen_cahn
  pde_coefficient: 0.001

model:
  net_type: SPINN
  architecture:
    n_hidden: 3
    width: 20
    rank: 64
    activations: sin
  fourier_features:
    enabled: true
    n_features: 128
    sigma: 10.0

training:
  n_iter: 100000
  lr: 1.0e-5
  num_domain: 22500
  lr_decay: ["warmup cosine", 1e-5, 1e-3, 1000, 100000, 1e-5]
  self_attention:
    enabled: false

task:
  type: forward
results:
  base_dir: results
seed: 0
```

### Loading and Modifying Configs

```python
from phd.config import load_config, copy_config

# Load a problem config
cfg = load_config("allen_cahn")
cfg = load_config("analytical_plate")

# Modify directly (configs are mutable DictConfig)
cfg.training.n_iter = 50000
cfg.model.net_type = "PINN"
cfg.model.fourier_features.enabled = False

# Use overrides during load (for CLI-style changes)
cfg = load_config("allen_cahn", overrides=["training.n_iter=5000"])

# Copy config for running multiple experiments
base_cfg = load_config("allen_cahn")
cfg_pinn = copy_config(base_cfg)
cfg_pinn.model.net_type = "PINN"

cfg_spinn = copy_config(base_cfg)
cfg_spinn.model.net_type = "SPINN"
```

### Running Experiments from Notebooks

```python
from phd.config import load_config, copy_config
from phd.models import allen_cahn
from phd.io import ResultsManager, save_run_data, load_run

# Load and modify config
cfg = load_config("allen_cahn")
cfg.training.n_iter = 10000
cfg.model.fourier_features.enabled = True

# Train
results = allen_cahn.train(cfg)

# Save results (problem-specific wrapper handles problem name internally)
allen_cahn.save_run_data(results, run_name="my_experiment")

# Load saved results
loaded = allen_cahn.load_run("my_experiment")
```

### Multiple Experiment Variants

```python
from phd.config import load_config, copy_config, apply_overrides
from phd.models.allen_cahn import train, save_run_data, load_run

# Define base config once
base_cfg = load_config("allen_cahn")
base_cfg.training.n_iter = 100000
base_cfg.seed = 0

# Define variants as dicts of changes
variants = {
    "pinn": {"model.net_type": "PINN"},
    "spinn": {"model.net_type": "SPINN"},
    "spinn_ff": {"model.net_type": "SPINN", "model.fourier_features.enabled": True},
}

# Run each variant
for name, changes in variants.items():
    cfg = copy_config(base_cfg)
    apply_overrides(cfg, changes)
    
    results = train(cfg)
    save_run_data(results, run_name=name)
```

### Accessing Config Values

```python
cfg = load_config("allen_cahn")
sa_enabled = cfg.training.self_attention.enabled
```

---

## Results Management

### Save and Load Functions

Each problem module provides `save_run_data()` and `load_run()` wrappers that automatically handle the problem name:

```python
# In phd.models.allen_cahn:
from phd.models.allen_cahn import save_run_data, load_run

save_run_data(results, run_name="my_experiment")  # -> results/allen_cahn/my_experiment/
results = load_run("my_experiment")

# In phd.models.cm.analytical_plate:
from phd.models.cm.analytical_plate import save_run_data, load_run

save_run_data(results, run_name="SPINN_forward")  # -> results/analytical_plate/SPINN_forward/
results = load_run("SPINN_forward")
```

### Generic API (phd.io)

For advanced use or custom problems:

```python
from phd.io import save_run_data, load_run

# Save with explicit problem name
save_run_data(results, run_name="experiment", problem="my_problem")

# Load with explicit problem name
results = load_run("experiment", problem="my_problem")
```

---

## WandB Sweeps

### Setup and Run Sweeps

Results are stored in `results/{problem}/sweeps/{sweep_name}/`:

```python
from phd.io import setup_wandb_sweep, run_sweep_agent, get_sweep_commands

# Create a new sweep
sweep_id = setup_wandb_sweep(
    problem="allen_cahn",
    sweep_name="hyperparameter_search",
    sweep_config={
        "method": "bayes",
        "metric": {"name": "l2_error", "goal": "minimize"},
        "parameters": {
            "training.n_iter": {"values": [10000, 20000, 50000]},
            "model.architecture.width": {"min": 16, "max": 64},
        }
    },
    project="Allen-Cahn"
)

# Run sweep agent (in notebook or script)
run_sweep_agent(sweep_id, problem="allen_cahn", project="Allen-Cahn")

# Or generate CLI commands for terminal execution
commands = get_sweep_commands(
    sweep_id=sweep_id,
    problem="allen_cahn", 
    project="Allen-Cahn",
    entity="your-entity"
)
print(commands)
```

### Comprehensive Logging

Use `log_training_results()` to log all metrics to wandb:

```python
from phd.io import log_training_results

# After training in a wandb run
results = train(cfg)
log_training_results(results, log_history=True)
```

This logs:
- **Summary metrics**: `l2_error`, `final_loss`, `elapsed_time_s`, per-field errors
- **Training history**: loss per step, L2 error evolution
- **Parameter evolution**: for inverse problems (e.g., `lmbd`, `mu` values over time)

### Sweep Config with Hydra Paths

Use Hydra-style parameter paths in sweep configs:

```python
sweep_config = {
    'method': 'grid',
    'metric': {'name': 'l2_error', 'goal': 'minimize'},
    'parameters': {
        'model.net_type': {'value': 'SPINN'},
        'model.architecture.rank': {'values': [32, 64]},
        'training.n_iter': {'value': 100000},
        'training.self_attention.enabled': {'values': [False, True]},
    }
}
```

---

## Coding Standards

### 1. Config-Driven Design

Models use **Hydra DictConfig** directly. Each problem has one flat config file.

Train functions load config if not provided:
```python
def train(cfg: DictConfig = None, overrides: list = None):
    if cfg is None:
        cfg = load_config("allen_cahn", overrides=overrides)
    # Access nested values directly
    net_type = cfg.model.net_type
    n_iter = cfg.training.n_iter
```

### 2. Function Signatures

```python
def train(cfg: DictConfig = None, overrides: list = None) -> dict:
    """Train model with Hydra config. Returns results dict."""
    
def load_run(run_dir: Path) -> dict:
    """Load saved run. Returns same structure as train()."""
    
def plot_results(results: dict, **kwargs) -> tuple[Figure, dict]:
    """Plot results. Returns (fig, artists) for animation."""
```

### 3. Imports

Use **absolute imports** from `phd.*`:
```python
from phd.io import ResultsManager, save_run_data, load_run, get_dataset_path
from phd.plot import book_config, get_current_config
from phd.plot.plot_cm import plot_results, animate
from phd.config import load_config
```

### 4. Results Dictionary

Training functions return a **standardized dict**:
```python
results = {
    "config": {...},  # OmegaConf.to_container(cfg, resolve=True)
    "model": dde.Model,
    "losshistory": LossHistory,
    "run_dir": str,
    "elapsed_time": float,
    "evaluation": {"l2_error": float, ...},
    "field_saver": FieldSaver,
    "variable_value_callback": ...,
}
```

---

## Notebooks

Notebooks in `chapters/{Chapter}/` follow this structure:

```python
# Cell 1: Imports and setup
import scienceplots
plt.style.use(['science', 'grid'])
from phd.plot import book_config
book_config.set_as_current()

chapter_dir = Path(__file__).parent  # Or use Path.cwd()
save_fig = True

# Cell 2+: Experiments with clear markdown headers
```

Save figures consistently:
```python
if save_fig:
    fig.savefig(chapter_dir / "images" / f"{name}.pgf")
    fig.savefig(chapter_dir / "images" / f"{name}.png", dpi=300)
```

Save tables:
```python
df.to_latex(chapter_dir / "tables" / f"{name}.tex")
```

---

## Best Practices

### Reproducibility
- Always set `seed` in config
- Save full config with results via `save_run_data()`
- Use `ResultsManager` with explicit `experiment_name` for important runs

### Minimal Code
- One purpose per function
- Avoid deep nesting (max 2-3 levels)
- Use comprehensions over explicit loops when clear

### Naming
- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_CASE` for constants

---

## DeepXDE/JAX Patterns

### Network Creation
```python
if config["net_type"] == "SPINN":
    net = dde.nn.SPINN(layer_sizes, activation, kernel_initializer)
else:
    net = dde.nn.FNN(layer_sizes, activation, kernel_initializer)
```

### Learning Rate Schedules
```python
"lr_decay": ["warmup cosine", init_lr, peak_lr, warmup_iters, total_iters, min_lr]
"lr_decay": ["exponential", init_lr, decay_steps, decay_rate]
```

---

## Dependencies

Core: `deepxde`, `jax`, `numpy`, `matplotlib`, `scienceplots`
Config: `hydra-core`, `omegaconf`
Data: `scipy` (for `.mat`), `pandas` (for CSV/tables)
Tracking: `wandb` (optional sweeps)
