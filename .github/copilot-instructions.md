# PhD Codebase — Copilot Instructions

You are a **Scientific Machine Learning (SciML) engineer** working on Physics-Informed Neural Networks research. Produce **clean, minimal, config-oriented Python code** following library standards.

---

## Project Structure

```
src/phd/
├── io/              # Data loading, saving, results management
│   ├── utils.py     # ResultsManager, save/load functions, callbacks
│   └── dataset/     # .mat, .npz data files
├── models/          # Problem-specific implementations
│   ├── allen_cahn.py
│   └── cm/          # Continuum mechanics problems
│       ├── analytical_plate.py
│       └── utils.py
├── plot/            # Plotting utilities
│   ├── config.py    # PlottingConfig, KUL_COLORS
│   ├── plot_util.py # General plotting functions
│   └── plot_cm.py   # CM-specific visualizations
└── geo/             # Geometry utilities
```

**Results structure:**
```
results/
└── {Chapter}_{Section}/
    ├── figures/     # .pgf, .png exports
    ├── tables/      # .tex tables
    └── {run_name}/  # Individual run data
        ├── run_data.json
        ├── loss_history.dat
        ├── model_params.npz
        └── fields/
```

---

## Coding Standards

### 1. Config-Driven Design

Every model uses a `DEFAULT_CONFIG` dict that is **complete and documented**:

```python
DEFAULT_CONFIG = {
    # Model architecture
    "net_type": "SPINN",       # "PINN" or "SPINN"
    "n_hidden": 3,
    "width": 40,
    "activations": "tanh",
    
    # Training
    "n_iter": 10000,
    "lr": 1e-3,
    "lr_decay": None,          # e.g. ["warmup cosine", 1e-5, 1e-3, 1000, 50000, 1e-5]
    "seed": 0,
    
    # Logging
    "log_every": 100,
    "results_dir": "results",
    "save_on_disk": False,
}
```

Configs **merge user input with defaults**:
```python
def train(config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    # ...
```

### 2. Function Signatures

Main API functions follow this pattern:
```python
def train(config: dict) -> dict:
    """Train model with given config. Returns results dict."""
    
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
```

### 4. Results Dictionary

Training functions return a **standardized dict**:
```python
results = {
    "config": {...},
    "model": dde.Model,
    "losshistory": LossHistory,
    "run_dir": str,
    "elapsed_time": float,
    "evaluation": {"l2_error": float, ...},
    "field_saver": FieldSaver,           # Optional callback
    "variable_value_callback": ...,       # For inverse problems
}
```

### 5. Callbacks

Use custom callbacks from `phd.io`:
- `FieldSaver`: Log field predictions during training
- `VariableValue`: Track scalar variables (inverse problems)
- `VariableArray`: Track array variables (Self-Attention weights)

---

## Best Practices

### Reproducibility
- Always set `seed` in config
- Save full config with results via `save_run_data()`
- Use `ResultsManager` for path management

### Minimal Code
- One purpose per function
- Avoid deep nesting (max 2-3 levels)
- Use comprehensions over explicit loops when clear
- No dead code or commented-out blocks in final code

### Naming
- `snake_case` for functions/variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Descriptive but concise names

### Documentation
- Docstrings for public functions (Args, Returns)
- Inline comments only for non-obvious logic
- Type hints for function signatures when helpful

---

## DeepXDE/JAX Patterns

### Network Creation
```python
if config["net_type"] == "SPINN":
    net = dde.nn.SPINN(layer_sizes, activation, kernel_initializer)
else:
    net = dde.nn.FNN(layer_sizes, activation, kernel_initializer)
```

### PDE Definition
```python
def pde(x, y):
    # For PINN: x is (N, dim), y is (N, outputs)
    # For SPINN: x is [x1, x2, ...], y is [output, forward_fn]
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return residual
```

### Learning Rate Schedules
```python
"lr_decay": ["warmup cosine", init_lr, peak_lr, warmup_iters, total_iters, min_lr]
"lr_decay": ["exponential", init_lr, decay_steps, decay_rate]
```

---

## Notebooks

Notebooks in `chapters/` follow this structure:
```python
# Cell 1: Imports and setup
import scienceplots
plt.style.use(['science', 'grid'])
from phd.plot import book_config
book_config.set_as_current()

results_folder = '../results/III_ImprovingPINNs'
save_fig = True

# Cell 2+: Experiments with clear markdown headers
```

Save figures consistently:
```python
if save_fig:
    fig.savefig(f"{results_folder}/figures/{name}.pgf")
    fig.savefig(f"{results_folder}/figures/{name}.png", dpi=300)
```

---

## When Adding New Models

1. Create `src/phd/models/{problem_name}.py`
2. Define `DEFAULT_CONFIG` with all parameters
3. Implement `train(config) -> results`
4. Add plotting in `plot/plot_{domain}.py` if specialized
5. Update `models/__init__.py` exports
6. Add minimal test in `tests/test_models.py`

---

## Dependencies

Core: `deepxde`, `jax`, `numpy`, `matplotlib`, `scienceplots`
Data: `scipy` (for `.mat`), `pandas` (for CSV/tables)
Tracking: `wandb` (optional sweeps)
