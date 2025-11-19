
# PINN Thesis Codebase — Quick Start (Packaging)

A short, practical README for installing the project as a local package and running the notebooks.

## 1) Install editable package

```bash
# From project root
pip install -e .
```

This installs the `examples` package from `src/examples/`, including bundled datasets.

## 2) Kernel and env notes

- Use a dedicated virtual environment (conda/venv) to avoid collisions with other packages.
- If you need jax GPU support, install the appropriate `jaxlib` before running experiments.

## 3) Import examples in notebooks

Now you can import examples cleanly without sys.path manipulation:

```python
from examples.allen_cahn import train, test_data, eval, plot_results

# Or use the original names (backward compatible)
from examples.allen_cahn import train_allen_cahn, test_data_allen_cahn

config = {"n_iter": 10000, "net_type": "SPINN"}
results = train(config)
```

The dataset is automatically loaded from the installed package—no manual path management needed.

## 4) Results and figures

- Notebooks save outputs under `results/` split by chapter/section (e.g., `results/IV_ImprovingTraining/IV_2_FourierFeatures_Figures`).
- Figures are exported as `.pgf` and `.png` and tables as `.tex`.

## 5) Package structure

```
src/examples/
├── __init__.py              # Package namespace
├── allen_cahn.py            # Allen-Cahn solver (clean function names)
└── data/
    ├── __init__.py          # Data package with get_dataset_path()
    └── Allen_Cahn.mat       # Dataset (packaged with installation)
```

Datasets are included via `package_data` in `setup.cfg` and loaded using `importlib.resources`, ensuring they work in development and production environments.

## 6) Optional: namespaced package

For safer namespacing to avoid the global `examples` name, rename the `src/examples/` directory and adjust imports accordingly.
