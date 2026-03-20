# PhD Structure

This document explains how the library is used to produce reproducible PhD outputs (results, figures, tables), and defines normalization strategies for consistent future edits.

## 1. Scope and Layers

The project has three operational layers:

1. Library layer (src/phd): training, physics, IO, plotting, geometry.
2. Experiment layer (configs + notebooks + chapters): concrete runs and analysis.
3. Artifact layer (results + chapter images/tables): persisted outputs for thesis writing.

## 2. Reproducible Experiment Flow

Canonical workflow:

1. Load problem config via phd.config.load_config("problem_name").
2. Apply run-specific overrides (either in load_config overrides or mutating cfg fields).
3. Train with problem module train(cfg).
4. Persist with save_run_data(...), usually into results/problem/run_name.
5. Reload via load_run(...) for plotting/comparison without re-training.
6. Export figures and tables from chapter notebooks.

Reproducibility anchors already implemented:

- Seed in each config and explicit dde.config.set_random_seed(...) in train functions.
- Resolved config saved in run_data.json.
- Loss, model parameters, callback traces, and field snapshots are persisted.

## 3. Filesystem Roles

configs:

- Flat per-problem YAML configurations (allen_cahn, analytical_plate, side_loaded_plate, deep_notched, clamped_plate).

src/phd:

- Core implementation and reusable APIs.

chapters:

- Thesis-focused notebooks, each chapter saving local figures/tables.

notebooks:

- Additional exploratory notebooks (often predecessors or sandbox analyses).

results:

- Persistent training outputs by problem and run name.

examples:

- Legacy/example scripts (not fully aligned with modern src/phd APIs).

## 4. How Outputs Are Generated

## 4.1 Training outputs

Produced by phd.io.save_run_data:

- run_data.json
- loss_history.dat
- model_params.npz
- variables.dat and variable_arrays.npz when inverse/SA are enabled
- fields snapshots when field callbacks are enabled

## 4.2 Figures

Observed chapter patterns:

- PNG + PGF in some notebooks.
- PNG + PDF in others.
- Mixed folder conventions: images/, images/pgf/, images/pdf/, or separate constants.

## 4.3 Tables

Observed patterns:

- pandas DataFrame.to_latex direct usage.
- phd.io.save_df_to_latex usage in some notebooks.
- Mixed naming and formatting conventions by chapter.

## 5. Main Discrepancies Identified

## 5.1 Output format inconsistency

- Different chapters export different format sets (PGF/PNG/PDF) without a project-wide rule.

## 5.2 Path convention inconsistency

- Some notebooks use chapter-relative constants; others use ad-hoc ./images style paths.
- Some use IMAGE_DIR/PGF_DIR/PDF_DIR, others hardcode inline save paths.

## 5.3 API inconsistency between old and current notebooks

- Many notebooks use model-specific save/load wrappers.
- Some legacy notebooks/scripts still assume older non-Hydra config styles.

## 5.4 Testing and quality gate drift

- tests/test_models.py currently targets legacy signatures and misses current config-driven model interfaces.
- This reduces confidence that structural refactors are safe.

## 5.5 Dual legacy and active content

- Chapter folders include old variants (III_SPINNOld, III_ImprovingPINNsOld).
- src/phd/fem contains notebooks inside package source tree.
- This is useful for historical context but can confuse active workflow automation.

## 6. Normalization Strategy for Future Edits

## 6.1 Standard run contract

Required for every train result:

- results["config"] as DictConfig
- results["model"]
- results["losshistory"]
- results["runtime_metrics"]
- results["callbacks"] with stable keys

All new models should adopt this schema exactly.

## 6.2 Standard run directory schema

Keep current core layout and enforce optional files only when relevant:

- run_data.json, loss_history.dat, model_params.npz mandatory
- variables.dat, variable_arrays.npz only when corresponding callbacks exist
- fields/ only when field logging enabled

## 6.3 Standard artifact policy for chapters

Recommended default:

- Figures:
  - chapters/CHAPTER/images/NAME.png
  - chapters/CHAPTER/images/pgf/NAME.pgf
- Optional PDF only when explicitly required for review/publishing pipeline.
- Tables:
  - chapters/CHAPTER/tables/NAME.tex

Add a tiny helper utility in src/phd/plot or src/phd/io to centralize figure/table save behavior.

## 6.4 Standard notebook preamble

All chapter notebooks should start with:

1. Deterministic style/config setup.
2. chapter_dir, IMAGE_DIR, PGF_DIR, TABLE_DIR constants.
3. A save_artifacts boolean flag.

This prevents path drift and mixed save conventions.

## 6.5 Standard config usage

- Always load from configs via load_config.
- Use copy_config/apply_overrides for experiment variants.
- Avoid ad-hoc plain dict configs for new experiments.

## 6.6 Standard model persistence usage

- Prefer shared io.save_run_data/load_run APIs.
- If wrappers are kept, they should be thin and generated from a single shared pattern.

## 6.7 Standard naming

Run naming recommendation:

- problem_task_nettype_variant_seed
- Example: side_loaded_plate_inverse_spinn_orthotropic_s0

This improves result discoverability and scriptability.

## 7. Priority Fixes That Improve Reproducibility Immediately

1. Repair wandb config loading path to align with flat config names.
2. Repair model restoration path in io.utils so load_run(..., restore_model=True) is reliable.
3. Modernize tests/test_models.py to current Hydra-based APIs.
4. Introduce shared artifact-saving helper used by active chapter notebooks.

## 8. Guidance for AI Agents Editing This Repository

- Treat configs as the single source of experiment truth.
- Avoid introducing new one-off save/load conventions in notebooks.
- Keep chapter outputs deterministic and path-stable.
- Prefer small reusable utilities over duplicated notebook snippets.
- When behavior changes, update both documentation and instructions to keep the project coherent.
