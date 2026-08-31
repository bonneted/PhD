# Library Structure

This document describes the Python library under src/phd, how modules interact, and where structure issues currently make future edits harder.

## 1. High-Level Architecture

The library follows a mostly modular layout:

- src/phd/config: Hydra/OmegaConf loading and override utilities.
- src/phd/models: Problem entrypoints (Allen-Cahn and continuum mechanics variants).
- src/phd/physics: Pure JAX mechanics kernels and PDE builders (linear elasticity and finite-strain hyperelasticity).
- src/phd/io: Dataset resolution, callbacks, run persistence, wandb helpers.
- src/phd/plot: Config-driven plotting, animation, CM visualization logic.
- src/phd/geo: Mapping/mesh utilities for non-rectangular geometries.
- src/phd/fem: FEniCS reference solutions. Legacy FEniCS 2019 (conda env `fenics`), not importable library code.

Main runtime flow:

1. A model train function loads a config via phd.config.load_config.
2. It builds PDE operators from phd.physics (or local equivalents).
3. It creates logging callbacks from phd.io (FieldSaver, VariableValue, VariableArray).
4. It trains with DeepXDE and returns a standardized results dictionary.
5. Save/load is delegated to phd.io.save_run_data and phd.io.load_run.
6. Plotting wrappers call phd.plot.plot_cm (or local Allen-Cahn plotting).

## 2. Module-by-Module API and Dependencies

## 2.1 src/phd/config

Primary API:

- load_config(config_name, overrides=None)
- copy_config(cfg)
- apply_overrides(cfg, overrides_dict)
- config_to_dict(cfg), dict_to_config(d)

Behavior:

- Configs are loaded from project-root/configs.
- GlobalHydra is cleared before each composition to support notebook execution.
- Configs are mutable DictConfig objects.

Downstream consumers:

- All train functions across src/phd/models.
- Notebooks and chapter scripts.
- WandB wrappers in src/phd/io/wandb_utils.py.

Notes:

- get_config is an alias for load_config.
- set_nested and apply_overrides support dot-path mutation.

## 2.2 src/phd/io

Primary API (src/phd/io/utils.py):

- save_run_data(results, run_name=None, problem=None, base_dir=None)
- load_run(run_name, problem, base_dir=None, restore_model=False, train_fn=None)
- continue_training(results, n_iter, ...)
- FieldSaver, VariableValue, VariableArray
- Dataset helpers and interpolation helpers

Persistence structure:

- run_data.json (resolved config + run metrics)
- loss_history.dat
- model_params.npz (+ external_vars)
- variables.dat + variables_meta.json
- variable_arrays.npz
- fields/steps.txt + fields_STEP.npz + optional x_eval.npz

Dataset helpers (src/phd/io/dataset/utils.py, re-exported from phd.io):

- get_side_loaded_plate_dataset_path, load_side_loaded_plate_reference_raw,
  load_side_loaded_plate_dic_sample
- get_biaxial_test_dataset_path(filename), load_biaxial_test_reference(filename)
  Biaxial-test references are .npz bundles (coords, states, u, P, force, meta) produced by
  src/phd/fem/biaxial_test.py, unlike the flat .dat files of the other problems.

WandB layer (src/phd/io/wandb_utils.py):

- setup_wandb_sweep
- run_sweep_agent
- get_sweep_commands
- log_training_results

Important issue:

- wandb_train_wrapper currently uses load_config(f"problem/{problem}", ...). Configs are flat files such as allen_cahn.yaml, analytical_plate.yaml, etc. This path style is inconsistent with the active config design and likely fails for sweeps.

## 2.3 src/phd/physics

Primary API (src/phd/physics/mechanics.py):

- jacobian_spinn, jacobian_pinn, jacobian
- strain_from_jacobian
- isotropic_linear_elasticity, make_constitutive_fn
- momentum_balance
- strain_from_output, stress_from_output, make_output_field_fn
- make_pde(net_type, formulation, ...)

Finite-strain hyperelasticity (src/phd/physics/hyperelasticity.py):

- neo_hookean_energy, goh_energy, make_energy_fn(law, params)
  Incompressible plane-stress strain energy densities. Laws: "nh"/"neo_hookean",
  "goh" (fibre families at +/- alpha) and "goh_ref" (both families at +alpha, reproducing
  the reference projects/FIBER/GOH.py exactly).
- get_parameter_names, get_parameter_bounds, PARAMETER_BOUNDS
- deformation_gradient, green_lagrange, first_pk_from_F, first_pk_from_F_batch,
  cauchy_from_first_pk
- spatial_jacobian (SPINN/PINN; differentiates only w.r.t. the first two inputs, so extra
  input coordinates such as a loading-state axis are supported)
- make_hyperelastic_pde(energy_fn, net_type, formulation="mixed")
  Mixed formulation only, 6 outputs MIXED_OUTPUTS = [Ux, Uy, Pxx, Pxy, Pyx, Pyy]:
  2 equilibrium residuals (div_X P = 0) + 4 constitutive residuals.
- make_hyperelastic_output_field_fn(net_type)

Credit: the constitutive equations are ported from the KU Leuven Soft Tissue Mechanics
reference implementation (projects/FIBER/{NH,GOH}.py, H. Fehervary and J. Vastmans, 2020),
reformulated as a differentiable energy so P follows from autodiff.

Utilities (src/phd/physics/utils.py):

- transform_coords for SPINN input list -> tensor-grid coordinates (any input dimension)
- compute_loss_weight_factors (grad/ntk style)
- apply_loss_weight_grad_norm

Role:

- Shared mechanics core for CM models.
- Intended to reduce PDE logic duplication, but deep_notched and clamped_plate still implement significant custom residual logic locally.

## 2.4 src/phd/models

Allen-Cahn (src/phd/models/allen_cahn.py):

- train(cfg=None, overrides=None)
- save_run_data/load_run wrappers
- exact_solution, test_data, snapshot_fields
- standalone plotting helpers specific to Allen-Cahn layout

CM family (src/phd/models/cm):

- analytical_plate.py
- side_loaded_plate.py
- deep_notched.py
- clamped_plate.py
- biaxial_test.py
- biaxial_rakes.py

Common CM pattern:

- train(cfg=None, overrides=None)
- optional inverse mode with external trainable variables
- optional self-attention weighting
- results dict with callbacks
- save_run_data/load_run wrappers per problem
- thin plot wrappers delegating to phd.plot.plot_cm

biaxial_test.py specifics:

- Finite strain, so it uses phd.physics.hyperelasticity rather than make_pde.
- SPINN only, with a third input coordinate indexing the loading state, so one model
  covers the whole (lambda11, lambda22) protocol. Collocation points are installed with
  data.replace_with_anchors instead of geometry sampling, because the loading axis must sit
  exactly on the protocol nodes.
- make_loading_features is applied as a feature transform so the third SPINN factor sees
  the stretch pair (lambda11-1, lambda22-1) rather than the bare state index. Without it
  the loading axis under-fits badly: the stress is a jagged function of the index (the
  protocol interleaves four ratio blocks) and every high-index state ends up with nearly
  the same predicted force.
- The deformation is homogeneous by construction, so the identifiable measurement is the
  edge force, imposed as a PointSetOperatorBC whose operator integrates the P output over
  an edge. Displacement observations are optional and uninformative for the parameters.
- Extra result keys: "reference" (FEM data), "material" (true/identified parameters),
  "loss_labels". Helpers: parameter_summary, predict_state, predicted_forces.
- Works in normalised coordinates xi = X/L and displacement u/L; all residuals are divided
  by a stress scale derived from the largest measured force.

Scaling is per loading state, not global. Both the stress output scale and the
equilibrium/constitutive residual scale come from each state's own measured force. With
GOH the protocol stresses span ~290x (the fibre term is exponential), and a single global
scale leaves the low-stretch states needing network outputs around 3e-3, where no relative
accuracy is available; that alone was the difference between MPE 47% and MPE 16% on the
GOH identification. NH spans only 5.3x and is insensitive to the choice.

Parameterisation of the trainable material variables is selected by
task.inverse.parameterization and built by build_material_variables (shared with
biaxial_rakes):

- physical (default): "log" for strictly positive parameters, "sigmoid" for range-bounded
  ones, per task.inverse.parameter_scales.
- bounded: sigmoid onto task.inverse.bounds for every parameter.
- unconstrained: the original raw * training_factor behaviour; can leave the physical
  region (it drove C10 negative on GOH).

Known limitation: C10 is only weakly identifiable from edge forces at physiological
stretch. Its relative sensitivity is ~1000x below that of kappa, and a 1% force noise
gives it a 65% coefficient of variation (the other four parameters stay under 12%). The
thesis reports the same effect. GOH currently identifies k1, k2, kappa and alpha to
1-5% while C10 remains ~70% low.

Observed divergence:

- Analytical, side-loaded, deep-notched, and clamped models expose similar APIs but with non-uniform internal assumptions (field names, exact solution signatures, plotting wrappers, measurement config paths).
- clamped_plate is displacement-only and custom-plotted; it diverges strongly from mixed-formulation CM assumptions in plot_cm.

## 2.5 src/phd/plot

Config layer (src/phd/plot/config.py):

- PlottingConfig class
- default_config/book_config/book_compact_config/A4_config
- global config state via set_current_config/get_current_config
- KU Leuven palette helpers

General plotting (src/phd/plot/plot_util.py):

- metric panels
- parameter panels
- generic field plotting
- colorbar and figure initialization
- comparison and field evolution utilities

CM plotting (src/phd/plot/plot_cm.py):

- process_results, init_plot, update_frame, plot_results, animate
- plot_compare, plot_slice_comparison, plot_metrics_comparison
- helper overlays for DIC regions and curvilinear domains

Role:

- plot_cm is the integration hub joining callbacks, metrics, mesh transforms, and exact-solution adapters.

## 2.6 src/phd/geo

Main file: src/phd/geo/mapping.py

- legacy mesh-generation functions and hcubeMesh class
- geometry_mapping base class with cached map generation
- deep_notched mapping specialization

Role:

- Supplies computational->physical map used by deep_notched model.

Observed issue:

- This file mixes old research utility code (including legacy naming/style and unused helpers) with newer class-based mapping API, reducing maintainability.

biaxial_rakes.py (rake-based test, Abaqus reference):

- Same finite-strain mixed formulation, but no hard BC: the outer boundary is
  traction-free and the load enters through 20 interior rake holes. The measured
  displacement field is a dense soft constraint instead, and the stress level is set by
  section-force integrals H*int P_xx dY = Right_Fx on several parallel cuts between
  opposite rake rows (verified against the Abaqus rake forces to 0.013%).
- Collocation, constitutive residuals and displacement data are masked inside the
  r = 0.15 mm rake holes, where there is no material.
- Unlike the idealised test the deformation is heterogeneous (~20% deviation from the
  best-fit affine field), so the displacement field does constrain the parameters.

## 2.7 src/phd/fem

FEniCS reference solutions used to generate the datasets in src/phd/io/dataset.

- deep_notched.ipynb, side_loaded_plate.ipynb: notebooks (linear elasticity)
- biaxial_test.py: runnable script (finite-strain NH/GOH, plane-stress incompressible).
  Run with `conda run -n fenics python src/phd/fem/biaxial_test.py --law <nh|goh>`;
  writes src/phd/io/dataset/biaxial_test/ideal_LxLmm_<law>.npz.
- abaqus_rakes.py: converts the Abaqus rake results in projects/FIBER/Abaqus into
  src/phd/io/dataset/biaxial_test/rakes_ideal_LxLmm_<law>.npz. Pure numpy/scipy, so it
  runs in the normal JAX environment (no Abaqus, no FEniCS).

This directory is not importable library code: it needs the legacy FEniCS environment,
which is separate from the JAX/DeepXDE environment the rest of the library runs in.

## 3. Public API Surface

Root package exports (src/phd/__init__.py):

- __all__ = ["models", "config", "io", "plot"]

Current mismatch in root docstring:

- It references symbols such as eval and get_config in a way that does not match current canonical usage.

Models package exports:

- src/phd/models/__init__.py exports cm module, but not direct problem-level train aliases.

Implication:

- Most consumers import from concrete modules, not from a strongly curated top-level API.

## 4. Cross-Module Connection Map

Primary edges:

- models -> config: load_config
- models -> physics: PDE and constitutive mechanics
- models -> io: callbacks, save/load, ResultsManager
- models -> plot: rendering wrappers and animation
- models/deep_notched -> geo: coordinate mapping generation
- models/biaxial_test -> physics/hyperelasticity: strain energy, P, PDE residuals
- models/biaxial_rakes -> models/biaxial_test: shared parameterisation and loading features
- fem/biaxial_test.py -> io/dataset/biaxial_test (offline, different conda environment)
- notebooks/chapters -> models/config/io/plot

Secondary edges:

- io/wandb_utils -> config + models (train wrappers)
- plot/plot_cm -> io callback histories + model config schemas

## 5. Structural Risks and Inconsistencies

## 5.1 Redundancy and API drift

- Problem-specific save_run_data/load_run wrappers are duplicated in every model while generic io utilities already support problem selection.
- Wrapper existence is useful for ergonomics, but repetitive implementations increase maintenance burden.

## 5.2 Inconsistent training config expectations

- Some helper code assumes dict-like config access, while core train functions primarily rely on DictConfig.
- tests/test_models.py uses a legacy dictionary config style and helper names that no longer match current APIs.

## 5.3 Restore/continue training fragility

- io.utils._restore_model injects a plain dict-like restore_config, but modern train functions expect DictConfig structure with nested keys. This can break model restoration behavior.
- continue_training contains legacy callback key expectations (field_saver, variable_value_callback naming drift) and may not align with current callback storage format in results["callbacks"].

## 5.4 Plotting assumptions are not uniform across models

- plot_cm has generalized logic, but clamped_plate uses custom displacement-only rendering conventions.
- Field naming and exact-solution output dimensionality are not fully normalized across all models.

## 5.5 Legacy code in active package

- src/phd/fem contains notebooks and standalone scripts, not importable library modules.
- src/phd/models/cm/utils.py mainly re-exports symbols and adds little new behavior.
- src/phd/geo/mapping.py includes substantial legacy procedural code mixed with current API.

## 6. Refactor Targets (Recommended Order)

1. Normalize save/load interface:
- Keep wrappers for ergonomics but move all wrapper logic to one shared helper.

2. Fix reproducibility-critical integration bugs:
- Correct wandb config loading path.
- Make restore_model path train-function compatible with DictConfig.

3. Stabilize callback schema:
- Enforce one callback structure under results["callbacks"] across all models.

4. Normalize field/plot contracts:
- Define mandatory field naming conventions per formulation and ensure plot_cm validation.

5. Reduce legacy surface:
- Move or archive src/phd/fem notebooks outside package code path.
- Split src/phd/geo/mapping.py into legacy and maintained modules.

## 7. Editing Guidance for Future Contributors

- Prefer placing shared math in phd.physics and shared persistence in phd.io.
- Keep train function signatures consistent: train(cfg=None, overrides=None).
- Preserve standardized result dictionary keys so plot and io utilities remain interoperable.
- Prefer config-driven behavior over hardcoded run-time constants.
- Add thin wrappers only when they improve user ergonomics and are implemented through shared internals.
