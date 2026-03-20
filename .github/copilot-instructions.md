# PhD Codebase Copilot Instructions

This repository implements Scientific Machine Learning workflows for Physics-Informed Neural Networks and related inverse problems in continuum mechanics.

These instructions define how an agent should edit the codebase consistently, with reproducibility and thesis artifact quality as first-class constraints.

## Primary Scope

- Domain: Scientific Machine Learning, PINNs, SPINN, inverse material identification, uncertainty workflows.
- Core package: src/phd.
- Experiment definition: configs/*.yaml.
- Main analysis and deliverables: chapters/** notebooks and generated images/tables.

## Required Companion Documents

Before making structural edits, read and align with:

- library-structure.md
- phd-structure.md

These two documents are the canonical architecture and workflow references.

## Working Principles

1. Simplicity first.
- Prefer clear, small functions over layered abstractions.
- Avoid introducing framework complexity unless it removes repeated code.

2. Reuse over duplication.
- Put shared PDE/math logic in phd.physics.
- Put shared persistence and logging logic in phd.io.
- Put shared plotting logic in phd.plot.
- Keep model modules thin and problem-specific.

3. Config-driven behavior.
- New behavior should be driven by config values, not hardcoded constants.
- Do not bypass load_config for core experiments.

4. Reproducibility is mandatory.
- Respect cfg.seed and keep deterministic training setup.
- Ensure run metadata and resolved config are saved with results.
- Do not break save/load compatibility without a migration plan.

5. Keep API contracts stable.
- Preserve train(cfg=None, overrides=None) style where used.
- Preserve standardized results dictionary structure expected by io and plot modules.

## Library Editing Rules

## 1) Module placement

- src/phd/config: only config loading/copy/override utilities.
- src/phd/models: problem entrypoints and problem-specific glue only.
- src/phd/physics: reusable constitutive/PDE/differentiation kernels.
- src/phd/io: data loading, callbacks, save/load, sweep integration.
- src/phd/plot: reusable plotting/animation and chapter-friendly helpers.
- src/phd/geo: geometry mapping logic.

If logic is shared by 2 or more models, move it to physics/io/plot instead of copying.

## 2) Results contract

Each train function should return a compatible structure including:

- config
- model
- losshistory
- runtime_metrics
- callbacks
- run_dir when available

Do not rename these keys without updating all dependent modules.

## 3) Save/load consistency

- Prefer using phd.io.save_run_data and phd.io.load_run as canonical persistence APIs.
- If model-level wrappers are kept, they should remain thin delegations.

## 4) Plot consistency

- Keep field naming and dimensions consistent with plot_cm expectations.
- If a model needs custom plotting behavior, isolate it in wrappers and avoid duplicating generic plot logic.

## 5) Backward compatibility

When changing signatures, paths, or output formats:

- Add compatibility shims when feasible.
- Update dependent notebooks/scripts in the same change.
- Document any breaking change in markdown docs.

## PhD Workflow Rules

1. Configs in configs/*.yaml are the source of truth for experiments.
2. Chapter outputs should be deterministic and path-stable.
3. Figure/table export conventions should be normalized, not ad-hoc.
4. Legacy notebooks and historical folders should not drive new architecture decisions.

## Quality and Validation

When editing code:

- Run targeted checks/tests for changed modules when possible.
- Validate imports and basic execution paths.
- Prefer fixing root causes over patching symptoms in notebooks.

When editing docs/instructions:

- Keep architecture docs synchronized with real code.
- If the code structure changes, update library-structure.md and phd-structure.md.
- If process expectations change, update this file in the same commit.

## Known Current Hotspots

Use extra caution in these areas:

- wandb integration path assumptions in src/phd/io/wandb_utils.py.
- restore/continue training paths in src/phd/io/utils.py.
- Legacy tests that target outdated APIs in tests/test_models.py.
- Mixed artifact save conventions across chapter notebooks.

## Agent Behavior Expectations

- Prefer minimal, surgical edits.
- Avoid broad refactors unless explicitly requested.
- Keep scientific intent explicit in comments and naming.
- Explain assumptions when problem physics or units are ambiguous.

## Instruction Maintenance Rule

This file is living guidance.

Whenever you make structural edits that affect architecture, workflows, or coding conventions, update:

1. library-structure.md
2. phd-structure.md
3. .github/copilot-instructions.md

in the same change set, so future agents operate on accurate project knowledge.
