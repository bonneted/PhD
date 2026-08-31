---
name: library-structure
description: Read library-structure.md before exploring or changing src/phd, and update it after any change to the library's shape. Use whenever you need to find where something lives in the phd package, add a model/config/dataset/physics module, or have just added, moved, renamed or removed anything under src/phd or configs/.
---

# phd library structure

`library-structure.md` at the repository root is the map of the `src/phd`
package: module-by-module API, cross-module dependency edges, and the known
structural risks. It is maintained by hand, so it is only useful if it is kept
in step with the code.

## Before exploring or editing `src/phd`

1. Read `library-structure.md` first. It tells you which module owns a
   responsibility (`phd.physics` for mechanics kernels, `phd.io` for
   persistence and callbacks, `phd.plot` for rendering, `phd.models.cm` for
   problem entry points) and which existing model is the closest template.
2. Only then grep the code. The document is the index; the code is the detail.
   Trust the code when they disagree — and fix the document (see below).
3. Follow the conventions in its section 7 "Editing Guidance":
   - `train(cfg=None, overrides=None)` signature for every model
   - standardised results dict keys so `phd.io` and `phd.plot` interoperate
   - shared math in `phd.physics`, shared persistence in `phd.io`
   - config-driven behaviour over hardcoded constants

## After changing the library

Update `library-structure.md` in the same change whenever any of these happen:

- a module, model, config file or dataset is added, renamed, moved or removed
- a module's public API changes (new exported function, changed signature)
- a new dependency edge appears between packages (e.g. a model starts using
  `phd.geo`)
- a structural risk listed in section 5 is fixed, or a new one is introduced

Which sections to touch:

| Change | Sections to update |
|---|---|
| New module or package | 1 (architecture), 2 (module API), 4 (connection map) |
| New model under `models/cm` | 2.4, plus 3 if it is exported from `__init__` |
| New physics kernel | 2.3 |
| New dataset + loader | 2.2 |
| New config file | 2.1 note or 2.4 as appropriate |
| Fixed / added structural problem | 5, and 6 if it was a listed refactor target |

Keep the existing tone: plain descriptive prose and bullet lists, no marketing,
and name real symbols and paths. Do not restructure the document wholesale to
record a small change — edit the affected sections in place.

## Related files

- `phd-structure.md` — the thesis/chapter structure, a separate document
- `configs/*.yaml` — one self-contained config per problem
- `src/phd/fem/` — FEniCS reference solutions (run in the `fenics` conda env),
  not importable library code
