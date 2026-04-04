# AGENTS.md

Last updated: 2026-04-04

## Elevator Pitch

This repository builds a reproducible baseline metrology pipeline on COCO animal images.

Current baseline:

- Phase 1 filters multi-animal COCO images and exports frozen dataset assets
- contours currently come from COCO `instance mask`
- Phase 2 localizes eyes via CV and AI baselines
- the AI path uses `GT bbox + top-down MMPose on CPU`
- Phase 3 measures inter-eye pixel distance and front/back relative-depth proxy
- the canonical operator entry is now `main.py` with `data / annotate / review / predict / validate`
- the canonical CLI lifecycle now uses an OOP `Application / Command / Context` design
- human labels are stored as reusable GT assets and can be reviewed through `main.py annotate` and `main.py review`
- prediction-side work now lives under `main.py predict`
- GT-based checking now lives under `main.py validate`
- Prediction Asset Layer supports formal export, reload, and validator reuse as a first-class asset contract
- `main.py validate` now runs from Dataset Asset + Human GT + Prediction Asset only; it no longer depends on raw COCO reload for the user-facing validation path
- Prediction Asset `run_id` is immutable by default; explicit overwrite is now required

## Read Order

1. [Architecture & Methodology](docs/01_architecture.md)
2. [Active Context](docs/02_active_context.md)
3. [Dev Journal](docs/03_dev_journal.md)

## Reference Index

### Core Knowledge

- [docs/01_architecture.md](/workspace/docs/01_architecture.md)
  Contains stable system design, asset model, formulas, runtime contracts, and architectural boundaries.

- [docs/02_active_context.md](/workspace/docs/02_active_context.md)
  Contains current project status, frozen sample assets, CLI contracts,
  Prediction Asset acceptance criteria, delivery TODOs, roadmap, and handoff notes.

- [docs/03_dev_journal.md](/workspace/docs/03_dev_journal.md)
  Contains archived decisions, pitfalls, resolved bugs, and historical experiment notes.

### Supporting Documents

- [system_architecture.md](/workspace/system_architecture.md)
  High-level system diagram and layer-to-module map for the current `A/B/C/D`
  asset model and `data -> annotate/review -> predict -> validate` flow.

- [project_integrated_review_report.md](/workspace/project_integrated_review_report.md)
  Narrative project review report integrating current scope, assumptions, and validation framing.

## Maintenance Rule

Keep this file short.

- Put stable concepts in `docs/01_architecture.md`
- Put current status and next steps in `docs/02_active_context.md`
- Put solved pitfalls and historical notes in `docs/03_dev_journal.md`
