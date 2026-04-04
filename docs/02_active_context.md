# 02 Active Context

Last updated: 2026-04-04

## 1. Product Goals

The synchronized end-state for this project is:

1. complete source code pushed to GitHub
2. `README.md` with project overview, stack, local run steps, Docker usage, API doc link, and test-account info if applicable
3. `.env.example`
4. at least one sample CSV artifact
5. system architecture diagram

## 2. Current Working Baseline

### Implemented

- dataset filtering pipeline
- CV eye-localization baseline
- AI eye-localization baseline using MMPose top-down on CPU
- contour baseline using COCO instance masks
- dataset asset export and reload by `dataset_id`
- human GT annotation and review workflow
- baseline measurement validator
- GT-based accuracy validator
- Prediction Asset Layer schema definitions (`run_id`, `run_meta.json`,
  canonical prediction CSV columns)
- Prediction Asset Layer export from current runtime flow
- Prediction Asset Layer loading from saved prediction assets
- Prediction Asset Layer lifecycle support and separation:
  - shared prediction builders in `src/prediction/builders.py`
  - `BaseValidator` / `EvaluationEngine` formally accept `prediction_asset`
  - `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
    directly consume saved prediction assets
- user-facing CLI convergence to five commands:
  - `data`
  - `annotate`
  - `review`
  - `predict`
  - `validate`
- CLI lifecycle has been refactored into an OOP entry model:
  - `CLIApplication` for bootstrap + dispatch
  - `CommandContext` for immutable runtime state
  - one command object per user-facing sub-command
- system architecture diagram:
  - `system_architecture.md`
- validator file naming has been normalized under `src/evaluation/`:
  - `localization.py`
  - `measurement.py`
  - `accuracy.py`

### Sample frozen asset in repo

Dataset asset:

- `assets/datasets/coco_val2017_cat-dog_23714276/manifest.json`
- `assets/datasets/coco_val2017_cat-dog_23714276/instances.csv`

Human GT:

- `assets/ground_truth/coco_val2017_cat-dog_23714276/human_labels.csv`
- `assets/ground_truth/coco_val2017_cat-dog_23714276/meta.json`

Current sample GT status:

- 8 images
- 18 animal instances
- 16 `LABELED`
- 2 `SKIPPED`

## 3. Current Operator Contract

Primary owner:

- `README.md` is the only operator-facing workflow guide

Canonical operator flow:

```text
data -> annotate / review -> predict -> validate
```

Active contract snapshot:

- `data`
  - creates or refreshes a Dataset Asset
- `annotate` / `review`
  - create and inspect Human GT on top of a frozen Dataset Asset
- `predict`
  - consumes Dataset Asset and produces Prediction Asset
  - owns localization + measurement
  - conceptual boundary:
    - `A -> C`
- `validate`
  - consumes Dataset Asset + Human GT + Prediction Asset
  - produces GT-based reports only
  - conceptual boundary:
    - `A + B + C -> D`
  - must not rerun detector inference

Internal note:

- `src/cli/cmd_evaluate.py` still exists as an internal task-centric backend
- it is not part of the primary operator mental model
- stable CLI examples should stay in `README.md`, not here

## 4. Verified Current Behavior

### Verified dataset facts

- cat+dog baseline asset exists and can be reloaded
- evaluation can reconstruct the frozen asset instead of re-filtering

### Verified annotation facts

- `main.py annotate` and `main.py review` are both operational
- headless environments automatically disable `cv2.imshow()`
- review mode can render GT overlays for the committed sample dataset

### Verified evaluation facts

- `main.py --help` exposes exactly five user-facing commands
- the canonical CLI lifecycle now follows:
  - `main.py` -> `CLIApplication` -> `CommandContext` -> concrete command object
- `main.py predict` runs localization + measurement and exports:
  - `run_meta.json`
  - `localization.csv`
  - `measurement_instances.csv`
  - `measurement_pairs.csv`
  - prediction `run_id` is immutable by default; explicit overwrite is required
- `main.py validate`:
  - requires `--dataset-id`
  - requires `--prediction-run-id`
  - requires matching Human GT
  - does not rerun detector inference
  - does not depend on raw COCO reload in the user-facing path
- `src/prediction/builders.py` owns shared localization / measurement
  prediction-table generation
- `LocalizationValidator` prefers saved `localization.csv`
- `MeasurementValidator` prefers saved
  `measurement_instances.csv` / `measurement_pairs.csv`
- `AccuracyValidator` prefers saved localization for `NME` and saved
  measurement assets for `RDE` / pairwise evaluation
- the formal lifecycle contract has been validated on the frozen sample asset:
  export once via `predict`, then re-run GT-based checking via `validate`
  without rerunning detector inference

### Acceptance checklist: current regression contract

Purpose:

- keep the current `Prediction Asset Layer` and `predict -> validate` path
  stable during delivery work

Preconditions:

- one frozen Dataset Asset exists
- matching Human GT exists for that dataset

Regression path:

1. export one Prediction Asset via `main.py predict`
2. validate that saved asset via `main.py validate`

Must verify:

- prediction export succeeds and writes:
  - `run_meta.json`
  - `localization.csv`
  - `measurement_instances.csv`
  - `measurement_pairs.csv`
- validation succeeds from saved assets only
- `main.py validate` does not rerun detector inference
- `main.py validate` does not depend on raw COCO reload in the user-facing path
- `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
  still consume formal `prediction_asset` input through the shared lifecycle
- GT-based outputs still include:
  - `eval_accuracy_instances.csv`
  - `eval_accuracy_pairs.csv`

Interpretation:

- treat this as the working regression baseline for current delivery work
- if behavior changes, re-run this path before updating docs or expanding scope

## 5. Delivery TODO

### P0: Must-finish before project handoff

1. Execute a repo-level doc sync & consolidation pass:
   - current primary docs are too large and partially overlap:
     - `AGENTS.md` (~44 lines)
     - `README.md` (~216 lines)
     - `docs/01_architecture.md` (~457 lines)
     - `docs/02_active_context.md` (~433 lines)
     - `docs/03_dev_journal.md` (~356 lines)
     - `system_architecture.md` (~198 lines)
   - define and enforce one clear owner per documentation surface:
     - `AGENTS.md` = short entry point only
     - `README.md` = user-facing setup, workflow, and outputs
     - `docs/01_architecture.md` = stable technical source of truth
     - `docs/02_active_context.md` = current status, P0/P1/P2, roadmap, acceptance
     - `docs/03_dev_journal.md` = historical notes only
     - `system_architecture.md` = diagram-centric summary only
   - remove duplicated CLI walkthroughs, repeated asset explanations, and repeated
     methodology prose across README / architecture / active context
   - define explicit "do not duplicate" rules:
     - command examples live primarily in `README.md`
     - stable formulas live primarily in `docs/01_architecture.md`
     - current TODO / roadmap live primarily in `docs/02_active_context.md`
     - history and migration notes live primarily in `docs/03_dev_journal.md`
2. Keep the current `main.py` CLI contract stable:
   - treat `main.py` as the canonical operator entry point
   - keep current subcommands stable:
     - `data`
     - `annotate`
     - `review`
     - `predict`
     - `validate`
   - preserve current global arguments:
     - `--config`
     - `--verbose`
   - keep docs, diagrams, and reports aligned with the implemented command layout
3. Keep the current `Prediction Asset Layer` contracts stable:
   - do not expand scope unless a delivery blocker appears
   - treat the acceptance checklist above as the regression contract
   - if behavior changes, re-run the full acceptance procedure
4. Keep `README.md` delivery-ready and aligned with the repo:
   - project overview
   - technical stack
   - local run steps
   - Docker / devcontainer usage
   - CLI entry points
   - output artifacts
   - current API / accounts wording
5. Keep the system architecture diagram aligned with the current `A/B/C/D` layer model.
6. Finish the methodology write-up:
   - contour source and current limitation
   - CV vs AI localization framing
   - front/back proxy definition and boundary
   - validation formulas and interpretation
7. Backfill real baseline results into project docs:
   - localization
   - measurement
   - accuracy
8. Normalize the evaluation narrative so baseline and GT-based reporting are explained consistently.

### P1: Delivery support items

1. Add `.env.example`.
2. Organize at least one sample CSV artifact for delivery.
3. Review README / report wording so current contour handling is described honestly as a COCO-mask baseline, not learned segmentation inference.

### P2: Stabilization after delivery-critical docs

1. Improve unified reporting across baseline validators and GT-based accuracy.
2. Clarify and strengthen validation for the front/back proxy methodology.

## 6. Roadmap

### 6.1 Near-term roadmap: finish the baseline story cleanly

Goal:

- deliver one coherent, reproducible baseline system before expanding scope

Focus:

- documentation consolidation across README / architecture / active context
- keep the 5-command operator model stable
- keep `predict -> validate` and Prediction Asset contracts stable
- real result backfill
- evaluation story consistency
- clear statement of current scope and limits

Current execution themes:

1. Documentation consolidation
   - keep `AGENTS.md` as a short index
   - keep `README.md` as the only primary operator workflow guide
   - keep `docs/01_architecture.md` as the only primary owner of stable
     formulas and contracts
   - keep this file focused on status, acceptance, TODOs, and roadmap
   - prevent any major topic from being fully re-explained in three or more
     active docs

2. Operator contract stability
   - keep `main.py` as the canonical entry point
   - keep user-facing commands stable:
     - `data`
     - `annotate`
     - `review`
     - `predict`
     - `validate`
   - keep `cmd_evaluate.py` internal and avoid reintroducing task-centric
     examples into user-facing docs

3. Predict / validate boundary stability
   - keep `measurement` on the prediction side
   - keep `predict` free of Human GT dependency
   - keep `validate` strict about:
     - `dataset_id`
     - matching Human GT
     - `prediction_run_id`
   - keep `validate` free of detector inference reruns

Non-goals during this phase:

- no large CLI framework rewrite
- no broad argument renaming
- no Prediction Asset schema churn unless a delivery blocker appears

### 6.2 ONNX roadmap

Current state:

- AI eye localization uses RTMPose / MMPose on CPU
- ONNX Runtime has not been integrated into the main prediction path

Why it matters:

- may reduce CPU inference cost
- useful for deployment-oriented benchmarking
- useful for proving the AI path can be operationalized beyond pure PyTorch runtime

Planned stages:

1. Verify the official RTMPose AP-10K ONNX artifact and runtime dependencies.
2. Benchmark `PyTorch CPU` vs `ONNX Runtime CPU` on the same frozen dataset asset.
3. Confirm output parity for:
   - left/right eye coordinates
   - confidence / keypoint stability
   - downstream measurement outputs
4. Decide whether ONNX becomes:
   - an optional deployment backend
   - or the default CPU inference route

Acceptance questions:

- does ONNX materially improve latency?
- does it preserve enough prediction consistency for downstream measurement?
- does it add maintenance burden that is unjustified at current project scope?

### 6.3 Segmentation-model roadmap

Current state:

- contours currently come from COCO `instance mask`
- this is acceptable for the current baseline but is not the same as repo-native segmentation inference

Why it matters:

- the original task is segmentation-driven metrology
- using a learned contour source would make the system less dependent on COCO GT mask availability
- it would move the project closer to a true end-to-end autonomous pipeline

Planned stages:

1. Keep COCO mask as the current contour baseline for reproducible evaluation.
2. Define what counts as a valid segmentation upgrade:
   - contour quality
   - mask-to-contour conversion stability
   - compatibility with downstream localization / measurement
3. Add a learned segmentation path as an experimental contour source.
4. Compare:
   - COCO GT mask baseline
   - learned segmentation contour output
5. Reassess whether the learned path is mature enough to replace the baseline contour source.

Acceptance questions:

- does the learned contour source preserve usable object boundaries?
- does it improve system realism without destabilizing the rest of the pipeline?
- can its errors be evaluated clearly enough to support the metrology story?

### 6.4 Prediction Asset Layer roadmap

Goal:

- keep the now-formal `Prediction Asset Layer` stable for the current scope

Current state:

- formal prediction assets now exist under `assets/predictions/<run_id>/...`
- export and reload are implemented
- `BaseValidator` / `EvaluationEngine` formally accept `prediction_asset`
- `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
  directly consume saved prediction assets
- runtime dataset rehydration now exists mainly for debug/context support, not
  as the primary prediction carrier

Current focus:

1. keep schema files and column contracts stable
2. use the acceptance checklist in section 4 as the regression baseline
3. prefer reporting and documentation cleanup over new prediction features
4. avoid broad refactors until delivery-critical docs and results are settled

### 6.5 Longer-term architecture convergence

1. Evaluation consumes explicit `A + B + C` assets rather than mostly
   runtime-generated prediction state.
2. The system converges toward:
   - frozen dataset asset
   - reusable GT asset
   - reloadable prediction asset
   - evaluator as a pure comparison/report layer

## 7. Known Gaps

1. Contours are still sourced from COCO GT mask, not repo-native segmentation inference.
2. AI localization still depends on GT bbox top-down inference.
3. Front/back output remains a monocular proxy, not physical distance.
4. Summary unification across baseline validators and GT-based accuracy is still incomplete.
5. Real measurement / accuracy results are not fully written back into README and final delivery docs yet.

## 8. Handoff Notes

When starting new work:

1. read `AGENTS.md`
2. read this file
3. check `git status --short --branch`
4. confirm whether the work belongs to:
   - architecture/methodology
   - active execution context
   - archived history

When updating docs:

- put stable concepts in `docs/01_architecture.md`
- put current status and next steps here
- put solved pitfalls and old decisions in `docs/03_dev_journal.md`
