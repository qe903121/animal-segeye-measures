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

## 3. Current CLI Contracts

### Canonical operator flow

```text
data -> annotate / review -> predict -> validate
```

### `main.py data`

```bash
python main.py --config config/config.yaml data --skip-download
```

Use when:

- creating a new Dataset Asset
- changing categories or filtering rules

### `main.py annotate`

```bash
python main.py annotate --dataset-id <dataset_id> --annotator <name> --skip-labeled --no-imshow
```

Use when:

- building Human GT
- resuming labeling on a frozen dataset

### `main.py review`

```bash
python main.py review --dataset-id <dataset_id> --no-imshow
```

Use when:

- reviewing saved GT overlays
- checking skipped instances, eye points, and depth ranks

### `main.py predict`

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download
```

Use when:

- running all prediction-side work from a frozen Dataset Asset
- generating:
  - localization output
  - measurement output
  - formal Prediction Asset

Key contract:

- input: Dataset Asset
- output: Prediction Asset
- conceptual boundary:
  - `A -> C`
- `measurement` belongs here, not in GT validation

### `main.py validate`

```bash
python main.py validate --dataset-id <dataset_id> --prediction-run-id <run_id>
```

Use when:

- comparing one saved Prediction Asset against Human GT
- producing GT-based reports such as:
  - `NME` (unordered eye pair aware)
  - `RDE`
  - pairwise ordering accuracy

Key contract:

- input:
  - Dataset Asset
  - Human GT Asset
  - Prediction Asset
- output:
  - validation report
- conceptual boundary:
  - `A + B + C -> D`
- execution boundary:
  - no COCO download check
  - no raw COCO annotation reload
  - no detector inference rerun

### Internal advanced backend

- `src/cli/cmd_evaluate.py` still exists as a task-centric backend module
- it is no longer the primary operator-facing CLI contract
- user-facing docs should teach `predict` / `validate`, not validator task names

## 3.1 Implemented UX Convergence: `predict` then `validate`

Why this matters:

- prediction-side work and GT-based validation now have separate operator entry points
- users no longer need to learn internal validator task names first
- the system now matches the asset model more directly:
  - `predict` = `A -> C`
  - `validate` = `A + B + C -> D`

Operator-facing interpretation:

- `measurement` is prediction output
- GT is not required to create:
  - `measurement_instances.csv`
  - `measurement_pairs.csv`
- GT is required only when validating those outputs

## 4. Verified Current Behavior

### Verified dataset facts

- cat+dog baseline asset exists and can be reloaded
- evaluation can reconstruct the frozen asset instead of re-filtering

### Verified annotation facts

- `main.py annotate` and `main.py review` are both operational
- headless environments automatically disable `cv2.imshow()`
- review mode can render GT overlays for the committed sample dataset

### Verified evaluation facts

- `main.py --help` exposes exactly five user-facing commands:
  - `data`
  - `annotate`
  - `review`
  - `predict`
  - `validate`
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

### Detailed acceptance checklist: Prediction Asset Layer end-state

Goal:

- prove that the formal `C` layer can be exported once and then reused by
  evaluators without re-running detector inference
- prove that the engine / validator lifecycle now formally accepts
  `prediction_asset`, instead of relying only on runtime config mutation

Preconditions:

- frozen Dataset Asset exists
  - example: `coco_val2017_cat-dog_23714276`
- corresponding Human GT exists for accuracy validation
  - example:
    `assets/ground_truth/coco_val2017_cat-dog_23714276/human_labels.csv`

Acceptance procedure:

1. Export one formal Prediction Asset from a real dataset run
   Command:
   `python main.py predict --dataset-id coco_val2017_cat-dog_23714276 --method cv --skip-download --run-id <run_id> --output-dir <tmp_export_dir>`
   Must verify:
   - run exits successfully
   - `assets/predictions/<run_id>/run_meta.json` exists
   - `assets/predictions/<run_id>/localization.csv` exists
   - `assets/predictions/<run_id>/measurement_instances.csv` exists
   - `assets/predictions/<run_id>/measurement_pairs.csv` exists
   - log includes:
     - `Prediction Asset 匯出完成`

2. Re-run GT-based validation from the saved Prediction Asset
   Command:
   `python main.py validate --dataset-id coco_val2017_cat-dog_23714276 --prediction-run-id <run_id> --output-dir <tmp_acc_dir>`
   Must verify:
   - run exits successfully
   - log includes:
     - `Validation 契約: Dataset Asset + Human GT + Prediction Asset -> Report`
     - `已從 Dataset Asset 建立 lightweight runtime dataset`
     - `AccuracyValidator: 優先使用 saved measurement prediction assets 做 RDE / pairwise 評估。`
   - output CSVs exist:
     - `<tmp_acc_dir>/accuracy/eval_accuracy_instances.csv`
     - `<tmp_acc_dir>/accuracy/eval_accuracy_pairs.csv`

3. Lifecycle contract check
   Must verify in code:
   - `BaseValidator.evaluate(..., prediction_asset=None)`
   - `BaseValidator.generate_report(..., prediction_asset=None)`
   - `EvaluationEngine.run(..., prediction_asset=None)`
   - `EvaluationEngine.run_all(..., prediction_asset=None)`
   - `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
     all accept the formal `prediction_asset` parameter

4. Final acceptance decision
   The Prediction Asset Layer can be treated as functionally complete for the
   current project scope when all of the following are true:
   - export contracts are stable
   - localization evaluation reads saved localization assets directly
   - measurement evaluation reads saved measurement assets directly
   - GT-based accuracy uses saved measurement assets for `RDE` and pairwise
     evaluation
   - `main.py validate` requires no COCO download checks and no raw COCO
     reloads
   - remaining runtime-dataset construction in the validation path is only a
     lightweight Dataset Asset rehydration for report context, not a carrier
     of prediction generation

## 5. Delivery TODO

### P0: Must-finish before project handoff

1. Keep the current `main.py` CLI contract stable:
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
2. Keep the current `Prediction Asset Layer` contracts stable:
   - do not expand scope unless a delivery blocker appears
   - treat the acceptance checklist above as the regression contract
   - if behavior changes, re-run the full acceptance procedure
3. Write `README.md` end to end:
   - project overview
   - technical stack
   - local run steps
   - Docker / devcontainer usage
   - CLI entry points
   - output artifacts
4. Keep the system architecture diagram aligned with the current `A/B/C/D` layer model.
5. Finish the methodology write-up:
   - contour source and current limitation
   - CV vs AI localization framing
   - front/back proxy definition and boundary
   - validation formulas and interpretation
6. Backfill real baseline results into project docs:
   - localization
   - measurement
   - accuracy
7. Normalize the evaluation narrative so baseline and GT-based reporting are explained consistently.

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

- canonical CLI stabilization and documentation cleanup
- documentation completeness
- real result backfill
- evaluation story consistency
- clear statement of current scope and limits

### 6.1.a Unified CLI roadmap

Goal:

- stabilize and document the implemented unified CLI around `main.py`

Current state:

- `main.py` is the preferred operator entry point
- `main.py` currently exposes:
  - `data`
  - `annotate`
  - `review`
  - `predict`
  - `validate`
- legacy `run_*.py` entry scripts have been removed from the repository
- command ownership still remains in module code under `src/cli/` and the
  underlying feature packages; `main.py` is only the router
- `src/cli/cmd_evaluate.py` remains as an internal advanced backend, not the
  operator-facing mental model

Remaining cleanup:

1. Documentation phase
   - keep README, docs, and diagrams aligned with the 5-command model
   - avoid reintroducing task-centric examples into user-facing guidance
2. Backend stability phase
   - keep the internal `cmd_evaluate.py` helpers stable while user-facing
     commands remain `predict` / `validate`

Non-goals:

- do not redesign all argument names during delivery-critical work
- do not move core business logic into `main.py`
- do not perform a large CLI-framework rewrite

### 6.1.b Predict / Validate convergence roadmap

Goal:

- keep the implemented `predict -> validate` UX stable and understandable

Status:

- user-facing convergence is implemented
- prediction-side work now lives under `main.py predict`
- GT-based checking now lives under `main.py validate`
- the internal task-centric backend is still available only as implementation
  support

Guiding principles:

1. use asset transitions as the primary UX
   - `data` creates Dataset Asset
   - `annotate` / `review` create and inspect Human GT
   - `predict` creates Prediction Asset
   - `validate` compares GT against Prediction Asset
2. keep `measurement` on the prediction side
   - measurement is derived from predicted eyes
   - GT is only needed when validating measurement accuracy
3. keep task-centric evaluation internal
4. avoid schema churn while changing the operator interface

Implemented command model:

```bash
python main.py data ...
python main.py annotate ...
python main.py review ...
python main.py predict --dataset-id <dataset_id> --method ai --skip-download
python main.py validate --dataset-id <dataset_id> --prediction-run-id <run_id>
```

Planned meaning:

- `predict`
  - runs localization + measurement
  - always produces a Prediction Asset
  - may also emit prediction-side summaries / debug outputs

- `validate`
  - requires Human GT
  - consumes a saved Prediction Asset
  - runs GT-based checks such as:
    - NME
    - RDE
    - pairwise ordering accuracy

Current stabilization steps:

1. Keep `predict` output contract stable
   - formal Prediction Asset files remain canonical
   - no hidden GT dependency is introduced into prediction generation

2. Keep `validate` input contract strict
   - require:
     - `dataset_id`
     - Human GT
     - `prediction_run_id`
   - keep fail-fast rules explicit

3. Keep docs aligned with the operator mental model
   - primary examples use:
     `data -> annotate/review -> predict -> validate`
   - task-centric language stays in architecture/internal sections only

Acceptance criteria:

1. `predict` can run on a frozen Dataset Asset without Human GT
2. `predict` always exports a formal Prediction Asset
3. `validate` fails fast if GT or `prediction_run_id` is missing
4. `validate` does not rerun detector inference
5. README and operator examples teach `predict -> validate`, not task names
6. internal validator/task decomposition can still exist without leaking into
   the primary operator UX

Non-goals:

- do not redesign the Prediction Asset schema for this change alone
- do not rewrite validators just to rename commands
- do not re-promote task-centric `evaluate --task ...` into user-facing docs

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

Target state:

- keep the current export / reload / evaluator-reuse contract stable
- use the current acceptance checklist as the regression baseline
- avoid unnecessary schema churn before delivery

Current focus:

1. keep schema files and column contracts stable
2. treat the acceptance checklist in section 4 as the regression contract
3. prefer incremental reporting/documentation cleanup over new prediction
   features
4. revisit dedicated prediction-specific CLI only after delivery

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
