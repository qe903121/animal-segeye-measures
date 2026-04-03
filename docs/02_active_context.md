# 02 Active Context

Last updated: 2026-04-03

## 1. Product Goals

The synchronized end-state for this project is:

1. complete source code pushed to GitHub
2. `README.md` with project overview, stack, local run steps, Docker usage, API doc link, and test-account info if applicable
3. `.env.example`
4. at least one sample CSV artifact
5. system architecture diagram

## 2. Current Working Baseline

### Implemented

- Phase 1 dataset filtering pipeline
- CV eye-localization baseline
- AI eye-localization baseline using MMPose top-down on CPU
- contour baseline using COCO instance masks
- dataset asset export and reload by `dataset_id`
- human GT annotation and review workflow
- baseline measurement validator
- GT-based accuracy validator
- Prediction Asset Layer Phase A schema definitions (`run_id`, `run_meta.json`,
  canonical prediction CSV columns)
- Prediction Asset Layer Phase B export from current runtime flow via
  `main.py evaluate --save-predictions`
- Prediction Asset Layer Phase C loading via `main.py evaluate --prediction-run-id`
- Prediction Asset Layer Phase D initial separation and formal lifecycle support:
  - shared prediction builders in `src/prediction/builders.py`
  - `BaseValidator` / `EvaluationEngine` formally accept `prediction_asset`
  - `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
    directly consume saved prediction assets

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

### Canonical CLI: Dataset

```bash
python main.py --config config/config.yaml data --skip-download
```

Use when:

- creating a new dataset asset
- changing categories or filtering rules

### Canonical CLI: Annotation

```bash
python main.py annotate --dataset-id <dataset_id> --annotator <name> --skip-labeled --no-imshow
python main.py review --dataset-id <dataset_id> --no-imshow
```

Use when:

- building GT
- reviewing GT overlays

### Canonical CLI: Evaluation

```bash
python main.py evaluate --task pipeline --method ai --dataset-id <dataset_id> --skip-download
python main.py evaluate --task accuracy --method ai --dataset-id <dataset_id> --skip-download
python main.py evaluate --task localization --method ai --dataset-id <dataset_id> --skip-download --save-predictions
python main.py evaluate --task accuracy --prediction-run-id <run_id> --skip-download
```

Use when:

- running the canonical evaluation entry point
- keeping experiments aligned with a frozen dataset asset

Current task meanings:

- `pipeline` -> baseline validators only: `localization + measurement`
- `accuracy` -> GT-based metrics; requires valid `--dataset-id`

Prediction export:

- `--save-predictions` exports formal Prediction Asset MVP files under
  `assets/predictions/<run_id>/...`
- in real mode, `--save-predictions` requires `--dataset-id`
- `--run-id` can be provided explicitly, otherwise it is auto-generated

Prediction reload:

- `--prediction-run-id` reloads saved prediction assets and skips
  Phase 2 inference
- the associated `dataset_id` is resolved from `run_meta.json`
- evaluators can then run against saved prediction state instead of
  freshly generated runtime inference
- `localization` reads saved `localization.csv`
- `measurement` reads saved `measurement_instances.csv` /
  `measurement_pairs.csv`
- `accuracy` reads saved localization for `NME` and saved measurement for
  `RDE` / pairwise ordering

## 4. Verified Current Behavior

### Verified dataset facts

- cat+dog baseline asset exists and can be reloaded
- evaluation can reconstruct the frozen asset instead of re-filtering

### Verified annotation facts

- `main.py annotate` and `main.py review` are both operational
- headless environments automatically disable `cv2.imshow()`
- review mode can render GT overlays for the committed sample dataset

### Verified evaluation facts

- `pipeline --mock` runs cleanly without trying to execute accuracy
- `accuracy` fails fast without `--dataset-id`
- `accuracy` runs successfully against the committed sample GT
- `main.py --config ... --verbose evaluate --mock` correctly parses global args
- `--save-predictions` exports:
  - `run_meta.json`
  - `localization.csv`
  - `measurement_instances.csv`
  - `measurement_pairs.csv`
- `--prediction-run-id` can drive evaluation from saved prediction assets
  without re-running eye localization inference
- Phase D initial separation is now in place:
  - `src/prediction/builders.py` owns shared localization / measurement
    prediction-table generation
  - `MeasurementValidator` now consumes those builders instead of generating
    measurement prediction rows inline
  - `main.py evaluate` exports prediction assets from the shared builders
- `MeasurementValidator` now prefers saved measurement assets directly
- `AccuracyValidator` now prefers saved measurement assets for
  `RDE` / pairwise evaluation
- `LocalizationValidator` now prefers saved localization assets directly
- the formal lifecycle contract has been validated on the frozen sample asset:
  export once, then re-run `localization`, `measurement`, and `accuracy`
  through `--prediction-run-id` without rerunning Phase 2 inference
- the unified CLI router is implemented:
  - `main.py --help` exposes `data`, `annotate`, `review`, and `evaluate`
  - the repository now exposes only `main.py` as the operational entry point

### Detailed acceptance checklist: Prediction Asset Layer end-state

Goal:

- prove that the formal `C` layer can be exported once and then reused by
  evaluators without re-running Phase 2 inference
- prove that the engine / validator lifecycle now formally accepts
  `prediction_asset`, instead of relying only on runtime config mutation

Preconditions:

- frozen Dataset Asset exists
  - example: `coco_val2017_cat-dog_23714276`
- corresponding Human GT exists for accuracy validation
  - example:
    `assets/ground_truth/coco_val2017_cat-dog_23714276/human_labels.csv`
- real COCO source files are available locally, or `--skip-download` is valid

Acceptance procedure:

1. Export one formal Prediction Asset from a real dataset run
   Command:
   `python main.py evaluate --task localization --method cv --dataset-id coco_val2017_cat-dog_23714276 --skip-download --save-predictions --run-id <run_id> --output-dir <tmp_export_dir>`
   Must verify:
   - run exits successfully
   - `assets/predictions/<run_id>/run_meta.json` exists
   - `assets/predictions/<run_id>/localization.csv` exists
   - `assets/predictions/<run_id>/measurement_instances.csv` exists
   - `assets/predictions/<run_id>/measurement_pairs.csv` exists
   - log includes:
     - `Prediction Asset 匯出完成`

2. Re-run localization from the saved Prediction Asset
   Command:
   `python main.py evaluate --task localization --prediction-run-id <run_id> --skip-download --output-dir <tmp_loc_dir>`
   Must verify:
   - run exits successfully
   - log includes:
     - `已載入 saved prediction，跳過 Phase 2 inference`
     - `LocalizationValidator: 直接使用 saved localization prediction asset。`
   - output CSV exists:
     - `<tmp_loc_dir>/localization/eval_localization.csv`
   - debug images still render successfully

3. Re-run measurement from the saved Prediction Asset
   Command:
   `python main.py evaluate --task measurement --prediction-run-id <run_id> --skip-download --output-dir <tmp_measure_dir>`
   Must verify:
   - run exits successfully
   - log includes:
     - `已載入 saved prediction，跳過 Phase 2 inference`
     - `MeasurementValidator: 直接使用 saved measurement prediction assets。`
   - output CSVs exist:
     - `<tmp_measure_dir>/measurement/measurement_eye_distances.csv`
     - `<tmp_measure_dir>/measurement/measurement_front_back_pairs.csv`

4. Re-run GT-based accuracy from the saved Prediction Asset
   Command:
   `python main.py evaluate --task accuracy --prediction-run-id <run_id> --skip-download --output-dir <tmp_acc_dir>`
   Must verify:
   - run exits successfully
   - log includes:
     - `已載入 saved prediction，跳過 Phase 2 inference`
     - `AccuracyValidator: 優先使用 saved measurement prediction assets 做 RDE / pairwise 評估。`
   - output CSVs exist:
     - `<tmp_acc_dir>/accuracy/eval_accuracy_instances.csv`
     - `<tmp_acc_dir>/accuracy/eval_accuracy_pairs.csv`

5. Lifecycle contract check
   Must verify in code:
   - `BaseValidator.evaluate(..., prediction_asset=None)`
   - `BaseValidator.generate_report(..., prediction_asset=None)`
   - `EvaluationEngine.run(..., prediction_asset=None)`
   - `EvaluationEngine.run_all(..., prediction_asset=None)`
   - `LocalizationValidator`, `MeasurementValidator`, and `AccuracyValidator`
     all accept the formal `prediction_asset` parameter

6. Final acceptance decision
   The Prediction Asset Layer can be treated as functionally complete for the
   current project scope when all of the following are true:
   - export contracts are stable
   - localization evaluation reads saved localization assets directly
   - measurement evaluation reads saved measurement assets directly
   - GT-based accuracy uses saved measurement assets for `RDE` and pairwise
     evaluation
   - no evaluator requires re-running Phase 2 inference when
     `--prediction-run-id` is provided
   - remaining runtime-dataset rehydration exists only as compatibility /
     context support (for example debug visualisation), not as the primary
     prediction carrier

## 5. Delivery TODO

### P0: Must-finish before project handoff

1. Keep the current `main.py` CLI contract stable and finish the migration story:
   - treat `main.py` as the canonical operator entry point
   - keep current subcommands stable:
     - `data`
     - `annotate`
     - `review`
     - `evaluate`
   - preserve current global arguments:
     - `--config`
     - `--verbose`
   - switch README examples to `main.py`
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
4. Produce a system architecture diagram that matches the current `A/B/C/D` layer model.
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
  - `evaluate`
- legacy `run_*.py` entry scripts have been removed from the repository
- command ownership still remains in module code under `src/cli/` and the
  underlying feature packages; `main.py` is only the router

Remaining cleanup:

1. Documentation phase
   - update README examples to use `main.py`
   - update architecture / active-context docs to describe the router pattern
   - keep narrative reports aligned with the unified CLI end-state

Non-goals:

- do not redesign all argument names during delivery-critical work
- do not move core business logic into `main.py`
- do not perform a large CLI-framework rewrite

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
