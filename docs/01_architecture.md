# 01 Architecture

Last updated: 2026-04-03

## 1. Purpose

This project builds a reproducible baseline pipeline for animal-image metrology on COCO:

- filter images containing at least two animals from two target categories
- obtain animal contours
- localize left/right eyes
- measure per-animal inter-eye distance
- estimate pairwise front/back relationship in a single image
- evaluate predictions against human ground truth when available

The current baseline prioritizes end-to-end reproducibility and validation over solving every subproblem with a learned model.

## 2. Scope And Boundaries

### In scope

- COCO-based filtering and dataset standardization
- contour availability through COCO instance masks
- eye localization via CV and AI baselines
- pixel-space inter-eye measurement
- monocular relative-depth proxy and front/back ordering
- human annotation and GT-based evaluation

### Out of scope for the current baseline

- training a segmentation model for contour generation
- full end-to-end object detection plus pose from scratch
- physical 3D distance recovery in metric units
- camera calibration or multi-view geometry

## 3. Key Assumptions

1. Animal contours currently come from COCO `instance mask`, not from this repo's own segmentation inference.
2. The AI localization path currently solves `GT bbox -> top-down keypoint localization`, not full-scene detection.
3. Front/back output is a `relative depth proxy`, not real-world distance.
4. Reproducibility depends on frozen dataset and GT assets, not on re-running raw filtering each time.

## 4. System Layers

### Phase 1: Dataset Layer

Responsibility:

- load COCO annotations
- filter by category, count, area, overlap
- standardize image/annotation structure
- export reusable dataset assets

Primary files:

- `main.py` (`data` sub-command)
- `src/cli/cmd_data.py`
- `src/data/downloader.py`
- `src/data/loader.py`
- `src/data/asset_exporter.py`

Outputs:

- `output/test_samples.csv`
- `assets/datasets/<dataset_id>/manifest.json`
- `assets/datasets/<dataset_id>/instances.csv`

### Phase 2: Eye Localization Layer

Responsibility:

- populate per-annotation `eyes` results

Strategies:

- CV baseline: heuristic cascade/blob approach
- AI baseline: MMPose top-down animal pose

Primary files:

- `main.py` (`evaluate --task localization`)
- `src/cli/cmd_evaluate.py`
- `src/localization/base.py`
- `src/localization/detector_cv.py`
- `src/localization/detector_ai.py`
- `src/localization/factory.py`

### Phase 3: Measurement Layer

Responsibility:

- derive pixel measurements from predicted eyes

Primary file:

- `src/evaluation/valid_measure.py`

Outputs:

- per-animal `eye_distance_px`
- pairwise `front_back_proxy_gap_px`

### Phase 4: Evaluation Layer

Responsibility:

- compute baseline statistics and GT-based accuracy
- export CSVs and debug artifacts

Primary files:

- `main.py` (`evaluate` sub-command)
- `src/cli/cmd_evaluate.py`
- `src/evaluation/base.py`
- `src/evaluation/engine.py`
- `src/evaluation/valid_loc.py`
- `src/evaluation/valid_measure.py`
- `src/evaluation/valid_accuracy.py`

## 5. Asset Model

The project follows a four-layer asset model:

### A. Dataset Asset

Purpose:

- freeze Phase 1 membership
- provide stable join keys

Files:

- `assets/datasets/<dataset_id>/manifest.json`
- `assets/datasets/<dataset_id>/instances.csv`

Primary keys:

- `dataset_id`
- `image_id`
- `annotation_id`

### B. Human Ground Truth

Purpose:

- store reusable human labels

Files:

- `assets/ground_truth/<dataset_id>/human_labels.csv`
- `assets/ground_truth/<dataset_id>/meta.json`

Core GT fields:

- `left_eye_x`, `left_eye_y`
- `right_eye_x`, `right_eye_y`
- `depth_rank`
- `label_status`

### C. Prediction

Purpose:

- hold runtime or exported model outputs

Current state:

- a formal `C` layer now exists for the current project scope
- validator reports still exist as human-facing outputs on top of that layer
- Phase A schema definitions, Phase B export wiring, and Phase C reload
  wiring now exist via:
  - `src/data/prediction_loader.py`
  - `src/data/prediction_store.py`
  - `main.py evaluate --save-predictions`
- prediction reload / reuse is wired into the canonical evaluation path via
  `main.py evaluate --prediction-run-id`
- Phase D initial separation now exists via:
  - `src/prediction/builders.py`
  - `src/prediction/__init__.py`
  - `MeasurementValidator` consuming shared prediction builders instead of
    owning measurement-table generation inline
- the formal evaluation lifecycle now accepts `prediction_asset` through:
  - `BaseValidator.evaluate(..., prediction_asset=None)`
  - `BaseValidator.generate_report(..., prediction_asset=None)`
  - `EvaluationEngine.run(..., prediction_asset=None)`
  - `EvaluationEngine.run_all(..., prediction_asset=None)`

Canonical MVP files:

- `assets/predictions/<run_id>/run_meta.json`
- `assets/predictions/<run_id>/localization.csv`
- `assets/predictions/<run_id>/measurement_instances.csv`
- `assets/predictions/<run_id>/measurement_pairs.csv`

Current implementation note:

- localization and measurement prediction generation are now centralized in
  `src/prediction/builders.py`
- evaluators now formally accept `prediction_asset` through the shared
  lifecycle contract
- `MeasurementValidator` now prefers saved
  `measurement_instances.csv` / `measurement_pairs.csv` when
  `--prediction-run-id` is used
- `AccuracyValidator` now prefers saved measurement assets for
  `RDE` and pairwise ordering, while still using saved localization points
  for `NME`
- `LocalizationValidator` now prefers saved `localization.csv` directly
- runtime-dataset rehydration still exists as compatibility / context support
  for current debug-oriented flows, but saved prediction assets are no longer
  only a side output

Canonical run metadata:

- `run_id`
- `dataset_id`
- `created_at`
- `schema_version`
- `method`
- `model_name`
- `task_scope`
- `git_commit`
- `config_fingerprint`

Canonical prediction tables:

- localization:
  - predicted left/right eye coordinates
  - status
  - confidence
- measurement instances:
  - `eye_distance_px`
  - `depth_proxy_px`
  - `measurement_valid`
- measurement pairs:
  - pairwise proxy gap / ratio
  - relation
  - validity and skip reason

### D. Evaluation

Purpose:

- compare prediction with GT
- generate metrics and reports

## 6. Core Formulas

### Inter-eye distance

```text
d_eye = sqrt((x_left - x_right)^2 + (y_left - y_right)^2)
```

Unit: pixel

### Front/back proxy

For each animal:

```text
depth_proxy_px = eye_distance_px
```

For a pair:

```text
front_back_proxy_gap_px = abs(depth_proxy_a - depth_proxy_b)
```

Interpretation:

- larger apparent eye distance suggests the animal is closer
- this is a monocular proxy, not physical depth

## 7. GT-Based Accuracy Metrics

### NME: Normalized Mean Error

Measures eye localization error normalized by GT inter-ocular distance.

```text
e_L = ||p_L_pred - p_L_gt||
e_R = ||p_R_pred - p_R_gt||
d_IOD_gt = ||p_L_gt - p_R_gt||

NME = (e_L + e_R) / (2 * d_IOD_gt)
```

### RDE: Relative Distance Error

Measures the relative error between predicted and GT inter-eye distance.

```text
d_pred = ||p_L_pred - p_R_pred||
d_gt   = ||p_L_gt - p_R_gt||

RDE = |d_pred - d_gt| / d_gt
```

### Pairwise Accuracy

Measures whether prediction preserves GT front/back ordering.

GT ordering source:

- `depth_rank` where `1 = closest`

Prediction ordering source:

- `eye_distance_px` as monocular proxy

Important:

- current pairwise accuracy is `conditional`, not full end-to-end
- it must be read together with pairwise coverage and exclusion counts

## 8. OOP And Module Design

### Detector abstraction

- `BaseEyeDetector` defines the common detector lifecycle
- `factory.py` selects CV or AI implementation
- keeps Phase 2 swappable without changing dataset or evaluator interfaces

### Validator abstraction

- `BaseValidator` defines `evaluate()` and `generate_report()`
- `EvaluationEngine` handles registration and execution
- each validator owns one responsibility:
  - localization statistics
  - measurement statistics
  - GT-based accuracy

### Separation of concerns

- Dataset preparation does not perform evaluation
- Validators do not run neural inference
- Annotation tooling does not own prediction logic
- Human GT is stored independently from prediction and reports

## 9. Runtime Contracts

### `main.py`

Canonical operator entry point.

Sub-commands:

- `data` -> Phase 1 COCO filtering and Dataset Asset export
- `annotate` -> interactive human GT annotation
- `review` -> GT overlay export and visual inspection
- `evaluate` -> localization / measurement / accuracy / prediction-asset flows

Global arguments:

- `--config`
- `--verbose`

### `main.py evaluate`

Task routing:

- `pipeline` -> `localization + measurement`
- `accuracy` -> GT-based validation and requires `--dataset-id`
- `all` -> all registered validators available for the current context
- `--save-predictions` -> export formal Prediction Asset MVP files for the
  current runtime flow
- `--prediction-run-id` -> load saved prediction assets and evaluate without
  re-running Phase 2 inference

## 10. Current Architectural Risks

1. Contours rely on COCO GT mask, not learned segmentation inference.
2. AI path relies on GT bbox, so current accuracy does not represent full-scene autonomous detection performance.
3. Front/back output is still a proxy and should never be described as metric depth.
4. Unified reporting and final delivery documents are still incomplete.
5. Final report consistency and delivery docs still need to stay aligned with the implemented CLI and asset model.
