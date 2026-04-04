# 01 Architecture

Last updated: 2026-04-04

## 1. Purpose

This project builds a reproducible baseline pipeline for animal-image metrology on COCO:

- filter images containing at least two animals from two target categories
- obtain animal contours
- localize left/right eyes
- measure per-animal inter-eye distance
- estimate pairwise front/back relationship in a single image
- validate saved predictions against human ground truth when available

The current baseline prioritizes end-to-end reproducibility and validation over solving every subproblem with a learned model.

Document boundary:

- this file owns stable technical concepts, formulas, and runtime contracts
- operator setup and step-by-step command examples belong in `README.md`
- current status, TODOs, and roadmap belong in `docs/02_active_context.md`

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

Diagram-heavy end-to-end flow and the current module map live primarily in
`system_architecture.md`. This section keeps only the stable layer semantics.

### Unified CLI Lifecycle

- bootstrap one immutable runtime context
- register user-facing commands
- dispatch command execution polymorphically

Current contract:

- `CLIApplication` owns parser construction, bootstrap, and dispatch
- `CommandContext` encapsulates loaded config and runtime flags
- each user-facing command implements the shared `BaseCLICommand` contract
- heavy domain logic stays in dedicated command modules rather than the
  top-level router

### Data Domain: Dataset Layer

- load COCO annotations
- filter by category, count, area, overlap
- standardize image/annotation structure
- export reusable dataset assets

### Prediction Domain: Localization Layer

- populate per-annotation `eyes` results

Strategies:

- CV baseline: heuristic cascade/blob approach
- AI baseline: MMPose top-down animal pose

### Prediction Domain: Measurement Layer

- derive pixel measurements from predicted eyes

Outputs:

- per-animal `eye_distance_px`
- pairwise `front_back_proxy_gap_px`

### Validation Domain: Report Layer

- compute baseline statistics and GT-based accuracy
- export CSVs and debug artifacts

## 5. Asset Model

The project follows a four-layer asset model:

### A. Dataset Asset

Purpose:

- freeze dataset membership
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
- formal schema definitions, export wiring, and reload support now exist via:
  - `src/data/prediction_loader.py`
  - `src/data/prediction_store.py`
  - `main.py predict`
- prediction reload / reuse is wired into the canonical evaluation path via
  `main.py validate --prediction-run-id`
- prediction-generation separation now exists via:
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
- the user-facing `main.py validate` path now rebuilds a lightweight runtime
  dataset directly from Dataset Asset metadata and does not reload raw COCO
  annotations or detector stacks
- runtime-dataset rehydration still exists in internal evaluation helpers for
  fresh prediction-side flows and debug-oriented contexts, but saved prediction
  assets are no longer only a side output

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

See `system_architecture.md` for the current asset directory view and module
inventory.

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

Because this assignment only requires locating the two eyes as an unordered
pair, NME is computed with unordered eye-pair matching:

```text
direct  = (||p_L_pred - p_L_gt|| + ||p_R_pred - p_R_gt||) / (2 * d_IOD_gt)
swapped = (||p_L_pred - p_R_gt|| + ||p_R_pred - p_L_gt||) / (2 * d_IOD_gt)

NME = min(direct, swapped)
```

This prevents laterality-convention mismatch from being counted as a large
localization failure when the two predicted eye points are spatially close to
the GT pair.

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
- keeps the localization backend swappable without changing dataset or validator interfaces

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

Canonical operator entry point for the five user-facing commands.

Global arguments:

- `--config`
- `--verbose`

### `main.py predict`

Owns prediction-side work:

- input: frozen Dataset Asset
- output: Prediction Asset under `assets/predictions/<run_id>/...`
- scope:
  - localization
  - measurement
- conceptual contract:
  - `A -> C`

### `main.py validate`

Owns GT-based validation:

- input:
  - frozen Dataset Asset
  - Human GT Asset
  - Prediction Asset
- output:
  - GT-based validation reports
- scope:
  - `NME`
  - `RDE`
  - pairwise ordering accuracy
- conceptual contract:
  - `A + B + C -> D`
- implementation boundary:
  - loads Dataset Asset directly
  - loads Human GT directly
  - loads Prediction Asset directly
  - does not require COCO download checks or rerun detector inference

### `src/cli/cmd_evaluate.py`

Internal advanced task-centric backend:

- still used as shared implementation support
- no longer the primary operator-facing CLI surface

## 10. Current Architectural Risks

1. Contours rely on COCO GT mask, not learned segmentation inference.
2. AI path relies on GT bbox, so current accuracy does not represent full-scene autonomous detection performance.
3. Front/back output is still a proxy and should never be described as metric depth.
4. Unified reporting and final delivery documents are still incomplete.
5. Final report consistency and delivery docs still need to stay aligned with the implemented CLI and asset model.
