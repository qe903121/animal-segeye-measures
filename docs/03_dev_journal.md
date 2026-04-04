# 03 Dev Journal

Last updated: 2026-04-04

This file archives historical engineering notes, pitfalls, and major decision changes. It is not the primary source of truth for current architecture or active tasks.

## 0. Recent Structural Changes

- Refactored the canonical CLI into an OOP lifecycle:
  - `main.py` now delegates to `CLIApplication`
  - `CommandContext` encapsulates runtime config and global flags
  - each user-facing sub-command is represented by one command object
- Preserved the existing operator contract:
  - `data / annotate / review / predict / validate`
- Kept domain logic in `src/cli/cmd_*` modules while moving parser/bootstrap
  responsibility into shared abstractions under `src/utils/cli.py`

## 1. Historical Timeline

### 2026-04-02

- Created the initial `AGENTS.md`.
- Completed the first Phase 1 data pipeline.

### 2026-04-03

- Added overlap filtering and narrowed the baseline to `cat + dog`.
- Implemented Phase 2 CV eye localization.
- Integrated MMPose for Phase 2.5 AI localization.
- Added dataset assets and human GT assets.
- Added annotation review overlays.
- Added measurement validator and GT-based accuracy validator.
- Refined evaluation routing so `pipeline` no longer auto-runs `accuracy`.
- Introduced a unified `main.py` sub-command router with:
  - `data`
  - `annotate`
  - `review`
  - an initial task-centric `evaluate`
- briefly used wrapper-based migration for older entry scripts
- then removed the old top-level `run_*.py` entry scripts after docs and the
  canonical operator path converged on `main.py`
- later converged the user-facing operator UX further into:
  - `data`
  - `annotate`
  - `review`
  - `predict`
  - `validate`
  while keeping `src/cli/cmd_evaluate.py` only as an internal advanced backend
- Added Prediction Asset Layer Phase A/B groundwork:
  schema definitions plus export from current runtime flow.
- Added Prediction Asset Layer Phase C loading:
  saved prediction assets can now be reloaded into evaluation without
  re-running Phase 2 inference.
- Added Phase D initial separation:
  prediction-generation logic for localization / measurement was extracted
  into `src/prediction/builders.py`, and `MeasurementValidator` now consumes
  those shared builders instead of owning measurement-table generation.
- Extended Phase D so evaluators can consume saved measurement assets more
  directly:
  `MeasurementValidator` prefers `measurement_instances.csv` /
  `measurement_pairs.csv`, and `AccuracyValidator` prefers those same saved
  measurement tables for `RDE` and pairwise ranking.
- Formalized the lifecycle contract so `prediction_asset` is now an explicit
  parameter in `BaseValidator` and `EvaluationEngine`, instead of being only
  an implicit runtime-config side channel.
- Validated the current Prediction Asset Layer end-state on the frozen sample
  dataset:
  one exported prediction run can now drive `localization`, `measurement`,
  and `accuracy` through `--prediction-run-id` without rerunning Phase 2
  inference.
- Decoupled the user-facing `main.py validate` path from raw COCO reload:
  it now rebuilds a lightweight runtime dataset directly from Dataset Asset
  metadata and consumes saved Prediction Asset tables as the primary
  validation input.
- Added overwrite protection for Prediction Assets:
  reusing a `run_id` now fails fast unless `predict` is called with an
  explicit `--overwrite`.
- Normalized validator filenames inside `src/evaluation/`:
  - `valid_loc.py` -> `localization.py`
  - `valid_measure.py` -> `measurement.py`
  - `valid_accuracy.py` -> `accuracy.py`
- Added `system_architecture.md` as a dedicated high-level architecture
  diagram and asset-flow reference.

## 2. Key Pitfalls And Resolutions

### 2.1 CV localization reached a structural ceiling

Observed:

- Haar cascades achieved effectively 0% useful hits on COCO animal scenes.
- Blob fallback could produce detections but lacked semantic robustness.

Resolution:

- keep CV as a baseline / comparison path
- move primary localization path to AI

### 2.2 Initial MMPose inferencer path did not truly respect GT bbox

Observed:

- using high-level `MMPoseInferencer.__call__` could still involve whole-image detection behavior
- annotation-level calls could attach the wrong pose instance

Resolution:

- switched to `inference_topdown + GT bbox`
- each annotation now runs against its own bbox-anchored pose inference

### 2.3 MMPose dependency setup was fragile

Observed:

- `numpy>=2` caused C-ABI or build/runtime issues with `xtcocotools` and `chumpy`
- incorrect bbox nesting format caused runtime errors

Resolution:

- pin `numpy<2.0`
- use `--no-build-isolation` for problematic packages when needed
- preserve bbox format as `[[[x1, y1, x2, y2]]]` when required by the inferencer path

### 2.4 Root-owned files caused git permission conflicts

Observed:

- files created inside the container as `root` later caused checkout/unlink failures on host-side git operations

Resolution:

- fix ownership with `chown`
- avoid mixing root-created repo files with non-root git workflows when possible

### 2.5 Headless annotation preview could crash on Qt / xcb

Observed:

- `cv2.imshow()` aborted in environments without `DISPLAY` / `WAYLAND_DISPLAY`

Resolution:

- the annotation workflow now auto-disables imshow in headless environments
- `--no-imshow` remains the safest default in container workflows

### 2.6 Dataset asset identity was initially too weak

Observed:

- hashing only config-like knobs risked overwriting different memberships under the same dataset id

Resolution:

- include actual exported `(image_id, annotation_id)` membership in dataset identity
- persist `membership_digest` in `manifest.json`

### 2.7 `pipeline` began unintentionally running GT-based accuracy

Observed:

- once `accuracy` was registered, `pipeline -> run_all()` caused mock and baseline runs to include accuracy
- this produced spurious errors when `--dataset-id` was absent

Resolution:

- `pipeline` now explicitly runs only `localization + measurement`
- `accuracy` is opt-in and requires a valid `--dataset-id`

## 3. Archived Design Decisions

### Keep human GT separate from dataset and prediction

Reason:

- GT should be reusable across repeated experiments
- annotation work must not be redone for every validator

### Earlier plan: keep review inside the annotation entry point

Reason:

- avoid adding more run scripts during the first GT-tooling phase

What changed later:

- once the unified CLI router was implemented, `review` was promoted to a
  first-class `main.py` sub-command for clearer operator UX
- the repository later removed the old top-level annotation wrapper entirely
  after the migration completed

### Use COCO masks as a temporary contour baseline

Reason:

- stabilize the metrology pipeline first
- delay learned segmentation integration until the rest of the system is evaluable

## 4. Archived Experiment Notes

### CV baseline summary

- dataset: 8 images / 18 instances
- success: 10
- failed: 8
- practical conclusion: useful only as a baseline

### AI baseline summary

- dataset: 8 images / 18 instances
- success: 18
- CPU time: ~18.57s
- practical conclusion: current primary localization route

### Sample GT-based CV accuracy snapshot

- comparable instances: 9 / 16 GT-usable
- NME_mean: 3.9219
- RDE_mean: 244.6%
- pairwise accuracy: 100% on 2 comparable pairs

Interpretation:

- this snapshot is useful for validating the pipeline and the reporting logic
- it is not a claim of final model quality

## 5. What Not To Re-Litigate By Default

Unless new evidence appears, do not reopen these settled choices by default:

1. Human GT should remain its own asset layer.
2. `pipeline` should stay baseline-only.
3. Front/back should be described as a proxy, not physical distance.
4. COCO GT mask is acceptable as the current contour baseline.
5. The canonical GT workflow now lives under `main.py annotate` and
   `main.py review`.
6. User-facing prediction and GT checking should now be taught as
   `main.py predict` and `main.py validate`, not task-centric `evaluate`.

## 6. Archived GT Asset Design Note

This section archives the essential decisions from the earlier standalone
`ground_truth_asset_design.md` note. The original design intent has now been
absorbed into the main docs set, so only the highest-signal decisions are kept
here.

### 6.1 Problem the design was solving

- COCO does not provide animal eye GT.
- COCO does not provide in-image front/back ordering GT.
- repeated manual labeling inside each evaluator would create duplicated work
  and non-reusable results.

### 6.2 Core design decision

Split the system into four layers:

- `A` Dataset Asset
- `B` Human Ground Truth
- `C` Prediction
- `D` Evaluation

Target relationship:

- baseline reporting can run with `A + C`
- GT-based validation runs with `A + B + C`

### 6.3 Why GT had to be its own asset layer

- manual labels should be created once and reused across experiments
- GT must not be overwritten by prediction output
- evaluation should consume GT, not own the labeling process
- all reusable joins should key on:
  - `dataset_id`
  - `image_id`
  - `annotation_id`

### 6.4 Asset layout that came out of this design

```text
assets/
├── datasets/<dataset_id>/
│   ├── manifest.json
│   └── instances.csv
├── ground_truth/<dataset_id>/
│   ├── human_labels.csv
│   └── meta.json
└── predictions/<run_id>/
    ├── run_meta.json
    ├── localization.csv
    ├── measurement_instances.csv
    └── measurement_pairs.csv
```

Practical outcome:

- dataset and GT layers were implemented first
- prediction assetization was later completed for the current project scope
- the original design note is still archived here because it explains why the
  system was layered as `A + B + C + D`

### 6.5 GT schema decisions worth preserving

Per-object GT rows should carry at least:

- `dataset_id`
- `image_id`
- `annotation_id`
- `category`
- `left_eye_x`, `left_eye_y`
- `right_eye_x`, `right_eye_y`
- `depth_rank`
- `label_status`
- `annotator`
- `labeled_at`

Important convention:

- `depth_rank = 1` means closest to camera

### 6.6 Annotation tool constraints that shaped implementation

The design intentionally required:

- terminal-first interaction
- no GUI framework dependency
- optional `cv2.imshow()`
- immediate per-object save
- support for `skip`, `quit`, and `redo`

This directly led to the current terminal-first annotation workflow, exposed
canonically through `main.py annotate` and `main.py review`.

### 6.7 Runtime join model

Evaluation was designed around these joins:

- dataset to GT by `dataset_id + image_id + annotation_id`
- dataset to prediction by `dataset_id + image_id + annotation_id`
- pairwise ranking by image-scoped annotation pairs

This is the conceptual basis for the current GT-aware accuracy path.

### 6.8 What changed since the original draft

The original note proposed several future-facing pieces that are still not
fully realized:

- a fully assetized `predictions/` layer
- broader GT merge utilities as independent modules
- richer evaluator auto-discovery around GT availability

Those items should now be treated as future work, not as current architecture.
