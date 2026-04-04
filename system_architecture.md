# System Architecture

Last updated: 2026-04-04

Document boundary:

- this file is the diagram-centric system summary
- operator workflow belongs in `README.md`
- stable formulas and runtime contracts belong in `docs/01_architecture.md`

## 1. End-to-End View

```text
COCO Source
   |
   v
main.py data
   |
   v
Dataset Asset (A)
assets/datasets/<dataset_id>/
├── manifest.json
└── instances.csv
   |
   +------------------------------+
   |                              |
   v                              v
main.py annotate              main.py predict
   |                              |
   v                              v
Human GT Asset (B)            Prediction Asset (C)
assets/ground_truth/          assets/predictions/<run_id>/
<dataset_id>/                 ├── run_meta.json
├── human_labels.csv          ├── localization.csv
└── meta.json                 ├── measurement_instances.csv
                              └── measurement_pairs.csv
   |                              |
   +--------------+---------------+
                  |
                  v
             main.py validate
                  |
                  v
         Validation Reports (D)
         output/validate/accuracy/
         ├── eval_accuracy_instances.csv
         └── eval_accuracy_pairs.csv
```

## 2. Reading Guide

- `A` = Dataset Asset
- `B` = Human GT Asset
- `C` = Prediction Asset
- `D` = Validation Reports

The canonical operator commands remain:

- `data`
- `annotate`
- `review`
- `predict`
- `validate`

See `README.md` for command examples and end-user workflow.

## 3. Layer Snapshot

### A. Dataset Asset
- COCO download checks
- dataset filtering
- standardized dataset membership
- frozen membership by `dataset_id`

### B. Human GT Asset

- `left_eye`
- `right_eye`
- `depth_rank`
- `label_status`
- reusable across repeated prediction runs

### C. Prediction Asset

- saved localization outputs
- saved measurement outputs
- prediction-side metadata via `run_id`
- immutable by default; explicit overwrite is required

### D. Validation Report

- Dataset Asset
- Human GT Asset
- Prediction Asset
- user-facing validation path does not rerun detector inference
- user-facing validation path does not require raw COCO reload

## 4. Internal Module Map

### CLI layer

- [main.py](/workspace/main.py)
- [cmd_data.py](/workspace/src/cli/cmd_data.py)
- [cmd_annotate.py](/workspace/src/cli/cmd_annotate.py)
- [cmd_review.py](/workspace/src/cli/cmd_review.py)
- [cmd_predict.py](/workspace/src/cli/cmd_predict.py)
- [cmd_validate.py](/workspace/src/cli/cmd_validate.py)

### Data / asset layer

- [downloader.py](/workspace/src/data/downloader.py)
- [loader.py](/workspace/src/data/loader.py)
- [asset_exporter.py](/workspace/src/data/asset_exporter.py)
- [asset_loader.py](/workspace/src/data/asset_loader.py)
- [gt_store.py](/workspace/src/data/gt_store.py)
- [prediction_store.py](/workspace/src/data/prediction_store.py)
- [prediction_loader.py](/workspace/src/data/prediction_loader.py)

### Prediction layer

- [builders.py](/workspace/src/prediction/builders.py)
- [detector_cv.py](/workspace/src/localization/detector_cv.py)
- [detector_ai.py](/workspace/src/localization/detector_ai.py)
- [detector_ai_onnx.py](/workspace/src/localization/detector_ai_onnx.py)
- [factory.py](/workspace/src/localization/factory.py)

### Evaluation layer

- [base.py](/workspace/src/evaluation/base.py)
- [engine.py](/workspace/src/evaluation/engine.py)
- [localization.py](/workspace/src/evaluation/localization.py)
- [measurement.py](/workspace/src/evaluation/measurement.py)
- [accuracy.py](/workspace/src/evaluation/accuracy.py)

## 5. Asset Directory View

### Dataset Asset

```text
assets/datasets/<dataset_id>/
├── manifest.json
└── instances.csv
```

### Human GT Asset

```text
assets/ground_truth/<dataset_id>/
├── human_labels.csv
└── meta.json
```

### Prediction Asset

```text
assets/predictions/<run_id>/
├── run_meta.json
├── localization.csv
├── measurement_instances.csv
└── measurement_pairs.csv
```

## 6. Boundary Notes

- Contours currently come from COCO `instance mask`, not learned segmentation inference.
- AI localization currently uses `GT bbox + top-down pose`, with either the
  default PyTorch/MMPose backend or the optional ONNX Runtime CPU backend.
- `measurement` belongs to prediction-side work, not GT validation.
- `front/back` is a monocular relative-depth proxy, not physical distance.
