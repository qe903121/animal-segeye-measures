# Animal SegEye Measures

Animal SegEye Measures is a reproducible baseline pipeline for animal-image metrology on COCO.

Current baseline scope:

- filter COCO images that contain at least two animals from two target categories
- use COCO `instance mask` as the current contour baseline
- localize animal eyes with either a CV baseline or an AI baseline
- measure per-animal inter-eye distance in pixels
- estimate pairwise front/back relationship as a monocular relative-depth proxy
- validate predictions against reusable human ground truth when available

The current baseline prioritizes end-to-end reproducibility, clear asset boundaries, and measurable validation over solving every subproblem with a fully autonomous learned pipeline.

## What The Project Solves

The project is organized around four asset layers:

- `A` Dataset Asset: frozen dataset membership exported from COCO filtering
- `B` Human GT Asset: reusable manual labels for left/right eyes and `depth_rank`
- `C` Prediction Asset: saved localization and measurement outputs by `run_id`
- `D` Validation Report: GT-based reporting and accuracy analysis

Important scope boundaries:

- contours currently come from COCO `instance mask`, not from this repo's own segmentation inference
- the AI localization path is `GT bbox + top-down MMPose on CPU`
- front/back output is a relative-depth proxy, not a physical 3D distance

## Tech Stack

- Python 3.10
- OpenCV
- NumPy / pandas
- PyYAML
- pycocotools
- OpenMMLab MMPose / MMCV
- Docker / VS Code Dev Container

## Repository Entry Point

The canonical CLI entry point is:

```bash
python main.py --config config/config.yaml [--verbose] <command> [command args]
```

Available sub-commands:

- `data`
- `annotate`
- `review`
- `predict`
- `validate`

Run help:

```bash
python main.py --help
python main.py data --help
python main.py annotate --help
python main.py review --help
python main.py predict --help
python main.py validate --help
```

## Local Setup

### Option 1: Local Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install "numpy<2.0.0" cython "setuptools<70.0.0"
pip install chumpy --no-build-isolation
pip install --no-build-isolation --no-binary xtcocotools xtcocotools
pip install -r requirements-ai.txt
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1/index.html
```

### Option 2: Dev Container

Use the existing Dev Container configuration:

- [devcontainer.json](/workspace/.devcontainer/devcontainer.json)
- [Dockerfile](/workspace/.devcontainer/Dockerfile)

In VS Code:

1. Open the repository.
2. Run `Dev Containers: Reopen in Container`.

### Option 3: Plain Docker

Build:

```bash
docker build -t animal-segeye-dev -f .devcontainer/Dockerfile .
```

Run:

```bash
docker run -it --rm --ipc=host -v "$(pwd)":/workspace -w /workspace animal-segeye-dev bash
```

## Typical Workflow

### 1. Create or refresh a Dataset Asset

```bash
python main.py data --skip-download
```

Useful variants:

```bash
python main.py data --categories cat dog --skip-download
python main.py data --visualize 5 --skip-download
python main.py data --visualize-all --skip-download
```

This produces:

- `output/test_samples.csv`
- `assets/datasets/<dataset_id>/manifest.json`
- `assets/datasets/<dataset_id>/instances.csv`

### 2. Create Human GT labels

Annotate:

```bash
python main.py annotate --dataset-id <dataset_id> --annotator hsien --skip-labeled --no-imshow
```

Review saved GT overlays:

```bash
python main.py review --dataset-id <dataset_id> --no-imshow
```

This produces:

- `assets/ground_truth/<dataset_id>/human_labels.csv`
- `assets/ground_truth/<dataset_id>/meta.json`

### 3. Run prediction from a frozen dataset

Prediction side:

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download
```

This command:

- runs eye localization
- runs measurement generation
- exports a formal Prediction Asset

Optional explicit run id:

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download --run-id demo_run
```

If `demo_run` already exists and you intentionally want to replace it:

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download --run-id demo_run --overwrite
```

This produces:

- `assets/predictions/<run_id>/run_meta.json`
- `assets/predictions/<run_id>/localization.csv`
- `assets/predictions/<run_id>/measurement_instances.csv`
- `assets/predictions/<run_id>/measurement_pairs.csv`

### 4. Validate a saved Prediction Asset against Human GT

```bash
python main.py validate --dataset-id <dataset_id> --prediction-run-id <run_id>
```

This command:

- requires frozen Dataset Asset + Human GT + Prediction Asset
- validates directly from saved assets; it does not require COCO download checks
- does not rerun detector inference
- outputs GT-based validation reports such as:
  - `NME`
  - `RDE`
  - `Pairwise Accuracy`

## Output Artifacts

### Dataset

- `output/test_samples.csv`
- `assets/datasets/<dataset_id>/manifest.json`
- `assets/datasets/<dataset_id>/instances.csv`

### Ground Truth

- `assets/ground_truth/<dataset_id>/human_labels.csv`
- `assets/ground_truth/<dataset_id>/meta.json`

### Prediction

- `assets/predictions/<run_id>/run_meta.json`
- `assets/predictions/<run_id>/localization.csv`
- `assets/predictions/<run_id>/measurement_instances.csv`
- `assets/predictions/<run_id>/measurement_pairs.csv`

### Evaluation

- `output/predict/...`
- `output/validate/...`
- `output/data/...`
- `output/review_labels/...`

## Methodology Summary

### Contours

- current source: COCO `instance mask`
- current role: reproducible contour baseline
- not yet: repo-native learned segmentation inference

### Eye Localization

- CV baseline: heuristic traditional image-processing approach
- AI baseline: MMPose top-down animal pose, using GT bbox and CPU inference

### Measurement

Inter-eye distance:

```text
d_eye = sqrt((x_left - x_right)^2 + (y_left - y_right)^2)
```

Front/back proxy:

- uses apparent eye-distance scale as monocular depth cue
- should be described as relative ordering / proxy gap, not physical distance

### Accuracy Metrics

- `NME`: normalized eye localization error with unordered eye-pair matching
- `RDE`: relative error of inter-eye distance
- `Pairwise Accuracy`: correctness of front/back ordering against GT `depth_rank`

## Architecture Diagram

See:

- [system_architecture.md](/workspace/system_architecture.md)
- [docs/01_architecture.md](/workspace/docs/01_architecture.md)

## Current Sample Asset In Repo

Committed sample dataset asset:

- `assets/datasets/coco_val2017_cat-dog_23714276/manifest.json`
- `assets/datasets/coco_val2017_cat-dog_23714276/instances.csv`

Committed sample GT:

- `assets/ground_truth/coco_val2017_cat-dog_23714276/human_labels.csv`
- `assets/ground_truth/coco_val2017_cat-dog_23714276/meta.json`

## API / Accounts

This repository is currently CLI-first.

- API service: not implemented in the current baseline
- API docs link: not applicable yet
- test account info: not applicable

## Known Limits

- contour source is still COCO GT mask
- AI path still depends on GT bbox top-down inference
- front/back is still a monocular proxy
- ONNX backend is planned but not integrated into the main path yet
- learned segmentation is still roadmap work, not the current contour source
