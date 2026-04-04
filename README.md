# Animal SegEye Measures

Animal SegEye Measures is a reproducible baseline pipeline for animal-image metrology on COCO.

It is organized around five operator commands:

- `data`
- `annotate`
- `review`
- `predict`
- `validate`

The current baseline focuses on:

- filtering COCO images that contain at least two animals from two target categories
- using COCO `instance mask` as the current contour baseline
- localizing animal eyes with either a CV baseline or an AI baseline
- measuring per-animal inter-eye distance in pixels
- estimating pairwise front/back relationship as a monocular relative-depth proxy
- validating saved predictions against reusable human ground truth

## Scope Boundaries

- Contours currently come from COCO `instance mask`, not from repo-native segmentation inference.
- The AI localization path currently uses `GT bbox + top-down MMPose on CPU`.
- Front/back output is a relative-depth proxy, not a physical 3D distance.

For stable formulas, asset contracts, and methodology details, see:

- [docs/01_architecture.md](./docs/01_architecture.md)
- [system_architecture.md](./system_architecture.md)

## Tech Stack

- Python 3.10
- OpenCV
- NumPy / pandas
- PyYAML
- pycocotools
- OpenMMLab MMPose / MMCV
- Docker / VS Code Dev Container

## Canonical Entry Point

```bash
python main.py --config config/config.yaml [--verbose] <command> [args]
```

Document boundary:

- this file is the primary owner of setup and operator workflow
- stable formulas and runtime contracts live in `docs/01_architecture.md`
- current status, TODOs, and roadmap live in `docs/02_active_context.md`

Help:

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

- [devcontainer.json](./.devcontainer/devcontainer.json)
- [Dockerfile](./.devcontainer/Dockerfile)

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

### 1. Build a Dataset Asset

```bash
python main.py data --skip-download
```

Useful variants:

```bash
python main.py data --categories cat dog --skip-download
python main.py data --visualize 5 --skip-download
python main.py data --visualize-all --skip-download
```

### 2. Create or update Human GT

Annotate:

```bash
python main.py annotate --dataset-id <dataset_id> --annotator hsien --skip-labeled --no-imshow
```

Review saved overlays:

```bash
python main.py review --dataset-id <dataset_id> --no-imshow
```

### 3. Generate a Prediction Asset

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download
```

With an explicit run id:

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download --run-id demo_run
```

Overwrite only when intentional:

```bash
python main.py predict --dataset-id <dataset_id> --method ai --skip-download --run-id demo_run --overwrite
```

### 4. Validate against Human GT

```bash
python main.py validate --dataset-id <dataset_id> --prediction-run-id <run_id>
```

This user-facing validation path:

- requires Dataset Asset + Human GT + Prediction Asset
- does not rerun detector inference
- does not require raw COCO reload

## Output Artifacts

### Dataset Asset

- `output/test_samples.csv`
- `assets/datasets/<dataset_id>/manifest.json`
- `assets/datasets/<dataset_id>/instances.csv`

### Human GT Asset

- `assets/ground_truth/<dataset_id>/human_labels.csv`
- `assets/ground_truth/<dataset_id>/meta.json`

### Prediction Asset

- `assets/predictions/<run_id>/run_meta.json`
- `assets/predictions/<run_id>/localization.csv`
- `assets/predictions/<run_id>/measurement_instances.csv`
- `assets/predictions/<run_id>/measurement_pairs.csv`

### Reports And Overlays

- `output/data/...`
- `output/review_labels/...`
- `output/predict/...`
- `output/validate/...`

## References

- [docs/01_architecture.md](./docs/01_architecture.md)
- [docs/02_active_context.md](./docs/02_active_context.md)
- [docs/03_dev_journal.md](./docs/03_dev_journal.md)
- [system_architecture.md](./system_architecture.md)

## API / Accounts

This repository is currently CLI-first.

- API service: not implemented
- API docs link: not applicable yet
- test account info: not applicable

## Known Limits

- contour source is still COCO GT mask
- AI path still depends on GT bbox top-down inference
- front/back is still a monocular proxy
- ONNX backend is planned but not integrated into the main path yet
- learned segmentation is still roadmap work, not the current contour source
