# ONNX Migration Plan

Last updated: 2026-04-04

## 1. Purpose

This document defines the next-step plan for introducing an ONNX Runtime
backend for the current AI localization path.

Current intent:

- keep the existing `predict -> validate` contract stable
- keep the current Prediction Asset schema stable
- replace only the AI inference backend, not the surrounding asset model
- preserve the current `GT bbox -> top-down pose -> EyeResult` contract

Current repo status:

- ONNX Runtime CPU backend is now implemented behind the existing
  `method=ai` detector contract
- backend selection is config-driven through
  `eye_detection.ai_model.runtime`
- the official ONNX artifact is user-fetched via
  `tools/fetch_rtmpose_onnx.py` and kept outside git history
- the current supported ONNX runtime path uses:
  - `CPUExecutionProvider`
- GPU deployment was explored but is currently deferred as an environment task,
  not part of the supported repo baseline

Recommended framing:

- do not describe this change as "remove MMPose"
- describe it as:
  - introduce ONNX Runtime as an alternative pose inference backend
  - benchmark parity and latency before changing the default runtime

## 2. Current Baseline In Repo

The current AI path is:

- `main.py predict`
- `src/localization/factory.py`
- `src/localization/detector_ai.py`
- `src/localization/detector_ai_onnx.py`

Current behavior:

- input:
  - image
  - GT bbox
- runtime options:
  - PyTorch CPU via:
    - `MMPoseInferencer`
    - `inference_topdown`
  - ONNX Runtime CPU via:
    - `CPUExecutionProvider`
- output:
  - `EyeResult`
    - `status`
    - `left_eye`
    - `right_eye`
    - `confidence`
    - `all_keypoints`
    - `all_scores`

Important baseline fact:

- this repo does not currently do full-scene detection in the AI path
- it does GT-bbox-anchored top-down keypoint inference

### Repo-verified model identity

Under the repo's pinned environment:

- `mmpose==1.3.2`
- `alias: "animal"` resolves to:
  - config:
    - `animal_2d_keypoint/rtmpose/ap10k/rtmpose-m_8xb64-210e_ap10k-256x256.py`
  - checkpoint:
    - `https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth`
- dataset / keypoint convention:
  - `AP-10K`
  - index `0 = left eye`
  - index `1 = right eye`

This means the repo is already using the same official AP-10K RTMPose pose
model family that OpenMMLab publishes for `animal`.

### Same model, different runtime path

Important distinction:

- the repo uses the same official `animal` pose estimator identity
- the repo does **not** use the full default `animal` inferencer pipeline as an
  operator contract
- the repo intentionally bypasses full-scene detection and runs:
  - image
  - caller-provided GT bbox
  - top-down pose inference

Therefore the ONNX migration target is:

- replace the **pose backend**
- keep the repo's current bbox-anchored orchestration intact
- do not redefine the project as detector+pose end-to-end inference

## 3. Research Summary

### What sits where in the stack

- application layer:
  - this repo owns dataset assets, GT assets, prediction assets, measurement,
    and validation
- toolkit layer:
  - MMPose currently provides the pose inference API and AP-10K keypoint
    convention
- OpenMMLab runtime layer:
  - MMEngine / MMCV provide shared infrastructure used by MMPose
- deep learning framework layer:
  - PyTorch CPU currently performs the actual tensor computation

### Why ONNX Runtime is feasible

- the current detector abstraction already isolates the AI backend behind
  `BaseEyeDetector`
- the rest of the system consumes `EyeResult`, not raw MMPose objects
- Prediction Asset generation and validation are already downstream of that
  stable detector contract
- OpenMMLab already publishes an official ONNX artifact for the same AP-10K
  RTMPose pose model family used by the repo

### Main technical caveat

The repo-level migration target is narrower than the default MMPose inferencer
story:

- OpenMMLab's `animal` alias resolves to a pose model plus a default detector
  story
- this repo only needs the pose model portion under a caller-provided GT bbox

So parity work must verify not only:

- exact model config
- exact checkpoint
- expected input size
- exact AP-10K keypoint ordering

but also:

- equivalent bbox crop / preprocess semantics
- equivalent SimCC decode behavior
- equivalent eye-point extraction and score-threshold behavior

Without this, parity checks against the current PyTorch path will be noisy and
hard to interpret even if the upstream model identity matches.

## 4. Recommended Migration Strategy

This section is now partly historical. For the current repo, Phases 1 to 3
have been completed for the CPU-only ONNX path, while the default-runtime
decision remains open.

### Phase 0: Pin the current model identity

Goal:

- turn the current `alias=animal` path into one reproducible model reference

Tasks:

1. record the exact MMPose config/checkpoint currently resolved by the alias
2. record the expected input size and preprocessing assumptions
3. record the AP-10K keypoint index mapping used by this repo
4. record the official ONNX artifact corresponding to that same pose model
5. make this identity explicit in config or migration notes

Deliverable:

- one pinned model identity that can be used for export and parity testing
- one explicit statement that the repo preserves `GT bbox + top-down pose`
  instead of adopting the default detector-driven inferencer path

### Phase 1: Export feasibility spike

Goal:

- prove that the pinned pose model can be exported and executed under ONNX
  Runtime

Recommended route:

- first verify the official published ONNX artifact for the pinned model
- use `MMDeploy` only if the official artifact is unsuitable for the repo's
  bbox-driven runtime contract or if export reproducibility must be reproduced
- avoid hand-rolling a `torch.onnx.export` pipeline unless export support
  forces it

Tasks:

1. verify whether the official published ONNX artifact can be used directly
2. if needed, prepare one export environment using MMDeploy + ONNX Runtime
3. export or unpack the pinned RTMPose/AP-10K model to ONNX
4. verify a single-image + single-bbox smoke inference
5. confirm the exported graph returns enough signal to reconstruct:
   - keypoints
   - scores

Deliverable:

- one ONNX artifact that can be loaded by ONNX Runtime

Current status:

- completed for the CPU-only path
- runtime selection now exists in `config/config.yaml`
- `src/localization/detector_ai_onnx.py` is in place
- `src/localization/factory.py` now switches between PyTorch and ONNX

### Phase 2: Add an ONNX backend to this repo

Goal:

- integrate ONNX Runtime without breaking the current CLI or asset contracts

Tasks:

1. extend config with backend selection
2. add one ONNX detector implementation
3. keep the current PyTorch implementation intact
4. let `factory.py` choose runtime backend under `method=ai`
5. preserve the current caller-provided GT bbox flow; do not reintroduce
   detector-driven full-scene inference

Recommended config extension:

```yaml
eye_detection:
  ai_model:
    runtime: "pytorch"   # or "onnx"
    alias: "animal"
    device: "cpu"
    score_threshold: 0.3
    onnx_model_path: "models/rtmpose_ap10k.onnx"
    providers:
      - "CPUExecutionProvider"
```

Recommended file additions:

- `src/localization/detector_ai_onnx.py`

Likely file updates:

- `src/localization/factory.py`
- `config/config.yaml`
- `requirements-ai.txt`

Non-goal:

- do not redesign `EyeResult`
- do not redesign the repo into detector+pose end-to-end automation

### Phase 3: Parity and benchmark

Goal:

- prove that ONNX Runtime is close enough to the current PyTorch path and
  provides operational value

Benchmark scope:

- same frozen Dataset Asset
- same GT bbox input
- same output asset schema
- same repo-side eye extraction contract from AP-10K keypoints `0/1`

Compare:

- predicted left/right eye coordinates
- `status`
- `confidence`
- downstream `NME`
- downstream `RDE`
- downstream pairwise ordering accuracy
- end-to-end `predict` wall-clock time

Acceptance questions:

- does ORT reduce CPU inference time enough to matter?
- does ORT preserve enough prediction parity for downstream validation?
- does the new backend keep the current `predict -> validate` contract intact?

Current benchmark snapshot on the frozen sample asset:

- end-to-end `predict` wall-clock:
  - PyTorch CPU:
    - about `5.25s`
  - ONNX Runtime CPU:
    - about `3.93s`
- `validate` summary:
  - PyTorch CPU:
    - `NME = 0.3910`
    - `RDE = 23.5%`
    - pairwise accuracy:
      - `66.7%`
  - ONNX Runtime CPU:
    - `NME = 0.4224`
    - `RDE = 23.5%`
    - pairwise accuracy:
      - `66.7%`

Interpretation:

- ONNX Runtime CPU is faster on the current frozen sample workflow
- validation parity is close but not numerically identical at the instance
  metric level
- the current repo still keeps PyTorch as the default runtime

### Phase 4: Default-runtime decision

Goal:

- decide whether ONNX becomes the default AI runtime

Promote ONNX to default only if:

- export is reproducible
- runtime integration is stable
- validation metrics do not regress materially
- CPU latency improvement is meaningful

Otherwise:

- keep PyTorch as default
- keep ONNX as optional backend

Current status:

- this is the current repo posture

## 5. File-Level Change Plan

### Minimal MVP change set

1. `config/config.yaml`
   - add `eye_detection.ai_model.runtime`
   - add ONNX model path / providers

2. `requirements-ai.txt`
   - add `onnxruntime`
   - document whether MMDeploy is required only for export or also for runtime

3. `src/localization/detector_ai_onnx.py`
   - implement `BaseEyeDetector.detect(...)`
   - own ONNX Runtime session initialization
   - own bbox-based inference path
   - return the same `EyeResult` contract

4. `src/localization/factory.py`
   - select `pytorch` or `onnx` runtime under `method="ai"`

5. optional tools
   - `tools/export_rtmpose_onnx.py`
   - `tools/fetch_rtmpose_onnx.sh`
   - or a dedicated export note/script

### Artifact handling recommendation

- do not commit the ONNX binary into the main git history by default
- keep the repo focused on:
  - config
  - downloader / fetch script
  - checksum
  - export notes
  - model identity metadata
- prefer:
  - official OpenMMLab download URL
  - or another documented external artifact location

## 6. Main Risks

1. Model identity drift
   - current alias-based loading is convenient but not a deployment contract

2. Export support mismatch
   - the exact RTMPose/AP-10K variant may need MMDeploy-specific handling

3. Pre/post-processing mismatch
   - if ORT runtime preprocessing or decode logic differs from the current
     MMPose path, downstream validation may drift

4. Runtime-path mismatch
   - an ONNX model can match the official pose weights but still diverge from
     the repo's current `GT bbox -> top-down pose` behavior if crop / bbox /
     decode handling changes

5. False optimization
   - ONNX may improve deployment portability more than actual runtime for a
     small CPU-only sample workflow

## 7. Effort Estimate

- export feasibility spike:
  - `0.5 ~ 1 day`
- repo MVP integration:
  - `2 ~ 4 days`
- stable delivery-ready version:
  - `4 ~ 7 days`

## 8. Acceptance Criteria

The ONNX migration can be considered successful for the repo when all of the
following are true:

1. `main.py predict` still produces the same Prediction Asset schema
2. `main.py validate` works unchanged against ONNX-produced prediction assets
3. AI backend can be switched by config without changing the CLI contract
4. the repo still uses the same `GT bbox + top-down pose` contract after the
   backend switch
5. parity and benchmark results are documented on the frozen sample dataset
6. README and architecture docs describe ONNX as either:
   - an optional backend
   - or the default backend after validation

## 9. Recommended Immediate Next Step

The original spike has already been completed for the current CPU-only ONNX
path. The next decision is narrower:

1. keep the ONNX CPU path documented and reproducible
2. decide later whether ONNX should become the default runtime
3. treat GPU enablement as a separate environment project if it becomes
   important

## 10. References

- ONNX Runtime Python API:
  - https://onnxruntime.ai/docs/api/python/api_summary.html
- MMDeploy official repository:
  - https://github.com/open-mmlab/mmdeploy
- MMDeploy model conversion guide:
  - https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/02-how-to-run/convert_model.md
- MMDeploy support for OpenMMLab codebases:
  - https://mmdeploy.readthedocs.io/en/v0.14.0/04-supported-codebases/mmpose.html
- MMPose inferencer alias mapping (`animal` -> AP-10K RTMPose):
  - https://mmpose.readthedocs.io/en/latest/user_guides/inference.html
- OpenMMLab RTMPose animal model zoo entry:
  - https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose#animal-2d-17-keypoints
- Official AP-10K RTMPose checkpoint used by the pinned repo environment:
  - https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.pth
- Official AP-10K RTMPose ONNX artifact:
  - https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip

## 11. Release-Ready Implementation Checklist

This section turns the migration plan into a delivery-oriented checklist for an
online-releaseable repo change.

### 11.1 P0 release blockers

1. ONNX pose backend must exist behind the current AI detector contract.
   - add `src/localization/detector_ai_onnx.py`
   - keep input contract:
     - image
     - GT bbox
     - same AP-10K eye index convention
   - keep output contract:
     - `EyeResult.status`
     - `left_eye`
     - `right_eye`
     - `confidence`
   - success condition:
     - `main.py predict --method ai` can run with `runtime=onnx` without any
       CLI contract change

2. Runtime selection must be explicit and stable.
   - update `config/config.yaml`
   - update `src/localization/factory.py`
   - recommended behavior:
     - `method=ai + runtime=pytorch` -> existing `detector_ai.py`
     - `method=ai + runtime=onnx` -> new `detector_ai_onnx.py`
   - success condition:
     - backend switching is config-driven and does not affect `predict` /
       `validate` usage

3. Model artifact handling must be releasable without committing binaries.
   - choose one canonical local cache path such as:
     - `models/rtmpose/rtmpose-m_ap10k.onnx`
   - add one fetch helper:
     - `tools/fetch_rtmpose_onnx.py`
     - or `tools/fetch_rtmpose_onnx.sh`
   - verify:
     - official URL
     - checksum or equivalent integrity metadata
   - add ignore rules for downloaded model artifacts:
     - `models/`
     - or a narrower ONNX-specific path
   - success condition:
     - a clean checkout can fetch the ONNX artifact deterministically

4. Environment setup must work in the supported container workflow.
   - update `requirements-ai.txt` with `onnxruntime`
   - ensure `.devcontainer/Dockerfile` still builds cleanly after dependency
     changes
   - document whether `mmdeploy` is:
     - not needed at runtime
     - export-only optional tooling
   - success condition:
     - a fresh devcontainer can execute the ONNX path using documented steps

5. Prediction Asset compatibility must remain unchanged.
   - confirm no schema change in:
     - `run_meta.json`
     - `localization.csv`
     - `measurement_instances.csv`
     - `measurement_pairs.csv`
   - add backend-identifying metadata if needed without breaking loaders:
     - for example inside `model_name` or `extra_meta`
   - success condition:
     - `main.py validate` consumes ONNX-produced prediction assets unchanged

6. Public docs must make the ONNX path operable for a new user.
   - update `README.md`
   - keep `onnx_migration_plan.md` as design / migration context
   - if ONNX becomes part of the supported repo story, reflect that in:
     - `docs/02_active_context.md`
     - optionally `docs/01_architecture.md` if the backend choice becomes a
       stable architectural fact
   - success condition:
     - a reviewer can discover:
       - how to fetch the model
       - how to enable ONNX
       - how to run `predict`
       - how to run `validate`
       - what limitations still remain

7. Benchmarks and parity results must be written down before release.
   - compare PyTorch vs ONNX on the same frozen dataset asset
   - record:
     - localization success rate
     - NME
     - RDE
     - pairwise accuracy
     - end-to-end `predict` time
   - success condition:
     - release notes and docs can justify why ONNX is:
       - optional
       - or the new default

### 11.2 Recommended implementation order

1. Pin final artifact metadata in docs and config.
   - exact official URL
   - expected local path
   - checksum
   - backend label convention for metadata

2. Add artifact fetch tooling before backend code.
   - this prevents the implementation from depending on manually downloaded
     local files

3. Implement `detector_ai_onnx.py`.
   - make it satisfy the same `BaseEyeDetector.detect(...)` contract
   - keep bbox-anchored top-down inference semantics

4. Wire runtime selection in `factory.py` and `config/config.yaml`.
   - keep `pytorch` as the safe initial default until parity is proven

5. Update dependency and container setup.
   - `requirements-ai.txt`
   - `.devcontainer/Dockerfile`
   - optionally supporting notes in `README.md`

6. Run smoke tests first, then full parity.
   - single sample bbox smoke test
   - full `predict`
   - full `validate`

7. Backfill benchmark and operator docs.
   - README usage
   - benchmark summary
   - limitations and fallback path

8. Make the default-runtime decision only after measured parity.

### 11.3 Concrete file checklist

Required code / config updates:

- `src/localization/detector_ai_onnx.py`
- `src/localization/factory.py`
- `config/config.yaml`
- `requirements-ai.txt`
- `.devcontainer/Dockerfile`
- `.gitignore`

Required tooling / support updates:

- `tools/fetch_rtmpose_onnx.py` or `tools/fetch_rtmpose_onnx.sh`
- optional `tools/export_rtmpose_onnx.py`

Required docs / release updates:

- `README.md`
- `onnx_migration_plan.md`
- `docs/02_active_context.md`
- optional `docs/01_architecture.md`
- optional `report.md` if it is the chosen benchmark summary surface

### 11.4 Release validation commands

The first public release should be blocked on running a command set like the
following successfully.

1. Fetch the model artifact.

```bash
python tools/fetch_rtmpose_onnx.py
```

2. Run the existing PyTorch baseline on the frozen sample dataset.

```bash
python main.py predict --dataset-id coco_val2017_cat-dog_23714276 --method ai --skip-download --run-id predict_ai_pytorch_release
python main.py validate --dataset-id coco_val2017_cat-dog_23714276 --prediction-run-id predict_ai_pytorch_release
```

3. Run the ONNX backend on the same frozen sample dataset.

```bash
python main.py predict --dataset-id coco_val2017_cat-dog_23714276 --method ai --skip-download --run-id predict_ai_onnx_release
python main.py validate --dataset-id coco_val2017_cat-dog_23714276 --prediction-run-id predict_ai_onnx_release
```

4. Compare exported assets and reported metrics.

Must confirm:

- prediction asset export succeeds for both backends
- `validate` succeeds unchanged for both backends
- ONNX does not change Prediction Asset schema
- ONNX preserves acceptable parity for:
  - eye coordinates
  - `status`
  - `confidence`
  - `NME`
  - `RDE`
  - pairwise ordering

### 11.5 Release outputs

The release should not be considered complete until all of the following exist:

- merged ONNX backend code path
- deterministic model fetch path
- benchmark comparison written into repo docs
- operator-facing README instructions
- explicit statement that the repo still uses:
  - `GT bbox + top-down pose`
  - not detector+pose end-to-end automation
- no ONNX binary committed into the normal git history by default

### 11.6 Explicitly out of scope for the first release

The first ONNX release should **not** expand scope into:

- replacing the dataset / GT / prediction asset model
- changing the user-facing command set
- redesigning `EyeResult`
- making full-scene detection part of the repo contract
- supporting every ONNX provider on day one
- making ONNX the default before parity is documented
- committing the ONNX model binary into the main repository history
