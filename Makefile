.DEFAULT_GOAL := help

PYTHON ?= python
CONFIG ?= config/config.yaml
MAIN := $(PYTHON) main.py --config $(CONFIG)

DATASET_ID ?= coco_val2017_cat-dog_23714276
DATA_CATEGORIES ?= cat dog

REVIEW_OUTPUT_DIR ?= output/review_labels_smoke

AI_RUN_ID ?= predict_ai_run
CV_RUN_ID ?= predict_cv_run

PREDICT_AI_OUTPUT ?= output/predict_ai
PREDICT_CV_OUTPUT ?= output/predict_cv
VALIDATE_AI_OUTPUT ?= output/validate_ai
VALIDATE_CV_OUTPUT ?= output/validate_cv

.PHONY: help \
	data review \
	predict-ai predict-cv \
	validate-ai validate-cv \
	full full-ai full-cv examiner

help:
	@echo "Available targets:"
	@echo "  make data         # Build the sample Dataset Asset"
	@echo "  make review       # Render committed GT overlays"
	@echo "  make predict-ai   # Generate AI Prediction Asset"
	@echo "  make predict-cv   # Generate CV Prediction Asset"
	@echo "  make validate-ai  # Validate the AI Prediction Asset"
	@echo "  make validate-cv  # Validate the CV Prediction Asset"
	@echo "  make full         # data -> review -> predict-ai -> validate-ai"
	@echo "  make full-cv      # data -> review -> predict-cv -> validate-cv"
	@echo "  make examiner     # data -> review -> predict-ai/cv -> validate-ai/cv"
	@echo ""
	@echo "Override variables if needed, for example:"
	@echo "  make full DATASET_ID=$(DATASET_ID)"

data:
	$(MAIN) data --categories $(DATA_CATEGORIES) --visualize-all

review:
	$(MAIN) review --dataset-id $(DATASET_ID) --no-imshow --review-output-dir $(REVIEW_OUTPUT_DIR)

predict-ai:
	$(MAIN) predict --dataset-id $(DATASET_ID) --method ai --run-id $(AI_RUN_ID) --output-dir $(PREDICT_AI_OUTPUT) --overwrite

predict-cv:
	$(MAIN) predict --dataset-id $(DATASET_ID) --method cv --run-id $(CV_RUN_ID) --output-dir $(PREDICT_CV_OUTPUT) --overwrite

validate-ai:
	$(MAIN) validate --dataset-id $(DATASET_ID) --prediction-run-id $(AI_RUN_ID) --output-dir $(VALIDATE_AI_OUTPUT)

validate-cv:
	$(MAIN) validate --dataset-id $(DATASET_ID) --prediction-run-id $(CV_RUN_ID) --output-dir $(VALIDATE_CV_OUTPUT)

full: full-ai

full-ai: data review predict-ai validate-ai

full-cv: data review predict-cv validate-cv

examiner: data review predict-ai predict-cv validate-ai validate-cv
