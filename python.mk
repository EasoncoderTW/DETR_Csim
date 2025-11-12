# Detect if 'uv' is available
PYTHON := $(shell command -v uv >/dev/null 2>&1 && echo 'uv run' || echo 'python3')

# directories
OUTPUT_DIR := ./output
PYTHON_DIR := ./Python
INPUT_DIR := ./model_bundle
IMAGE_DIR := ./demo_images

PYTHON_OUTPUT_DIR := $(OUTPUT_DIR)/python
CSIM_DEBUG_DIR := $(OUTPUT_DIR)/Csim/debug

# files
INPUT_BIN := $(INPUT_DIR)/model_input.bin
WEIGHT_BIN := $(INPUT_DIR)/detr_weight.bin
CONFIG_JSON := $(INPUT_DIR)/config.json
STATISTIC_CSV := $(OUTPUT_DIR)/statistics.csv

# model
MODEL_REF := facebookresearch/detr:main
MODEL_NAME := detr_resnet50

# inference image
TARGET_IMGAE = $(word 2, $(MAKECMDGOALS))
ifeq ($(TARGET_IMGAE),)
TARGET_IMGAE = $(addprefix $(IMAGE_DIR)/, $(shell ls $(IMAGE_DIR) | grep -E '\.(jpg|jpeg|png)' | head -n 1))
endif

.PHONY: py_gen_weights py_inference csim_verify py_analyze clean_python visualize_detections help

py_gen_weights: # python script for model weight generation
	mkdir -p $(INPUT_DIR)
	mkdir -p $(OUTPUT_DIR)

	$(PYTHON) $(PYTHON_DIR)/detr_tools.py model_weight \
		--repo_or_dir '$(MODEL_REF)'\
		--model '$(MODEL_NAME)'\
		--output '$(WEIGHT_BIN)'\
		--list true

py_inference: # python script for model input generation and output verification
	mkdir -p $(INPUT_DIR)
	mkdir -p $(PYTHON_OUTPUT_DIR)

	$(PYTHON) $(PYTHON_DIR)/detr_tools.py model_inference\
		--repo_or_dir '$(MODEL_REF)'\
		--model '$(MODEL_NAME)'\
		--bin_output '$(PYTHON_OUTPUT_DIR)'\
		--image_path '$(TARGET_IMGAE)'\
		--verbose

	cp $(PYTHON_OUTPUT_DIR)/model_input.bin $(INPUT_BIN)

csim_verify: # verify the model output
	$(PYTHON) $(PYTHON_DIR)/detr_tools.py csim_verify \
		--csim $(CSIM_DEBUG_DIR) \
		--golden $(PYTHON_OUTPUT_DIR) \
		--data_type 'fp32' \
		--rtol 1e-4 \
		--atol 1e-3 \
		--statistic

py_analyze: # plot statistic result
	$(PYTHON) $(PYTHON_DIR)/detr_tools.py analyzer \
		--input_csv $(STATISTIC_CSV) \
		--out_dir $(OUTPUT_DIR)

clean_python: # clean Python output
	rm -rf $(PYTHON_OUTPUT_DIR)

visualize_detections: # visualize csim detection results
	$(PYTHON) $(PYTHON_DIR)/detr_tools.py visualize_detections \
		--image_path '$(TARGET_IMGAE)' \
		--output_path '$(OUTPUT_DIR)/Csim/detection_output.png' \
		--score_path '$(OUTPUT_DIR)/Csim/model_output_scores.bin' \
		--boxes_path '$(OUTPUT_DIR)/Csim/model_output_boxes.bin' \
		--threshold 0.7

help: # display help message
	@echo "\033[1;34mPython-related targets:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[1;32m  %-20s\033[0m %s\n", $$1, $$2}'

# other targets
%:
	@