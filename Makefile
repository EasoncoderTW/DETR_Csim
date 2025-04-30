CC := gcc
PYTHON := python3

# directorys
OUTPUT_DIR := ./output
CSIM_DIR := ./Csim
PYTHON_DIR := ./Python
INPUT_DIR := ./model_bundle
IMAGE_DIR := ./demo_images

CSIM_OUTPUT_DIR := $(OUTPUT_DIR)/Csim
CSIM_DEBUG_DIR := $(OUTPUT_DIR)/Csim/debug
PYTHON_OUTPUT_DIR := $(OUTPUT_DIR)/python

# files
OUT_BOXES_BIN := $(CSIM_OUTPUT_DIR)/model_output_boxes.bin
OUT_SCORES_BIN := $(CSIM_OUTPUT_DIR)/model_output_scores.bin

INPUT_BIN := $(INPUT_DIR)/model_input.bin
WEIGHT_BIN := $(INPUT_DIR)/detr_weight.bin
CONFIG_JSON := $(INPUT_DIR)/config.json

# model
MODEL_REF := facebookresearch/detr:main
MODEL_NAME := detr_resnet50

# inference image
TARGET_IMGAE = $(word 2, $(MAKECMDGOALS))
ifeq ($(TARGET_IMGAE),)
TARGET_IMGAE = $(addprefix $(IMAGE_DIR)/, $(shell ls $(IMAGE_DIR) | grep -E '\.(jpg|jpeg|png)' | head -n 1))
endif

# config
CFLAGS := -I$(CSIM_DIR)/include
LDFLAGS := -lm
ELF_NAME := detr

.PHONY: py_gen_weights py_inference csim_verify debug test build run clean all

all:
	make py_gen_weights
	make py_inference
	make debug_run
	make csim_verify

# python script for model weight generation
py_gen_weights:
	mkdir -p $(INPUT_DIR)
	mkdir -p $(OUTPUT_DIR)

	$(PYTHON) $(PYTHON_DIR)/ModelWeight.py \
		--repo_or_dir '$(MODEL_REF)'\
		--model '$(MODEL_NAME)'\
		--output '$(WEIGHT_BIN)'\
		--list true

# python script for model input generation and output verification
py_inference:
	mkdir -p $(INPUT_DIR)
	mkdir -p $(PYTHON_OUTPUT_DIR)

	$(PYTHON) $(PYTHON_DIR)/ModelInference.py\
		--repo_or_dir '$(MODEL_REF)'\
		--model '$(MODEL_NAME)'\
		--bin_output '$(PYTHON_OUTPUT_DIR)'\
		--image_path '$(TARGET_IMGAE)'\
		--verbose

	cp $(PYTHON_OUTPUT_DIR)/model_input.bin $(INPUT_BIN)

# verify the model output
csim_verify:
	$(PYTHON) $(PYTHON_DIR)/CsimVerify.py\
		--csim $(CSIM_DEBUG_DIR) \
		--golden $(PYTHON_OUTPUT_DIR) \
		--data_type 'fp32' \
		--rtol 1e-6 \
		--atol 1e-4
# Csim
debug:
	$(CC) -DDEBUG -DDUMP_TENSOR_DIR='"$(CSIM_DEBUG_DIR)/"' -O0 \
	-o $(ELF_NAME) \
	$(CSIM_DIR)/detr.c $(CSIM_DIR)/src/model.c \
	$(CFLAGS) \
	$(LDFLAGS)

release:
	$(CC) \
	-o $(ELF_NAME) \
	$(CSIM_DIR)/detr.c $(CSIM_DIR)/src/model.c \
	$(CFLAGS) \
	$(LDFLAGS)

debug_run: debug
	mkdir -p $(CSIM_OUTPUT_DIR)
	mkdir -p $(CSIM_DEBUG_DIR)
	./$(ELF_NAME) $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) 1> output.log 2> debug.log

run: release
	mkdir -p $(CSIM_OUTPUT_DIR)
	./$(ELF_NAME) $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) | tee output.log

clean_csim:
	rm -rf $(CSIM_OUTPUT_DIR)

clean_python:
	rm -rf $(PYTHON_OUTPUT_DIR)

clean: clean_csim clean_python
	rm -rf $(OUTPUT_DIR)
	rm -f $(ELF_NAME)
	rm -f *.log

# other targets
%:
	@