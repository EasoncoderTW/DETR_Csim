CC := gcc
# Detect if 'uv' is available
PYTHON := $(shell command -v uv >/dev/null 2>&1 && echo 'uv run' || echo 'python3')
VALGRIND := valgrind

# directorys
OUTPUT_DIR := ./output
CSIM_DIR := ./Csim
PYTHON_DIR := ./Python
INPUT_DIR := ./model_bundle
IMAGE_DIR := ./demo_images
LOG_DIR := ./log

CSIM_OUTPUT_DIR := $(OUTPUT_DIR)/Csim
CSIM_DEBUG_DIR := $(OUTPUT_DIR)/Csim/debug
PYTHON_OUTPUT_DIR := $(OUTPUT_DIR)/python

# files
OUT_BOXES_BIN := $(CSIM_OUTPUT_DIR)/model_output_boxes.bin
OUT_SCORES_BIN := $(CSIM_OUTPUT_DIR)/model_output_scores.bin

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

# config
CFLAGS := -I$(CSIM_DIR)/include
LDFLAGS := -lm
ELF_NAME := detr

ifeq ($(DEBUG), 1)
CFLAGS += -DDEBUG
endif

ifeq ($(DUMP), 1)
CFLAGS += -DDUMP -DDUMP_TENSOR_DIR='"$(CSIM_DEBUG_DIR)/"'
endif

ifeq ($(ANALYZE), 1)
CFLAGS += -DANALYZE -DSTATISTICS_CSV_FILENAME='"$(STATISTIC_CSV)"'
endif

.PHONY: all py_gen_weights py_inference csim_verify build run clean_csim clean_python clean debug

all:
	make py_gen_weights
	make py_inference
	make build DEBUG=1 DUMP=1 ANALYZE=1
	make run
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

# plot statistic result
py_analyze:
	$(PYTHON) $(PYTHON_DIR)/Analyzer.py \
		--input_csv $(STATISTIC_CSV) \
		--out_dir $(OUTPUT_DIR)

# Csim
debug:
	make build DEBUG=1 DUMP=1 ANALYZE=1

build:
	$(CC) \
	$(CFLAGS) \
	$(CSIM_DIR)/detr.c $(CSIM_DIR)/src/*.c \
	-o $(ELF_NAME) \
	$(LDFLAGS)

run:
	mkdir -p $(CSIM_OUTPUT_DIR)
	mkdir -p $(CSIM_DEBUG_DIR)
	mkdir -p $(LOG_DIR)

	./$(ELF_NAME) $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) 1> $(LOG_DIR)/output.log 2> $(LOG_DIR)/debug.log

valgrind:
	mkdir -p $(CSIM_OUTPUT_DIR)
	mkdir -p $(CSIM_DEBUG_DIR)
	mkdir -p $(LOG_DIR)

	$(VALGRIND) --tool=massif \
	--log-file=$(LOG_DIR)/$(ELF_NAME)_massif.log \
	--heap=yes \
	--time-unit=i \
	--detailed-freq=1 \
	--max-snapshots=1000 \
	--ignore-fn=fopen --ignore-fn=fread --ignore-fn=fwrite --ignore-fn=fclose \
	--massif-out-file=$(LOG_DIR)/massif.out.%p_$(ELF_NAME) \
	./$(ELF_NAME) $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) 1> $(LOG_DIR)/output.log 2> $(LOG_DIR)/debug.log

clean_csim:
	rm $(ELF_NAME)
	rm -rf $(CSIM_OUTPUT_DIR)

clean_python:
	rm -rf $(PYTHON_OUTPUT_DIR)

clean: clean_csim clean_python
	rm -rf $(LOG_DIR)
	rm -rf $(OUTPUT_DIR)
	rm -f *.log

# other targets
%:
	@