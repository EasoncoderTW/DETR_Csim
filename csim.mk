CC := gcc
VALGRIND := valgrind

# directories
OUTPUT_DIR := ./output
CSIM_DIR := ./Csim
INPUT_DIR := ./model_bundle
LOG_DIR := ./log

CSIM_OUTPUT_DIR := $(OUTPUT_DIR)/Csim
CSIM_DEBUG_DIR := $(OUTPUT_DIR)/Csim/debug

# files
OUT_BOXES_BIN := $(CSIM_OUTPUT_DIR)/model_output_boxes.bin
OUT_SCORES_BIN := $(CSIM_OUTPUT_DIR)/model_output_scores.bin
INPUT_BIN := $(INPUT_DIR)/model_input.bin
WEIGHT_BIN := $(INPUT_DIR)/detr_weight.bin
CONFIG_JSON := $(INPUT_DIR)/config.json

# config
CFLAGS := -I$(CSIM_DIR)/include -O3 -Wall
LDFLAGS := -lm
ELF_NAME := detr

# default flags
ifeq ($(DEBUG), 1)
CFLAGS += -DDEBUG
endif

ifeq ($(DUMP), 1)
CFLAGS += -DDUMP -DDUMP_TENSOR_DIR='"$(CSIM_DEBUG_DIR)/"'
endif

ifeq ($(ANALYZE), 1)
CFLAGS += -DANALYZE -DSTATISTICS_CSV_FILENAME='"$(OUTPUT_DIR)/statistics.csv"'
endif

ifeq ($(SOLE), 1)
CFLAGS += -DSOFTMAX_METHOD=SOFTMAX_SOLE
CFLAGS += -DLAYERNORM_METHOD=LAYERNORM_SOLE
endif

ifeq ($(FP16), 1)
CFLAGS += -DFP16
endif

.PHONY: build run debug valgrind clean_csim help

debug: # build and run Csim with debug options
	make -f $(lastword $(MAKEFILE_LIST)) build DEBUG=1 DUMP=1 ANALYZE=1

build: # build Csim executable
	mkdir -p $(CSIM_OUTPUT_DIR)
	$(CC) \
	$(CFLAGS) \
	$(CSIM_DIR)/detr.c $(CSIM_DIR)/src/*.c \
	-o $(ELF_NAME) \
	$(LDFLAGS)

run: # run Csim executable
	mkdir -p $(CSIM_OUTPUT_DIR)
	mkdir -p $(CSIM_DEBUG_DIR)
	mkdir -p $(LOG_DIR)

	./$(ELF_NAME) $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) 1> $(LOG_DIR)/output.log 2> $(LOG_DIR)/debug.log

valgrind: # run Csim executable with Valgrind massif tool
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

clean_csim: # clean Csim output and executable
	rm -f $(ELF_NAME)
	rm -rf $(CSIM_OUTPUT_DIR)

help: # display help message
	@echo "\033[1;36mC compilation targets:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[1;32m  %-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "\033[1;36mAvailable flags:\033[0m"
	@echo "\033[1;35m  DEBUG=1         \033[0m Enable debug mode"
	@echo "\033[1;35m  DUMP=1          \033[0m Enable tensor dumping to $(CSIM_DEBUG_DIR)"
	@echo "\033[1;35m  ANALYZE=1       \033[0m Enable analysis and generate statistics.csv"
	@echo "\033[1;35m  SOLE=1          \033[0m Use SOLE methods for SOFTMAX and LAYERNORM"
	@echo "\033[1;35m  FP16=1          \033[0m Enable FP16 precision"
	@echo ""
	@echo "\033[1;36mExample usage:\033[0m"
	@echo "  make -f csim.mk build DEBUG=1 DUMP=1"
	@echo "  make -f csim.mk debug  # equivalent to DEBUG=1 DUMP=1 ANALYZE=1"