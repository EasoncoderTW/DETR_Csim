INPUT_BIN := ../Python/bin/model_input.bin
OUT_BOXES_BIN := ./model_output_boxes.bin
OUT_SCORES_BIN := ./model_output_scores.bin
WEIGHT_BIN := ./detr_weight.bin
CONFIG_JSON := config.json
# This Makefile is for compiling the C code for the DETR model simulation.

.PHONY: build run debug clean

debug:
	gcc -DDEBUG -O0 -o detr ./detr.c ./src/model.c -I./include -lm
	mkdir -p ./debug

test:
	./detr $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN) 2>&1 | tee output.log

build:
	gcc -o detr ./detr.c ./src/model.c -I./include -lm

run:
	./detr $(CONFIG_JSON) $(WEIGHT_BIN) $(INPUT_BIN) $(OUT_BOXES_BIN) $(OUT_SCORES_BIN)

valgrind:
	valgrind --leak-check=full --error-limit=no ./detr config.json ./detr_weight.bin 2>&1 | tee output.log

clean:
	rm -f detr *.log *.o
	rm -r ./debug
