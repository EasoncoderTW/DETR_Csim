.PHONY: build run debug clean

debug:
	gcc -DDEBUG -O0 -o detr ./detr.c ./src/model.c -I./include -lm

build:
	gcc -o detr ./detr.c ./src/model.c -I./include -lm

run:
	./detr config.json ./detr_weight.bin 2>&1 | tee output.log

valgrind:
	valgrind --leak-check=full --error-limit=no ./detr config.json ./detr_weight.bin 2>&1 | tee output.log

clean:
	rm -f detr *.log *.o
