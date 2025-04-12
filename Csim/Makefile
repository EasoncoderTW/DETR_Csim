.PHONY: all clean
all: build

build:
	gcc -o detr ./detr.c ./src/model.c -I./include -lm

clean:
	rm -f detr