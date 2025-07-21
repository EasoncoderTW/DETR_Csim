# directories
OUTPUT_DIR := ./output
LOG_DIR := ./log

.PHONY: all clean help

default: help # default target to display help message

all: # run the entire workflow
	make -f python.mk py_gen_weights
	make -f python.mk py_inference
	make -f csim.mk build DEBUG=1 DUMP=1 ANALYZE=1
	make -f csim.mk run
	make -f python.mk csim_verify

clean: # clean all output and logs
	make -f python.mk clean_python
	make -f csim.mk clean_csim
	rm -rf $(LOG_DIR)
	rm -f *.log

help: # display help message
	@echo "\033[1;34mMain targets:\033[0m"
	@grep -E '^[a-zA-Z_-]+:.*?#' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?#"}; {printf "\033[1;32m  %-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "\033[1;34mFor Python tasks:\033[0m"
	@echo "  make -f python.mk <target>"
	@echo ""
	@echo "\033[1;34mFor C compilation tasks:\033[0m"
	@echo "  make -f csim.mk <target>"