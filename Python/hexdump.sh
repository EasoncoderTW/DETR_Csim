hexdump -v -e '4/4 "%10.5f" "\n"' ../Csim/model_output_boxes.bin > ./test2.log
hexdump -v -e '4/4 "%10.5f" "\n"' ./bin/model_pred_boxes.bin > ./test.log