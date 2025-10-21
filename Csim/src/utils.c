#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * ============================
 * DEBUG API
 * ============================
 */

void dump_tensor(const char* name, const DATA_TYPE* tensor, int size){
    FILE* fp = fopen(name, "wb");
    if (!fp) {
        fprintf(stderr, "%s:%d %s(): open file failed: %s\n", __FILE__, __LINE__, __func__, name);
        exit(EXIT_FAILURE);
    }

    size_t write_out = fwrite(tensor, sizeof(DATA_TYPE), size, fp);
    if (write_out != size) {
        fprintf(stderr, "%s:%d %s(): write file failed: %s\n", __FILE__, __LINE__, __func__, name);
        fprintf(stderr, "%s:%d %s(): expected to write = %d, but wrote = %zu\n", __FILE__, __LINE__, __func__, size, write_out);
        exit(EXIT_FAILURE);
    }
    DEBUG_LOG("Tensor: %s, size = %d", name, size);
    fclose(fp);
}