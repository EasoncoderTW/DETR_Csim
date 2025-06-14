#include "statistic.h"
#include <stdio.h>

void statistic_create_csv(char* filename) {
  FILE* file = fopen(filename, "w");
  if (file == NULL) {
    fprintf(stderr, "Error opening file %s\n", filename);
    return;
  }
  fprintf(file, STATISTICS_CSV_FILE_HEADER);
  fclose(file);
}

void statistic_append_csv(statistics_t stat, const char* name, const char* subname, const char* filename) {
  FILE* file = fopen(filename, "a");
  if (file == NULL) {
    fprintf(stderr, "Error opening file %s\n", filename);
    return;
  }

  fprintf(file, STATISTICS_CSV_FILE_FORMAT, name, subname,
          stat.add,
          stat.mul,
          stat.div,
          stat.non_linear_op,
          stat.sram_read,
          stat.sram_write,
          stat.dram_read,
          stat.dram_write,
          stat.sram_size,
          stat.sram_used);
  fclose(file);
}