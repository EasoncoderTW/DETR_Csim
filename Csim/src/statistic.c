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

  fprintf(file, STATISTICS_CSV_FILE_FORMAT, name, subname, stat.mac,
          stat.non_linear_op, stat.memory_read, stat.memory_write);

  fclose(file);
}