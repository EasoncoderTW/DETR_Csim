#ifndef STATISTIC_H
#define STATISTIC_H

#include <stdint.h>

typedef struct {
  uint64_t mac;
  uint64_t non_linear_op;
  uint64_t memory_read;
  uint64_t memory_write;
} statistics_t;

#define STATISTICS_INIT {0, 0, 0, 0}

#ifndef STATISTICS_CSV_FILENAME
#define STATISTICS_CSV_FILENAME "statistics.csv"
#endif

#define STATISTICS_CSV_FILE_HEADER \
  "Operation Name,MAC,Non-Linear Operations,Memory Read,Memory Write\n"
#define STATISTICS_CSV_FILE_FORMAT \
  "%s.%s,%lu,%lu,%lu,%lu\n"

void statistic_create_csv(char* filename) ;
void statistic_append_csv(statistics_t stat, const char* name, const char* subname ,const char* filename);

#ifdef ANALYZE
#define STATISTICS_INIT_CSV \
   statistic_create_csv(STATISTICS_CSV_FILENAME)

#define STATISTICS_APPEND_CSV(stat) \
  statistic_append_csv(stat, __func__, #stat, STATISTICS_CSV_FILENAME)

#define STATISTICS_CREATE(stat) statistics_t stat = STATISTICS_INIT
#define STATISTICS_RESET(stat) {\
   stat.mac = 0;\
   stat.non_linear_op = 0;\
   stat.memory_read = 0;\
   stat.memory_write = 0;\
}
#define STATISTICS_ADD_MAC(stat, v) stat.mac += (v)
#define STATISTICS_ADD_NON_LINEAR_OP(stat, v) stat.non_linear_op += (v)
#define STATISTICS_ADD_MEMORY_READ(stat, v) stat.memory_read += (v)
#define STATISTICS_ADD_MEMORY_WRITE(stat, v) stat.memory_write += (v)
#else
#define STATISTICS_INIT_CSV
#define STATISTICS_APPEND_CSV(stat)
#define STATISTICS_CREATE(stat)
#define STATISTICS_RESET(stat)
#define STATISTICS_ADD_MAC(stat, v)
#define STATISTICS_ADD_NON_LINEAR_OP(stat, v)
#define STATISTICS_ADD_MEMORY_READ(stat, v)
#define STATISTICS_ADD_MEMORY_WRITE(stat, v)
#endif

#endif // STATISTIC_H