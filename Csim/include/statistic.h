#ifndef STATISTIC_H
#define STATISTIC_H

#include <stdint.h>

typedef struct {
  uint64_t add;
  uint64_t mul;
  uint64_t div;
  uint64_t non_linear_op;
  uint64_t sram_read;
  uint64_t sram_write;
  uint64_t dram_read;
  uint64_t dram_write;
  uint64_t sram_size;
  uint64_t sram_used;
} statistics_t;

#define STATISTICS_INIT {0, 0, 0, 0, 0, 0, 0, 0}

#ifndef STATISTICS_CSV_FILENAME
#define STATISTICS_CSV_FILENAME "statistics.csv"
#endif

#define STATISTICS_CSV_FILE_HEADER \
  "Operation Name,ADD,MUL,DIV,Non-Linear Operations,SRAM Read,SRAM Write,DRAM Read,DRAM Write,SRAM size,SRAM used\n"
#define STATISTICS_CSV_FILE_FORMAT \
  "%s.%s,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n"

void statistic_create_csv(char* filename) ;
void statistic_append_csv(statistics_t stat, const char* name, const char* subname ,const char* filename);

#ifdef ANALYZE
#define STATISTICS_INIT_CSV \
   statistic_create_csv(STATISTICS_CSV_FILENAME)

#define STATISTICS_APPEND_CSV(stat) \
  statistic_append_csv(stat, __func__, #stat, STATISTICS_CSV_FILENAME)

#define STATISTICS_CREATE(stat) statistics_t stat = STATISTICS_INIT
#define STATISTICS_RESET(stat) {\
  stat.add = 0;\
  stat.mul = 0;\
  stat.div = 0;\
  stat.non_linear_op = 0;\
  stat.sram_read = 0;\
  stat.sram_write = 0;\
  stat.dram_read = 0;\
  stat.dram_write = 0;\
  stat.sram_size = 0;\
  stat.sram_used = 0;\
}
#define STATISTICS_INC_ADD(stat, v) stat.add += (v)
#define STATISTICS_INC_MUL(stat, v) stat.mul += (v)
#define STATISTICS_INC_DIV(stat, v) stat.div += (v)
#define STATISTICS_INC_MAC(stat, v) STATISTICS_INC_ADD(stat, v); STATISTICS_INC_MUL(stat, v)
#define STATISTICS_INC_NON_LINEAR_OP(stat, v) stat.non_linear_op += (v)
#define STATISTICS_INC_SRAM_READ(stat, v) stat.sram_read += (v)
#define STATISTICS_INC_SRAM_WRITE(stat, v) stat.sram_write += (v)
#define STATISTICS_INC_DRAM_READ(stat, v) stat.dram_read += (v)
#define STATISTICS_INC_DRAM_WRITE(stat, v) stat.dram_write += (v)
#define STATISTICS_INC_DRAM_TO_SRAM(stat, v) \
  STATISTICS_INC_DRAM_READ(stat, v); \
  STATISTICS_INC_SRAM_WRITE(stat, v)
#define STATISTICS_INC_SRAM_TO_DRAM(stat, v) \
  STATISTICS_INC_SRAM_READ(stat, v); \
  STATISTICS_INC_DRAM_WRITE(stat, v)
#define STATISTICS_SET_SRAM_SIZE(stat, v) \
  stat.sram_size = (v)
#define STATISTICS_SET_SRAM_USED(stat, v) \
  stat.sram_used = (v)

#else
#define STATISTICS_INIT_CSV
#define STATISTICS_APPEND_CSV(stat)
#define STATISTICS_CREATE(stat)
#define STATISTICS_RESET(stat)
#define STATISTICS_INC_ADD(stat, v)
#define STATISTICS_INC_MUL(stat, v)
#define STATISTICS_INC_DIV(stat, v)
#define STATISTICS_INC_MAC(stat, v)
#define STATISTICS_INC_NON_LINEAR_OP(stat, v)
#define STATISTICS_INC_SRAM_READ(stat, v)
#define STATISTICS_INC_SRAM_WRITE(stat, v)
#define STATISTICS_INC_DRAM_READ(stat, v)
#define STATISTICS_INC_DRAM_WRITE(stat, v)
#define STATISTICS_INC_DRAM_TO_SRAM(stat, v)
#define STATISTICS_INC_SRAM_TO_DRAM(stat, v)
#define STATISTICS_SET_SRAM_SIZE(stat, v)
#define STATISTICS_SET_SRAM_USED(stat, v)
#endif

#endif // STATISTIC_H