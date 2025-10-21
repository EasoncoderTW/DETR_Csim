#include "model.h"
#include "statistic.h"
#include "sram.h"


#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * ============================
 * Activation Functions (No SRAM Tiling)
 * ============================
 */

/**
 * Applies the ReLU activation function. (not tiled, and not SRAM optimized)
 *
 * @param out The output array.
 * @param x The input array.
 * @param size The size of the array.
 */
void relu(DATA_TYPE* out, DATA_TYPE* x, int size) {
  assert(out != NULL);
  assert(x != NULL);
  DEBUG_LOG("size = %d", size);
  STATISTICS_CREATE(stat);
  int i;
  for (i = 0; i < size; i++) {
    out[i] = (x[i] < 0) ? 0 : x[i];
  }
  STATISTICS_INC_DRAM_READ(stat, size * sizeof(DATA_TYPE)); // x
  STATISTICS_INC_DRAM_WRITE(stat, size * sizeof(DATA_TYPE)); // out
  STATISTICS_INC_NON_LINEAR_OP(stat, size); // compare
  STATISTICS_APPEND_CSV(stat);
}

/**
  * Applies the softmax activation function. (compute in SRAM, will be record in multihead_attention)
  *
  * @param out The output array.
  * @param x The input array.
  * @param size The size of the array.
  */

#if SOFTMAX_METHOD == SOFTMAX_NORMAL
void softmax(DATA_TYPE* out, DATA_TYPE* x, int size) {
  assert(out != NULL);
  assert(x != NULL);
  // DEBUG_LOG("size = %d", size);
  int i;
  // find max value (for numerical stability)
  DATA_TYPE max_val = x[0];
  for (i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  // exp and sum
  DATA_TYPE sum = 0.0f;
  for (i = 0; i < size; i++) {
    out[i] = expf(x[i] - max_val);
    sum += out[i];
  }
  // normalize
  for (int i = 0; i < size; i++) {
    out[i] /= sum;
  }

  DEBUG_LOG("Softmax sum check: %f", sum);
  DEBUG_LOG("Softmax max value: %f", max_val);

}
#else // SOFTMAX_METHOD == SOFTMAX_SOLE

#define EXP_QBIS 4
#define QBITS ((uint64_t)1 << EXP_QBIS)
#define Q_ONE ((uint64_t)1 << QBITS)

typedef uint64_t qtype;

int approximate_log2(float x) {
  int n = (int)(-x);

  if (n > QBITS) {
    return QBITS;
  }

  return n;
}

float approximate_divide(int shift, qtype sum_i) {
  // leading one of sum_i
  int leading_one_pos = 0;
  for (int bit = 31; bit >= 0; bit--) {
    if ( (sum_i >> bit) & 0x1 ) {
      leading_one_pos = bit;
      break;
    }
  }

  //uint64_t a = (Q_ONE >> shift);
  uint64_t sum_i_approx = sum_i >> (leading_one_pos - 1);
  sum_i_approx = sum_i_approx << (leading_one_pos - 1);

  float r = (sum_i_approx & 0x1)? 0.568f : 0.818f;

  //return (float)a / (float)sum_i_approx;
  return r / (float)((uint64_t)1 << (leading_one_pos + shift - QBITS));
}

// SOLE softmax
void softmax(float* out, float* x, int size)  {
  assert(out != NULL);
  assert(x != NULL);

  float *y = (float*)malloc(sizeof(float) * size);
  int *m = (int*)malloc(sizeof(int) * size);
  assert(y != NULL);
  assert(m != NULL);

  // DEBUG_LOG("size = %d", size);
  int i;
  // find max value (for numerical stability)
  DATA_TYPE max_val = x[0];
  for (i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }

  float max_val_q = (int)(max_val * Q_ONE) / (float)(Q_ONE);

  // exp and sum
  //float sum = 0;
  qtype sum_i = 0;
  const float log2exp_approxiamte = 1.4375f; // log2(e) approximate to 23/16
  for (i = 0; i < size; i++) {
    float x_q = (int)(x[i] * Q_ONE) / (float)(Q_ONE);
    y[i] = (x_q - max_val_q) * log2exp_approxiamte;
    m[i] = approximate_log2(y[i]);
    sum_i += (uint64_t)(Q_ONE >> m[i]);
  }

  // normalize
  for (int i = 0; i < size; i++) {
    out[i] = approximate_divide(m[i], sum_i);
  }

  free(y);
  free(m);
}

#endif

/**
  * Applies the sigmoid activation function.
  *
  * @param out The output array.
  * @param x The input array.
  * @param size The size of the array.
  */
  void sigmoid(DATA_TYPE* out, DATA_TYPE* x, int size) {
  assert(out != NULL);
  assert(x != NULL);
  DEBUG_LOG("size = %d", size);
  int i;
  STATISTICS_CREATE(stat);
  for (i = 0; i < size; i++) {
    out[i] = 1 / (1 + expf(-x[i]));
  }
  STATISTICS_INC_DRAM_READ(stat, size * sizeof(DATA_TYPE)); // x
  STATISTICS_INC_DRAM_WRITE(stat, size * sizeof(DATA_TYPE)); // out
  STATISTICS_INC_NON_LINEAR_OP(stat, size);
  STATISTICS_INC_ADD(stat, size);
  STATISTICS_INC_DIV(stat, size);
  STATISTICS_APPEND_CSV(stat);
}
