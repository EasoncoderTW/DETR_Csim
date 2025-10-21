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

#define XQBITS 12
#define YQBITS 4
#define OQBITS 12
#define XQ_ONE (1 << XQBITS)
#define YQ_ONE (1 << YQBITS)
#define OQ_ONE (1 << OQBITS)

typedef int qtype;

qtype faster_exp2(float x) {
  int n = (int)(x);

  if (n < -YQBITS) {
    return 0;
  }

  if (n > 0 ) {
    return (YQ_ONE << n);
  }
  return (YQ_ONE >> (-n));
}

// approximate a / b = a * (1 / b), where 1 / b is approximated by 2^(-log2(b))
float faster_divide(qtype a, qtype b) {
  if (b == 0) return 0.0f;
  uint16_t aint = (uint16_t)a;
  uint16_t bint = (uint16_t)b;
  float result = (float)aint / (float)bint;
  result = (int)(result * OQ_ONE) / (float)OQ_ONE;
  return result;
}

// online softmax
void softmax(float* out, float* x, int size)  {
  assert(out != NULL);
  assert(x != NULL);

  float *y = (float*)malloc(sizeof(float) * size);
  qtype *m = (qtype*)malloc(sizeof(qtype) * size);
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

  float max_val_q = (int)(max_val * XQ_ONE) / (float)(XQ_ONE);

  // exp and sum
  //float sum = 0;
  qtype sum_i = 0;
  const float log2exp_approxiamte = 1.4375f; // log2(e) approximate to 23/16
  for (i = 0; i < size; i++) {
    float x_q = (int)(x[i] * XQ_ONE) / (float)(XQ_ONE);
    y[i] = (x_q - max_val_q) * log2exp_approxiamte;
    m[i] = faster_exp2(y[i]);
    sum_i += m[i];
  }

  // normalize
  for (int i = 0; i < size; i++) {
    out[i] = faster_divide(m[i], sum_i);
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




///------- TBD --------
#ifdef SOFTMAX_SOLE_FIXEDPOINT
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define Q 16
#define Q_ONE   (1 << Q)
#define LOG2E_Q ((int32_t)(1.4426950408889634 * Q_ONE))  // log2(e)

//--------------------------------------
// fixed-point helper
//--------------------------------------
static inline int32_t float_to_fixed(float f) {
    return (int32_t)(f * (float)Q_ONE + (f >= 0 ? 0.5f : -0.5f));
}

static inline float fixed_to_float(int32_t x) {
    return (float)x / (float)Q_ONE;
}

static inline int32_t fixed_mul(int32_t a, int32_t b) {
    return (int32_t)(((int64_t)a * b) >> Q);
}

//--------------------------------------
// LUT for pow2(fractional)
//--------------------------------------
#define LUT_SIZE 256
static int32_t pow2_lut[LUT_SIZE + 1];
static int initialized = 0;

void init_pow2_lut() {
    for (int i = 0; i <= LUT_SIZE; i++) {
        float frac = (float)i / LUT_SIZE;
        pow2_lut[i] = float_to_fixed(powf(2.0f, frac));
    }
}

//--------------------------------------
// 2^x using bit operations (Q16.16)
//--------------------------------------
int32_t fixed_pow2(int32_t x) {
    int32_t int_part = x >> Q;
    int32_t frac_part = x & (Q_ONE - 1);
    int idx = (frac_part * LUT_SIZE) >> Q;

    int32_t base = pow2_lut[idx];

    if (int_part >= 0) return base << int_part;
    else return base >> (-int_part);
}

//--------------------------------------
// log2(x) approximation
//--------------------------------------
int32_t fixed_log2(int32_t x) {
    if (x <= 0) return -32768 << Q;
    int shift = 0;
    while (x >= (2 * Q_ONE)) { x >>= 1; shift++; }
    while (x < Q_ONE) { x <<= 1; shift--; }
    // fractional part linear approx
    int32_t frac = x - Q_ONE;
    int32_t frac_q = frac >> (Q - 1); // ~frac/2
    return float_to_fixed((float)shift) + frac_q;
}

//--------------------------------------
// Softmax without any division
//--------------------------------------
void softmax_fixed_nodiv(float *x, float *y, int N) {

    if (!initialized) {
        init_pow2_lut();
        initialized = 1;
    }

    int32_t z[N];
    int32_t p[N];
    int32_t sum_p = 0;

    // convert to log2 domain
    for (int i = 0; i < N; i++)
        z[i] = fixed_mul(float_to_fixed(x[i]), LOG2E_Q);

    // find max
    int32_t m = z[0];
    for (int i = 1; i < N; i++)
        if (z[i] > m) m = z[i];

    // compute 2^(z_i - m)
    for (int i = 0; i < N; i++) {
        int32_t diff = z[i] - m;
        p[i] = fixed_pow2(diff);
        sum_p += p[i];
    }

    // log2(sum)
    int32_t log_sum = fixed_log2(sum_p);

    // compute normalized y_i = 2^{(z_i - m) - log_sum}
    for (int i = 0; i < N; i++) {
        int32_t exp_term = (z[i] - m) - log_sum;
        int32_t yi_fixed = fixed_pow2(exp_term);
        y[i] = fixed_to_float(yi_fixed);
    }

    // ⚠️ no division — results not strictly normalized, but consistent scale
}
#endif