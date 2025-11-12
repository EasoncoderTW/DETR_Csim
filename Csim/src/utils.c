#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

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

/*
 * ============================
 * fp16 clip
 * ============================
 */
#ifdef FP16_2
float fp16_clip(float x_val) {
    union {
        float f;
        uint32_t u;
    } in, out;

    in.f = x_val;

    uint32_t sign = (in.u >> 31) & 0x1;
    int32_t exp  = (in.u >> 23) & 0xFF;
    uint32_t mant = in.u & 0x7FFFFF;

    uint16_t h; // half precision bits

    // Handle NaN / Inf
    if (exp == 0xFF) {
        if (mant) { // NaN
            h = (sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0);
        } else { // Inf
            h = (sign << 15) | (0x1F << 10);
        }
    } else if (exp > 0x70 + 0x1E) {
        // Overflow: too large for FP16 (max exp=0x1F)
        h = (sign << 15) | (0x1F << 10);
    } else if (exp < 0x70 - 10) {
        // Underflow: too small for FP16
        h = (sign << 15); // ±0
    } else {
        // Normalized range
        int32_t new_exp = exp - 0x70 + 0xF;  // bias adjust (127 -> 15)
        uint32_t new_mant = mant >> 13;      // 23 → 10 bits
        // round to nearest, ties to even
        uint32_t round_bit = (mant >> 12) & 1;
        uint32_t sticky_bits = mant & 0xFFF;
        if (round_bit && (sticky_bits || (new_mant & 1)))
            new_mant++;

        // handle mantissa overflow
        if (new_mant == 0x400) {
            new_mant = 0;
            new_exp++;
        }

        if (new_exp >= 0x1F) { // overflow again
            h = (sign << 15) | (0x1F << 10);
        } else if (new_exp <= 0) { // subnormal
            if (new_exp < -10)
                h = (sign << 15); // too small
            else {
                new_mant = (new_mant | 0x400) >> (1 - new_exp);
                h = (sign << 15) | (new_mant & 0x3FF);
            }
        } else {
            h = (sign << 15) | ((new_exp & 0x1F) << 10) | (new_mant & 0x3FF);
        }
    }

    // Convert back to float
    // Expand half back to single precision (approximate, keeping FP16 precision)
    uint32_t sign32 = (h >> 15) & 0x1;
    uint32_t exp16  = (h >> 10) & 0x1F;
    uint32_t mant16 = h & 0x3FF;

    uint32_t exp32, mant32;
    if (exp16 == 0) {
        if (mant16 == 0) {
            exp32 = 0;
            mant32 = 0;
        } else {
            // subnormal
            int shift = 0;
            while ((mant16 & 0x400) == 0) {
                mant16 <<= 1;
                shift++;
            }
            mant16 &= 0x3FF;
            exp32 = 127 - 15 - shift;
            mant32 = mant16 << 13;
        }
    } else if (exp16 == 0x1F) {
        exp32 = 0xFF;
        mant32 = mant16 ? 0x7FFFFF : 0;
    } else {
        exp32 = exp16 - 15 + 127;
        mant32 = mant16 << 13;
    }

    out.u = (sign32 << 31) | (exp32 << 23) | mant32;
    return out.f;
}
#endif

#ifdef FP16
float fp16_clip(float x_val) {
    union { float f; uint32_t u; } v = { x_val };

    const int SHIFT = 13; // 23 mantissa bits - 10 bits for FP16 mantissa
    const uint32_t ROUND = 1u << (SHIFT - 1);
    const uint32_t MASK = (1u << SHIFT) - 1;

    // 快速 rounding: 加上半個 LSB，再清掉低位
    v.u += ROUND;

    // 若溢出 exponent（少數情況下會發生），不特別處理 — 自然進位即可
    v.u &= ~MASK;

    return v.f;
}
#endif
