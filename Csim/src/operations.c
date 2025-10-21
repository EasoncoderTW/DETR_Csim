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
 * Operations
 * ============================
 */

/**
 * Performs a 2D convolution operation with tiling and data reuse.
 *
 * @param output The output ConvolutionTensor.
 * @param input The input ConvolutionTensor.
 * @param config The ConvConfig structure.
 * @param weights The ConvWeights structure.
 */
void conv2D(ConvolutionTensor* output, const ConvolutionTensor* input,
  const ConvConfig* config, const ConvWeights* weights) {
  assert(weights->weight != NULL);
  assert(input->x != NULL);
  assert(config->in_channels == input->channels);

  // calculate dimensions from input and output
  int output_height =
    (input->height + 2 * config->padding - config->kernel_size) /
  config->stride +
    1;
  int output_width =
    (input->width + 2 * config->padding - config->kernel_size) /
  config->stride +
    1;

  // prepare output ConvolutionTensor
  output->height = output_height;
  output->width = output_width;
  output->channels = config->out_channels;
  malloc_conv2D_tensor(output);
  DEBUG_LOG("kernels = (%d, %d, %d * %d), stride = %d, padding = %d",
  config->in_channels, config->out_channels, config->kernel_size,
  config->kernel_size, config->stride, config->padding);

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_out_height = CONV_TILED_OUT_HEIGHT;  // Height of the output tile
  int tile_out_width = CONV_TILED_OUT_WIDTH;   // Width of the output tile
  int tile_out_channels = CONV_TILED_OUT_CHANNELS; // Number of output channels per tile
  int tile_in_height = (CONV_TILED_OUT_HEIGHT - 1) * config->stride + config->kernel_size; // Adjust input tile height, keep padding
  int tile_in_width = (CONV_TILED_OUT_WIDTH -1) * config->stride + config->kernel_size;  // Adjust input tile width, keep padding
  int tile_in_channels = CONV_TILED_IN_CHANNELS;  // Number of input channels per tile

  DEBUG_LOG("Tiling parameters: tile_out_height = %d, tile_out_width = %d, tile_out_channels = %d, tile_in_height = %d, tile_in_width = %d, tile_in_channels = %d",
    tile_out_height, tile_out_width, tile_out_channels, tile_in_height, tile_in_width, tile_in_channels);

  // Buffers for data reuse
  DATA_TYPE* input_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_in_channels * tile_in_height * tile_in_width * sizeof(DATA_TYPE));
  DATA_TYPE* weight_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_out_channels * config->in_channels * config->kernel_size * config->kernel_size * sizeof(DATA_TYPE));
  DATA_TYPE* output_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_out_channels * tile_out_height * tile_out_width * sizeof(DATA_TYPE));

  DATA_TYPE* bias_buffer = NULL;
  // if bias is present, allocate memory for bias buffer, and copy bias values
  if (weights->bias) {
    bias_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_out_channels * sizeof(DATA_TYPE));
  }

  // Perform tiled convolution
  for (int oc_tile = 0; oc_tile < config->out_channels; oc_tile += tile_out_channels) {

    // Load bias into bias buffer if present
    if (bias_buffer) {
      for (int oc = 0; oc < tile_out_channels && (oc + oc_tile) < config->out_channels; ++oc) {
        bias_buffer[oc] = weights->bias[oc + oc_tile];
        STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
      }
    }

    // Load weights into weight buffer - SRAM
    for (int oc = 0; oc < tile_out_channels && (oc + oc_tile) < config->out_channels; ++oc) {
      for (int ic = 0; ic < config->in_channels; ++ic) {
        for (int r = 0; r < config->kernel_size; ++r) {
          for (int c = 0; c < config->kernel_size; ++c) {
            weight_buffer[CONVWEIGHT_INDEX(config->in_channels, config->kernel_size, config->kernel_size, oc, ic, r, c)] =
              weights->weight[CONVWEIGHT_INDEX(config->in_channels, config->kernel_size, config->kernel_size, oc + oc_tile, ic, r, c)];
            STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
          }
        }
      }
    }

    for (int oh_tile = 0; oh_tile < output_height; oh_tile += tile_out_height) {
      for (int ow_tile = 0; ow_tile < output_width; ow_tile += tile_out_width) {
        for (int ic_tile = 0; ic_tile < config->in_channels; ic_tile += tile_in_channels) {

          // Load input tile into input buffer
          for (int ic = 0; ic < tile_in_channels && (ic + ic_tile) < config->in_channels; ++ic) {
            for (int h = 0; h < tile_in_height; ++h) {
              for (int w = 0; w < tile_in_width; ++w) {

                int ih = h + oh_tile * config->stride - config->padding;
                int iw = w + ow_tile * config->stride - config->padding;

                if (ih >= 0 && ih < input->height && iw >= 0 && iw < input->width) {
                  input_buffer[CONVTENSOR_INDEX(tile_in_height, tile_in_width, ic, h, w)] =
                    input->x[CONVTENSOR_INDEX(input->height, input->width, ic + ic_tile, ih, iw)];
                  STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
                } else {
                  input_buffer[CONVTENSOR_INDEX(tile_in_height, tile_in_width, ic, h, w)] = 0; // Zero padding
                }

              }
            }
          }


          //
          // Compute in SRAM
          //

          // Perform convolution for the current tile
          for (int oh = 0; oh < tile_out_height && (oh + oh_tile) < output_height; ++oh) {
            for (int ow = 0; ow < tile_out_width && (ow + ow_tile) < output_width; ++ow) {
              for (int oc = 0; oc < tile_out_channels && (oc + oc_tile) < config->out_channels; ++oc) {

                DATA_TYPE sum;
                if (ic_tile == 0){
                  if (bias_buffer) {
                    // Initialize sum with bias if present
                    sum = bias_buffer[oc];
                    STATISTICS_INC_SRAM_READ(stat, sizeof(DATA_TYPE));
                  } else {
                    // Initialize sum to zero if no bias
                    sum = 0.0f;
                  }
                }
                else {
                  // Load sum from previous output pixel (partial sum)
                  sum = output_buffer[CONVTENSOR_INDEX(tile_out_height, tile_out_width, oc, oh, ow)];
                  STATISTICS_INC_SRAM_READ(stat, sizeof(DATA_TYPE));
                }


                // Perform convolution for the current output pixel
                for (int r = 0; r < config->kernel_size; ++r) {
                  for (int c = 0; c < config->kernel_size; ++c) {
                    for (int ic = 0; ic < tile_in_channels && (ic + ic_tile) < config->in_channels; ++ic) {
                      int ih = oh * config->stride + r;
                      int iw = ow * config->stride + c;
                      if (ih < tile_in_height && iw < tile_in_width) {
                        uint32_t x_idx = CONVTENSOR_INDEX(tile_in_height, tile_in_width, ic, ih, iw);
                        uint32_t w_idx = CONVWEIGHT_INDEX(config->in_channels, config->kernel_size, config->kernel_size, oc, ic + ic_tile, r, c);

                        // OP: MAC (Multiply-Accumulate)
                        sum += input_buffer[x_idx] * weight_buffer[w_idx];

                        STATISTICS_INC_MAC(stat, 1);
                        STATISTICS_INC_SRAM_READ(stat, 2 * sizeof(DATA_TYPE));
                      }
                    }
                  }
                }
                output_buffer[CONVTENSOR_INDEX(tile_out_height, tile_out_width, oc, oh, ow)] = sum;
                STATISTICS_INC_SRAM_WRITE(stat, sizeof(DATA_TYPE));
              }
            }
          }

          //
          // Compute in SRAM (end)
          //

        } // ic_tile

        // Write output tile back to DRAM
        for (int oc = 0; oc < tile_out_channels && (oc + oc_tile) < config->out_channels; ++oc) {
          for (int oh = 0; oh < tile_out_height && (oh + oh_tile) < output_height; ++oh) {
            for (int ow = 0; ow < tile_out_width && (ow + ow_tile) < output_width; ++ow) {
              output->x[CONVTENSOR_INDEX(output_height, output_width, oc + oc_tile, oh + oh_tile, ow + ow_tile)] =
                output_buffer[CONVTENSOR_INDEX(tile_out_height, tile_out_width, oc, oh, ow)];
              STATISTICS_INC_SRAM_TO_DRAM(stat, sizeof(DATA_TYPE));
            } // ow
          } // oh
        } // oc

      } // ow_tile
    } // oh_tile
  } // oc_tile

  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // save statistics
  STATISTICS_APPEND_CSV(stat);
}

/**
  * Performs a 2D max pooling operation with SRAM tiling.
  *
  * @param output The output ConvolutionTensor.
  * @param input The input ConvolutionTensor.
  * @param config The MaxPoolConfig structure.
  */
void maxpooling2D(ConvolutionTensor* output, const ConvolutionTensor* input,
      const MaxPoolConfig* config) {
  assert(input->x != NULL);
  DEBUG_LOG();
  int padded_height = input->height + 2 * config->padding;
  int padded_width = input->width + 2 * config->padding;

  int height = (padded_height - config->kernel_size) / config->stride + 1;
  int width = (padded_width - config->kernel_size) / config->stride + 1;

  output->height = height;
  output->width = width;
  output->channels = input->channels;
  malloc_conv2D_tensor(output);

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_out_height = MAXPOOL_TILED_OUT_HEIGHT;  // Height of the output tile
  int tile_out_width = MAXPOOL_TILED_OUT_WIDTH;   // Width of the output tile
  int tile_channels = MAXPOOL_TILED_CHANNELS; // Number of channels per tile
  int tile_in_height = tile_out_height * config->stride + config->kernel_size - 1; // Adjust input tile height
  int tile_in_width = tile_out_width * config->stride + config->kernel_size - 1;  // Adjust input tile width

  // Buffers for data reuse
  DATA_TYPE* input_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * tile_in_height * tile_in_width * sizeof(DATA_TYPE));
  DATA_TYPE* output_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * tile_out_height * tile_out_width * sizeof(DATA_TYPE));

  // Perform tiled max pooling
  for (int h_tile = 0; h_tile < height; h_tile += tile_out_height) {
    for (int w_tile = 0; w_tile < width; w_tile += tile_out_width) {
      for (int ch_tile = 0; ch_tile < input->channels; ch_tile += tile_channels) {

      //
      // Compute in SRAM
      //

      // Load input tile into input buffer
      for (int ch = 0; ch < tile_channels && (ch + ch_tile) < input->channels; ++ch) {
        for (int h = 0; h < tile_in_height; ++h) {
          for (int w = 0; w < tile_in_width; ++w) {
            int in_h = h_tile * config->stride + h - config->padding;
            int in_w = w_tile * config->stride + w - config->padding;

            if (in_h >= 0 && in_h < input->height && in_w >= 0 && in_w < input->width) {
              input_buffer[CONVTENSOR_INDEX(tile_in_height, tile_in_width, ch, h, w)] =
                input->x[CONVTENSOR_INDEX(input->height, input->width, ch + ch_tile, in_h, in_w)];
              STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
            } else {
              input_buffer[CONVTENSOR_INDEX(tile_in_height, tile_in_width, ch, h, w)] = -__FLT_MAX__; // Padding
            }
          }
        }
      }

      // Perform max pooling for the current tile
      for (int h = 0; h < tile_out_height && (h + h_tile) < height; ++h) {
        for (int w = 0; w < tile_out_width && (w + w_tile) < width; ++w) {
          for (int ch = 0; ch < tile_channels && (ch + ch_tile) < input->channels; ++ch) {

            DATA_TYPE max_value = -__FLT_MAX__;

            for (int r = 0; r < config->kernel_size; ++r) {
              for (int c = 0; c < config->kernel_size; ++c) {
                int in_h = h * config->stride + r;
                int in_w = w * config->stride + c;

                if (in_h >= 0 && in_h < tile_in_height && in_w >= 0 && in_w < tile_in_width) {
                  DATA_TYPE value = input_buffer[CONVTENSOR_INDEX(tile_in_height, tile_in_width, ch, in_h, in_w)];
                  STATISTICS_INC_SRAM_READ(stat, 1 * sizeof(DATA_TYPE));

                  // OP: compare
                  if (value > max_value) {
                    max_value = value;
                  }
                  // statistics - compare
                  STATISTICS_INC_NON_LINEAR_OP(stat, 1);
                }
              } // c
            } // r

            output_buffer[CONVTENSOR_INDEX(tile_out_height, tile_out_width, ch, h, w)] = max_value;
            // statistics
            STATISTICS_INC_SRAM_WRITE(stat, 1 * sizeof(DATA_TYPE));
          } // ch
        } // w
      } // h

      // Write output tile back to DRAM
      for (int ch = 0; ch < tile_channels && (ch + ch_tile) < input->channels; ++ch) {
        for (int h = 0; h < tile_out_height && (h + h_tile) < height; ++h) {
          for (int w = 0; w < tile_out_width && (w + w_tile) < width; ++w) {
            output->x[CONVTENSOR_INDEX(height, width, ch + ch_tile, h + h_tile, w + w_tile)] =
            output_buffer[CONVTENSOR_INDEX(tile_out_height, tile_out_width, ch, h, w)];
            STATISTICS_INC_SRAM_TO_DRAM(stat, sizeof(DATA_TYPE));
          } // w
        } // h
      } // ch

      //
      // Compute in SRAM (end)
      //

      } // ch_tile
    } // w_tile
  } // h_tile

  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // save statistics
  STATISTICS_APPEND_CSV(stat);
}

/**
  * Applies batch normalization with SRAM tiling.
  *
  * @param output The output ConvolutionTensor.
  * @param input The input ConvolutionTensor.
  * @param bn The BatchNormWeights structure.
  */
void batchnorm2D(ConvolutionTensor* output, const ConvolutionTensor* input,
      BatchNormWeights* bn) {
  assert(output->x != NULL);
  assert(input->x != NULL);
  int C = input->channels;
  int H = input->height;
  int W = input->width;
  const DATA_TYPE eps = 1e-5;
  DEBUG_LOG("input shape = (%d, %d, %d)", C, H, W);

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_channels = BATCHNORM_TILED_CHANNELS; // Number of channels per tile
  int tile_height = BATCHNORM_TILED_HEIGHT;  // Height of the tile
  int tile_width = BATCHNORM_TILED_WIDTH;   // Width of the tile

  // Buffers for data reuse
  DATA_TYPE* input_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * tile_height * tile_width * sizeof(DATA_TYPE));
  DATA_TYPE* output_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * tile_height * tile_width * sizeof(DATA_TYPE));
  DATA_TYPE* scale_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * sizeof(DATA_TYPE));
  DATA_TYPE* bias_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_channels * sizeof(DATA_TYPE));

  // Perform tiled batch normalization
  for (int c_tile = 0; c_tile < C; c_tile += tile_channels) {
    //
    // Compute in SRAM
    //

    // Load weights and statistics into buffersm, and compute scale and bias
    for (int c = 0; c < tile_channels && (c + c_tile) < C; ++c) {
      scale_buffer[c] = bn->weight[c + c_tile] / sqrtf(bn->running_var[c + c_tile] + eps);
      bias_buffer[c] = bn->bias[c + c_tile] - bn->running_mean[c + c_tile] * scale_buffer[c];
      STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE) * 4); // gamma, beta, mean, var
    }
    STATISTICS_INC_NON_LINEAR_OP(stat, tile_channels); // sqrtf
    STATISTICS_INC_ADD(stat, tile_channels); // scale
    STATISTICS_INC_DIV(stat, tile_channels); // scale
    STATISTICS_INC_MAC(stat, tile_channels); // bias

    for (int h_tile = 0; h_tile < H; h_tile += tile_height) {
      for (int w_tile = 0; w_tile < W; w_tile += tile_width) {
        // Load input tile into input buffer
        for (int c = 0; c < tile_channels && (c + c_tile) < C; ++c) {
          for (int h = 0; h < tile_height && (h + h_tile) < H; ++h) {
            for (int w = 0; w < tile_width && (w + w_tile) < W; ++w) {
              input_buffer[CONVTENSOR_INDEX(tile_height, tile_width, c, h, w)] =
              input->x[CONVTENSOR_INDEX(H, W, c + c_tile, h + h_tile, w + w_tile)];
              STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE)); // input_buffer
            }
          }
        }

        // Perform batch normalization for the current tile
        for (int c = 0; c < tile_channels && (c + c_tile) < C; ++c) {
          for (int h = 0; h < tile_height && (h + h_tile) < H; ++h) {
            for (int w = 0; w < tile_width && (w + w_tile) < W; ++w) {
              int index = CONVTENSOR_INDEX(tile_height, tile_width, c, h, w);
              output_buffer[index] = input_buffer[index] * scale_buffer[c] + bias_buffer[c];
            }
          }
          STATISTICS_INC_SRAM_READ(stat, tile_height * tile_width * sizeof(DATA_TYPE)); // input_buffer
          STATISTICS_INC_SRAM_WRITE(stat, tile_height * tile_width * sizeof(DATA_TYPE)); // output_buffer
          STATISTICS_INC_MAC(stat, tile_height * tile_width);
        }

        // Write output tile back to DRAM
        for (int c = 0; c < tile_channels && (c + c_tile) < C; ++c) {
          for (int h = 0; h < tile_height && (h + h_tile) < H; ++h) {
            for (int w = 0; w < tile_width && (w + w_tile) < W; ++w) {
              output->x[CONVTENSOR_INDEX(H, W, c + c_tile, h + h_tile, w + w_tile)] =
              output_buffer[CONVTENSOR_INDEX(tile_height, tile_width, c, h, w)];
              STATISTICS_INC_SRAM_TO_DRAM(stat, sizeof(DATA_TYPE)); // output_buffer
            }
          }
        }
      } // w_tile
    } // h_tile

    //
    // Compute in SRAM (end)
    //
  }// c_tile
  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // save statistics
  STATISTICS_APPEND_CSV(stat);
}

/**
  * Performs a general matrix multiplication (GEMM) operation with SRAM tiling.
  *
  * @param out The output array.
  * @param x The input array.
  * @param w The weights array.
  * @param b The bias array.
  * @param n The number of rows in the output.
  * @param id The input dimension.
  * @param od The output dimension.
  */
void gemm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n,
      int id, int od) {
  assert(out != NULL);
  assert(x != NULL);
  assert(w != NULL);
  assert(b != NULL);
  DEBUG_LOG("W (%d, %d) @ x(%d, %d) -> xout (%d, %d)", id, od, id, n, od, n);
  // W (od, id) @ x(id, n) -> xout (od, n)

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_od = GEMM_TILED_OUT_DIM; // Number of output dimensions per tile
  int tile_id = GEMM_TILED_IN_DIM; // Number of input dimensions per tile
  int tile_n = GEMM_TILED_N;  // Number of rows per tile

  // Buffers for data reuse
  DATA_TYPE* x_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_id * tile_n * sizeof(DATA_TYPE));
  DATA_TYPE* w_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_od * tile_id * sizeof(DATA_TYPE));
  DATA_TYPE* b_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_od * sizeof(DATA_TYPE));
  DATA_TYPE* out_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_od * tile_n * sizeof(DATA_TYPE));

  // Perform tiled GEMM
  for (int o_tile = 0; o_tile < od; o_tile += tile_od) {

    // Load biases into b buffer
    for (int o = 0; o < tile_od && (o + o_tile) < od; ++o) {
      b_buffer[o] = b[o + o_tile];
      STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
    }

    for (int n_tile = 0; n_tile < n; n_tile += tile_n) {
      for (int i_tile = 0; i_tile < id; i_tile += tile_id) {

        //
        // Compute in SRAM
        //

        // Load input tile into x buffer
        for (int i = 0; i < tile_id && (i + i_tile) < id; ++i) {
          for (int j = 0; j < tile_n && (j + n_tile) < n; ++j) {
            x_buffer[i * tile_n + j] = x[(i + i_tile) * n + (j + n_tile)];
            STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
          }
        }

        // Load weights into w buffer
        for (int o = 0; o < tile_od && (o + o_tile) < od; ++o) {
          for (int i = 0; i < tile_id && (i + i_tile) < id; ++i) {
            w_buffer[o * tile_id + i] = w[(o + o_tile) * id + (i + i_tile)];
            STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE));
          }
        }

        // Perform GEMM for the current tile
        for (int o = 0; o < tile_od && (o + o_tile) < od; ++o) {
          for (int j = 0; j < tile_n && (j + n_tile) < n; ++j) {
            DATA_TYPE sum;
            if (i_tile == 0) {
              // Initialize sum with bias if present
              sum = b_buffer[o];
              STATISTICS_INC_SRAM_READ(stat, sizeof(DATA_TYPE));
            } else {
              // Initialize sum to zero if no bias
              sum = out_buffer[o * tile_n + j];
            }

            for (int i = 0; i < tile_id && (i + i_tile) < id; ++i) {
              sum += w_buffer[o * tile_id + i] * x_buffer[i * tile_n + j];
              STATISTICS_INC_MAC(stat, 1);
              STATISTICS_INC_SRAM_READ(stat, 2 * sizeof(DATA_TYPE)); // w, x
            }
            out_buffer[o * tile_n + j] = sum;
            STATISTICS_INC_SRAM_WRITE(stat, sizeof(DATA_TYPE));
          }
        }

        //
        // Compute in SRAM (end)
        //

      } // i_tile

      // Write output tile back to DRAM
      for (int o = 0; o < tile_od && (o + o_tile) < od; ++o) {
        for (int j = 0; j < tile_n && (j + n_tile) < n; ++j) {
          out[(o + o_tile) * n + (j + n_tile)] = out_buffer[o * tile_n + j];
          STATISTICS_INC_SRAM_TO_DRAM(stat, sizeof(DATA_TYPE));
        }
      }

    } // n_tile
  } // o_tile

  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // save statistics
  STATISTICS_APPEND_CSV(stat);
}


/**
  * Adds two arrays element-wise with SRAM tiling.
  *
  * @param out The output array.
  * @param x The first input array.
  * @param y The second input array.
  * @param size The size of the arrays.
  */
void add(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* y, int size) {
  assert(out != NULL);
  assert(x != NULL);
  assert(y != NULL);
  DEBUG_LOG("size = %d", size);

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_size = ADD_TILED_SIZE; // Number of elements per tile

  // Buffers for data reuse
  DATA_TYPE* x_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_size * sizeof(DATA_TYPE));
  DATA_TYPE* y_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_size * sizeof(DATA_TYPE));
  DATA_TYPE* out_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_size * sizeof(DATA_TYPE));

  // Perform tiled addition
  for (int i_tile = 0; i_tile < size; i_tile += tile_size) {
    int current_tile_size = (i_tile + tile_size > size) ? (size - i_tile) : tile_size;

    //
    // Compute in SRAM
    //

    // Load input tiles into buffers
    memcpy(x_buffer, x + i_tile, current_tile_size * sizeof(DATA_TYPE));
    memcpy(y_buffer, y + i_tile, current_tile_size * sizeof(DATA_TYPE));
    STATISTICS_INC_DRAM_TO_SRAM(stat, 2 * current_tile_size * sizeof(DATA_TYPE)); // x, y

    // Perform addition for the current tile
    for (int i = 0; i < current_tile_size; i++) {
      out_buffer[i] = x_buffer[i] + y_buffer[i];
    }
    STATISTICS_INC_SRAM_READ(stat, 2 * current_tile_size * sizeof(DATA_TYPE)); // x_buffer, y_buffer
    STATISTICS_INC_SRAM_WRITE(stat, current_tile_size * sizeof(DATA_TYPE)); // out_buffer
    STATISTICS_INC_ADD(stat, current_tile_size);

    // Write output tile back to DRAM
    memcpy(out + i_tile, out_buffer, current_tile_size * sizeof(DATA_TYPE));
    STATISTICS_INC_SRAM_TO_DRAM(stat, current_tile_size * sizeof(DATA_TYPE)); // out

    //
    // Compute in SRAM (end)
    //
  }

  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // Save statistics
  STATISTICS_APPEND_CSV(stat);
}

/**
  * Performs a multi-head attention operation with SRAM tiling.
  *
  * @param out The output array.
  * @param qx The query array.
  * @param kx The key array.
  * @param vx The value array.
  * @param att The attention array.
  * @param att_mask The attention mask array.
  * @param n_heads The number of attention heads.
  * @param dim The dimension of each head.
  * @param q_len The length of the query.
  * @param kv_len The length of the key/value.
  */
void multihead_attention(DATA_TYPE* out, DATA_TYPE* qx, DATA_TYPE* kx,
              DATA_TYPE* vx, DATA_TYPE* att, MASK_TYPE* att_mask,
              int n_heads, int dim, int q_len, int kv_len) {
  assert(out != NULL);
  assert(qx != NULL);
  assert(kx != NULL);
  assert(vx != NULL);
  assert(att != NULL);

  int use_att_mask = (att_mask != NULL);
  int head_size = dim / n_heads;
  DATA_TYPE head_size_sqrt = sqrtf(head_size);

  DEBUG_LOG("n_heads = %d, dim = %d, q_len = %d, kv_len = %d",
      n_heads, dim, q_len, kv_len);

  STATISTICS_CREATE(stat_softmax);
  STATISTICS_CREATE(stat_self_attn);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_q_len = MULTIHEAD_ATTENTION_TILED_Q_LEN;  // Query length per tile
  int tile_kv_len = MULTIHEAD_ATTENTION_TILED_KV_LEN; // Key/Value length per tile

  // Buffers for data reuse
  DATA_TYPE* q_buffer = (DATA_TYPE*)sram_alloc(&sram, head_size * tile_q_len * sizeof(DATA_TYPE));
  DATA_TYPE* k_buffer = (DATA_TYPE*)sram_alloc(&sram, head_size * tile_kv_len * sizeof(DATA_TYPE));
  DATA_TYPE* v_buffer = (DATA_TYPE*)sram_alloc(&sram, head_size * tile_kv_len * sizeof(DATA_TYPE));
  DATA_TYPE* att_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_q_len * kv_len * sizeof(DATA_TYPE));
  DATA_TYPE* out_buffer = (DATA_TYPE*)sram_alloc(&sram, head_size * tile_q_len * sizeof(DATA_TYPE));

  for (int h = 0; h < n_heads; h++) {
    // Get q, k, v of this head
    DATA_TYPE* Q = qx + h * head_size * q_len;
    DATA_TYPE* K = kx + h * head_size * kv_len;
    DATA_TYPE* V = vx + h * head_size * kv_len;
    DATA_TYPE* OUT = out + h * head_size * q_len;

    for (int q_tile = 0; q_tile < q_len; q_tile += tile_q_len) {

      // calculate QK^T
      for (int k_tile = 0; k_tile < kv_len; k_tile += tile_kv_len) {

        //
        // Compute in SRAM
        //

        // Load K tile into k buffer
        for (int d = 0; d < head_size; ++d) {
          for (int k = 0; k < tile_kv_len && (k + k_tile) < kv_len; ++k) {
            k_buffer[d * tile_kv_len + k] = K[d * kv_len + (k + k_tile)];
            STATISTICS_INC_DRAM_TO_SRAM(stat_self_attn, sizeof(DATA_TYPE));
          }
        }

        // Load Q tile into q buffer
        for (int d = 0; d < head_size; ++d) {
          for (int q = 0; q < tile_q_len && (q + q_tile) < q_len; ++q) {
            q_buffer[d * tile_q_len + q] = Q[d * q_len + (q + q_tile)];
            STATISTICS_INC_DRAM_TO_SRAM(stat_self_attn, sizeof(DATA_TYPE));
          }
        }

        // Compute attention scores
        for (int q = 0; q < tile_q_len && (q + q_tile) < q_len; ++q) {
          for (int k = 0; k < tile_kv_len && (k + k_tile) < kv_len; ++k) {
            if (use_att_mask && (!att_mask[q + q_tile] || !att_mask[k + k_tile])) {
              att_buffer[q * kv_len + (k + k_tile)] = -__FLT_MAX__;
              STATISTICS_INC_SRAM_WRITE(stat_self_attn, sizeof(DATA_TYPE));
              continue;
            }

            DATA_TYPE score = 0;

            for (int d = 0; d < head_size; ++d) {
              score += q_buffer[d * tile_q_len + q] * k_buffer[d * tile_kv_len + k];
            }
            STATISTICS_INC_MAC(stat_self_attn, head_size);
            STATISTICS_INC_SRAM_READ(stat_self_attn, 2 * head_size * sizeof(DATA_TYPE));

            att_buffer[q * kv_len + (k + k_tile)] = score / head_size_sqrt;
            STATISTICS_INC_DIV(stat_self_attn, 1);
            STATISTICS_INC_SRAM_WRITE(stat_self_attn, sizeof(DATA_TYPE));
          }
        }
      } // k_tile

      // Apply softmax to attention scores
      for (int q = 0; q < tile_q_len && (q + q_tile) < q_len; ++q) {
        softmax(att_buffer + q * kv_len, att_buffer + q * kv_len, kv_len);
        STATISTICS_INC_MAC(stat_softmax, 2 * kv_len);
        STATISTICS_INC_NON_LINEAR_OP(stat_softmax, kv_len);
        STATISTICS_INC_SRAM_READ(stat_softmax, 3 * kv_len * sizeof(DATA_TYPE));
        STATISTICS_INC_SRAM_WRITE(stat_softmax, 2 * kv_len * sizeof(DATA_TYPE));
      }

      // Compute output of attention
      for (int v_tile = 0; v_tile < kv_len; v_tile += tile_kv_len) {

        // Load V tile into v buffer
        for (int d = 0; d < head_size; ++d) {
          for (int v = 0; v < tile_kv_len && (v + v_tile) < kv_len; ++v) {
            v_buffer[d * tile_kv_len + v] = V[d * kv_len + (v + v_tile)];
            STATISTICS_INC_DRAM_TO_SRAM(stat_self_attn, sizeof(DATA_TYPE));
          }
        }

        // Tile output computation (tile_q_len x tile_kv_len)
        for (int q = 0; q < tile_q_len && (q + q_tile) < q_len; ++q) {
          for (int d = 0; d < head_size; ++d) {

            DATA_TYPE sum;
            if (v_tile == 0) {
              sum = 0; // Initialize sum to zero for the first tile
            } else {
              sum = out_buffer[d * tile_q_len + q]; // Use previous tile's output (partial sum)
              STATISTICS_INC_SRAM_READ(stat_self_attn, sizeof(DATA_TYPE)); // Read previous sum
            }

            for (int v = 0; v < tile_kv_len && (v + v_tile) < kv_len; ++v) {
              sum += att_buffer[q * kv_len + (v + v_tile)] * v_buffer[d * tile_kv_len + v];
            }
            STATISTICS_INC_MAC(stat_self_attn, kv_len);
            STATISTICS_INC_SRAM_READ(stat_self_attn, 2 * kv_len * sizeof(DATA_TYPE));

            out_buffer[d * tile_q_len + q] = sum;
            STATISTICS_INC_SRAM_WRITE(stat_self_attn, sizeof(DATA_TYPE));
          } // d
        } // q
      } // v_tile

      // Write output tile back to DRAM
      for (int d = 0; d < head_size; ++d) {
        for (int q = 0; q < tile_q_len && (q + q_tile) < q_len; ++q) {
          OUT[d * q_len + (q + q_tile)] = out_buffer[d * tile_q_len + q];
          STATISTICS_INC_SRAM_TO_DRAM(stat_self_attn, sizeof(DATA_TYPE));
        }
      }

      //
      // Compute in SRAM (end)
      //

    } // q_tile
  } // h

  STATISTICS_SET_SRAM_SIZE(stat_self_attn, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat_self_attn, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
    sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);

  // Save statistics
  STATISTICS_APPEND_CSV(stat_softmax);
  STATISTICS_APPEND_CSV(stat_self_attn);
}


/**
  * Applies layer normalization with SRAM tiling.
  *
  * @param out The output array.
  * @param x The input array.
  * @param w The weights array.
  * @param b The bias array.
  * @param n The number of elements.
  * @param dim The dimension of each element.
  */
void layernorm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n,
    int dim) {
  assert(out != NULL);
  assert(x != NULL);
  assert(w != NULL);
  assert(b != NULL);
  DEBUG_LOG("n = %d, dim = %d", n, dim);
  const DATA_TYPE eps = 1e-5;

  STATISTICS_CREATE(stat);

  // Initialize SRAM manager
  SRAM_Manager_t sram;
  sram_init(&sram, SRAM_DEFAULT_SIZE);

  // Tiling parameters (can be configured)
  int tile_n = LAYERNORM_TILED_N;   // Number of elements per tile

  // Buffers for data reuse
  DATA_TYPE* x_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_n * dim * sizeof(DATA_TYPE));
  DATA_TYPE* w_buffer = (DATA_TYPE*)sram_alloc(&sram, dim * sizeof(DATA_TYPE));
  DATA_TYPE* b_buffer = (DATA_TYPE*)sram_alloc(&sram, dim * sizeof(DATA_TYPE));
  DATA_TYPE* out_buffer = (DATA_TYPE*)sram_alloc(&sram, tile_n * dim * sizeof(DATA_TYPE));

  // Load weights and biases into buffers
  memcpy(w_buffer, w, dim * sizeof(DATA_TYPE));
  memcpy(b_buffer, b, dim * sizeof(DATA_TYPE));
  STATISTICS_INC_DRAM_TO_SRAM(stat, dim * sizeof(DATA_TYPE) * 2); // w, b

  // Perform tiled layer normalization (on dimension dim)
  for (int n_tile = 0; n_tile < n; n_tile += tile_n) {
    int current_tile_n = (n_tile + tile_n > n) ? (n - n_tile) : tile_n;

    //
    // Compute in SRAM
    //

    // Load input tile into x buffer (dim, n)
    for (int i = 0; i < current_tile_n; i++) {
      for (int d = 0; d < dim; d++) {
        x_buffer[d * current_tile_n + i] = x[d * n + i + n_tile];
        STATISTICS_INC_DRAM_TO_SRAM(stat, sizeof(DATA_TYPE)); // read x
      }
    }

    // Perform layer normalization for the current tile
    for (int i = 0; i < current_tile_n; i++) {
      DATA_TYPE mean = 0.0f;
      DATA_TYPE variance = 0.0f;

      // Compute mean
      for (int d = 0; d < dim; d++) {

        // OP: add
        mean += x_buffer[d * current_tile_n + i];
      }

      // OP: divide
      mean /= dim;
      STATISTICS_INC_ADD(stat, dim); // mean addition
      STATISTICS_INC_SRAM_READ(stat, dim * sizeof(DATA_TYPE)); // read x_buffer
      STATISTICS_INC_DIV(stat, 1);

      // Compute variance
      for (int d = 0; d < dim; d++) {
        // OP: sub
        DATA_TYPE diff = x_buffer[d * current_tile_n + i] - mean;
        // OP: square and add
        variance += diff * diff;
      }
      variance /= dim;
      STATISTICS_INC_MAC(stat, dim); // variance addition
      STATISTICS_INC_SRAM_READ(stat, dim * sizeof(DATA_TYPE)); // read x_buffer
      STATISTICS_INC_DIV(stat, 1); // variance division

      // Normalize and apply scale and bias
      for (int d = 0; d < dim; d++) {
        // OPs: normalize, scale, and bias
        DATA_TYPE normalized = (x_buffer[d * current_tile_n + i] - mean) / sqrtf(variance + eps);
        // OP: scale and bias
        out_buffer[d * current_tile_n + i] = normalized * w_buffer[d] + b_buffer[d];
      }

      STATISTICS_INC_MAC(stat, dim * 2); // scale and bias
      STATISTICS_INC_NON_LINEAR_OP(stat, dim); // sqrt
      STATISTICS_INC_ADD(stat, dim * 2); // mean subtraction and bias addition
      STATISTICS_INC_DIV(stat, dim); // variance normalization
    }

    // Write output tile back to DRAM
    for (int i = 0; i < current_tile_n; i++) {
      for (int d = 0; d < dim; d++) {
        out[d * n + i + n_tile] = out_buffer[d * current_tile_n + i];
        STATISTICS_INC_SRAM_TO_DRAM(stat, sizeof(DATA_TYPE)); // write out
      }
    }

    //
    // Compute in SRAM (end)
    //
  } // n_tile

  STATISTICS_SET_SRAM_SIZE(stat, sram.sram_size);
  STATISTICS_SET_SRAM_USED(stat, sram.used_size);

  DEBUG_LOG("SRAM usage: %zu bytes, SRAM size: %zu bytes (%.2f%% used)",
  sram.used_size, sram.sram_size, (float)sram.used_size / sram.sram_size * 100.0f);

  // Free buffers
  sram_free_all(&sram);
  // save statistics
  STATISTICS_APPEND_CSV(stat);
}