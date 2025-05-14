#include "model.h"
#include "statistic.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*
 * ============================
 * MACROS
 * ============================
 */

#define CONV_SIZE(tensor) ((tensor).height * (tensor).width * (tensor).channels)

#define CONV_SHAPE_COPY(to, from)    \
  do {                               \
    (to).height = (from).height;     \
    (to).width = (from).width;       \
    (to).channels = (from).channels; \
  } while (0)

#define CONVTENSOR_INDEX(H, W, c, h, w) (((H) * (W) * (c)) + ((h) * (W)) + (w))

#define CONVWEIGHT_INDEX(IC, H, W, oc, ic, h, w) \
  (((IC) * (H) * (W) * (oc)) + ((H) * (W) * (ic)) + ((h) * (W)) + (w))

#define BBOX_COORDS 4
#define CONV_OP_PER_BUTTLENECK 3
#define CONV_ID(resblock_id, conv_id) \
  ((conv_id) + CONV_OP_PER_BUTTLENECK * (resblock_id))

// DEBUG_LOG is used for debugging purposes
#ifdef DEBUG
#define DEBUG_LOG(fmt, ...)                                                 \
  fprintf(stderr, "[DEBUG_LOG]  %s:%d %s(): " fmt "\n", __FILE__, __LINE__, \
          __func__, ##__VA_ARGS__)
#else
#define DEBUG_LOG(fmt, ...)
#endif

// MALLOC_CHECK is used to check if memory allocation was successful
#define MALLOC_CHECK(ptr)                                                  \
  do {                                                                     \
    if (!(ptr)) {                                                          \
      fprintf(stderr, "malloc failed: %s:%d %s(): (" #ptr ")\n", __FILE__, \
              __LINE__, __func__);                                         \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  } while (0)

// MALLOC is a macro to allocate memory for a pointer of a given type and size
#define MALLOC(ptr, type, size)                   \
  do {                                            \
    (ptr) = (type*)malloc(sizeof(type) * (size)); \
    MALLOC_CHECK(ptr);                            \
  } while (0)

// FIND_TENSOR is a macro to find a tensor in the model by its name
#define NAME_BUFFER name_buf
#define NAME_BUFFER_SIZE 256
#define FIND_TENSOR(model, ptr, type, fmt, ...)                  \
  do {                                                           \
    char NAME_BUFFER[NAME_BUFFER_SIZE];                          \
    snprintf(NAME_BUFFER, NAME_BUFFER_SIZE, fmt, ##__VA_ARGS__); \
    (ptr) = (type*)find_tensor(NAME_BUFFER, model);              \
  } while (0)

// DUMP_TENSOR is a macro to dump a tensor to a file
#ifndef DUMP_TENSOR_DIR
#define DUMP_TENSOR_DIR "debug/"
#endif

#define BACKBONE_NAME "backbone.0.body"
#define DECODER_NAME "transformer.decoder"
#define ENCODER_NAME "transformer.encoder"


#ifdef DUMP
#define DUMP_TENSOR(ptr, type, size, fmt, ...)                  \
  do {                                                           \
    char NAME_BUFFER[NAME_BUFFER_SIZE];                          \
    snprintf(NAME_BUFFER, NAME_BUFFER_SIZE, "%s"fmt, DUMP_TENSOR_DIR ,##__VA_ARGS__); \
    dump_tensor(NAME_BUFFER, (type*)ptr, size);                        \
  } while (0)
#else
#define DUMP_TENSOR(ptr, type, size, fmt, ...)
#endif

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

  size_t chunk_size = 1024; // Define a chunk size to write in smaller pieces
  size_t remaining = size;
  const DATA_TYPE* current_ptr = tensor;

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
 * Utility Functions
 * ============================
 */

/**
 * Extracts an integer value from a JSON string in the format `"key": value`.
 *
 * @param key The key to search for in the JSON string.
 * @param json The JSON string to parse.
 * @return The extracted integer value, or 0 if the key is not found.
 */
int extract_int_value(const char* key, const char* json) {
  char* pos = strstr(json, key);
  if (!pos)
    return 0;
  pos = strchr(pos, ':');
  if (!pos)
    return 0;
  return atoi(pos + 1);
}

/**
 * Loads an entire JSON file into memory.
 *
 * @param filename The path to the JSON file.
 * @return A pointer to the loaded JSON string, or NULL if the file cannot be opened.
 */
char* load_json_file(const char* filename) {
  FILE* f = fopen(filename, "rb");
  if (!f)
    return NULL;

  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  rewind(f);

  char* buffer = NULL;
  MALLOC(buffer, char, len + 1);
  fread(buffer, 1, len, f);
  buffer[len] = '\0';
  fclose(f);
  return buffer;
}

/*
 * ============================
 * Configuration Parsing
 * ============================
 */

/**
 * Parses a ConvConfig block from a JSON section.
 *
 * @param section The JSON section containing the ConvConfig.
 * @param config The ConvConfig structure to populate.
 */
void parse_conv_config(const char* section, ConvConfig* config) {
  config->in_channels = extract_int_value("\"in_channels\"", section);
  config->out_channels = extract_int_value("\"out_channels\"", section);
  config->kernel_size = extract_int_value("\"kernel_size\"", section);
  config->stride = extract_int_value("\"stride\"", section);
  config->padding = extract_int_value("\"padding\"", section);
}

/**
 * Parses a MaxPoolConfig block from a JSON section.
 *
 * @param section The JSON section containing the MaxPoolConfig.
 * @param config The MaxPoolConfig structure to populate.
 */
void parse_maxpool_config(const char* section, MaxPoolConfig* config) {
  config->kernel_size = extract_int_value("\"kernel_size\"", section);
  config->stride = extract_int_value("\"stride\"", section);
  config->padding = extract_int_value("\"padding\"", section);
}

/**
 * Loads the DETR configuration from a JSON file.
 *
 * @param filename The path to the JSON configuration file.
 * @param detr The DETR structure to populate.
 * @return 0 on success, or a negative value on failure.
 */
int load_config(const char* filename, DETR* detr) {
  DETRConfig* config = &(detr->config);

  printf("load config file: %s\n", filename);
  char* json = load_json_file(filename);
  if (!json)
    return -1;

  // Top-level config values
  char* t = strstr(json, "\"imaege_size\"");
  config->image_width = extract_int_value("\"width\"", json);
  config->image_height = extract_int_value("\"height\"", json);
  config->image_channels = extract_int_value("\"channels\"", json);

  config->num_classes = extract_int_value("\"num_classes\"", json);
  config->num_boxes = extract_int_value("\"num_boxes\"", json);

  // Transformer config
  t = strstr(json, "\"transformer\"");
  config->transformer.dim = extract_int_value("\"dim\"", t);
  config->transformer.hidden_dim = extract_int_value("\"hidden_dim\"", t);
  config->transformer.n_encoder_layers =
      extract_int_value("\"n_encoder_layers\"", t);
  config->transformer.n_decoder_layers =
      extract_int_value("\"n_decoder_layers\"", t);
  config->transformer.n_heads = extract_int_value("\"n_heads\"", t);
  config->transformer.decoder_seq_len =
      extract_int_value("\"decoder_seq_len\"", t);

  // ResNet50 top conv1
  char* resnet = strstr(json, "\"resnet50\"");
  char* conv1 = strstr(resnet, "\"conv1\"");
  parse_conv_config(conv1, &config->resnet50.conv1);

  //maxpool
  char* maxpool = strstr(resnet, "\"maxpool\"");
  parse_maxpool_config(maxpool, &config->resnet50.maxpool);

  // ResBlock array (only handles 1 for simplicity)
  config->resnet50.num_resblock = extract_int_value("\"num_resblock\"", resnet);
  MALLOC(config->resnet50.resblock, ResidualConfig,
         config->resnet50.num_resblock);  // allocate for resblocks
  char* resblock = strstr(resnet, "\"resblock\"");
  if (!resblock)
    return -2;

  for (int i = 0; i < config->resnet50.num_resblock; i++) {
    // For simplicity, we’ll parse only one resblock from index 0
    ResidualConfig* rb = &config->resnet50.resblock[i];
    rb->num_bottleneck = extract_int_value("\"num_bottleneck\"", resblock);

    // Parse downsample
    char* downsample = strstr(resblock, "\"downsample\"");
    parse_conv_config(downsample, &rb->downsample);

    // Parse bottleneck convs (3 max)
    MALLOC(rb->conv, ConvConfig,
           rb->num_bottleneck * CONV_OP_PER_BUTTLENECK);  // allocate for convs
    char* conv_array = strstr(resblock, "\"conv\"");
    for (int j = 0; j < rb->num_bottleneck * CONV_OP_PER_BUTTLENECK; j++) {
      // crude jump to next object
      char* conv = strchr(conv_array, '{');
      if (!conv)
        break;
      conv_array = strchr(conv + 1, '}');
      if (!conv_array)
        break;

      size_t block_len = conv_array - conv + 1;
      char* conv_block = NULL;
      MALLOC(conv_block, char, block_len + 1);
      strncpy(conv_block, conv, block_len);
      conv_block[block_len] = '\0';

      parse_conv_config(conv_block, &(rb->conv[j]));
      free(conv_block);
    }
    resblock = conv_array;  // update string
  }

  // input_proj
  char* input_proj = strstr(resnet, "\"input_proj\"");
  parse_conv_config(input_proj, &config->resnet50.input_proj);

  free(json);
  return 0;
}

/**
 * Frees the memory allocated for the DETR configuration.
 *
 * @param detr The DETR structure containing the configuration to free.
 */
void free_config(DETR* detr) {
  DETRConfig* config = &(detr->config);
  if (!config)
    return;

  // Loop through each resblock
  for (int i = 0; i < config->resnet50.num_resblock; i++) {
    ResidualConfig* rb = &config->resnet50.resblock[i];

    // Free the conv array inside each resblock
    if (rb->conv) {
      free(rb->conv);
      rb->conv = NULL;
    }
  }

  // Free the resblock array itself
  if (config->resnet50.resblock) {
    free(config->resnet50.resblock);
    config->resnet50.resblock = NULL;
  }
}

/*
 * ============================
 * Weight Management
 * ============================
 */

/**
  * Generates mask and position embeddings for the DETR model.
  *
  * @param detr The DETR structure to populate with embeddings.
  * @return 0 on success, or a non-zero value on failure.
  */
int generate_mask_and_pos_embedding(DETR* detr) {
  DETRConfig* config = &(detr->config);
  DETRWeights* weights = &(detr->weights);

  int resnet_num_bottleneck = config->resnet50.num_resblock;
  int dim = config->transformer.dim;

  int interpolate_h = config->image_height / pow(2, resnet_num_bottleneck + 1);
  int interpolate_w = config->image_width / pow(2, resnet_num_bottleneck + 1);

  DEBUG_LOG("interpolate_h: %d, interpolate_w: %d", interpolate_h,
            interpolate_w);
  // transformer encoder seq len (from image size)
  int seq_len = interpolate_h * interpolate_w;
  config->transformer.encoder_seq_len = seq_len;

  // Allocate memory for attention mask
  MALLOC(weights->preprocess.att_mask, MASK_TYPE, seq_len);
  memset(weights->preprocess.att_mask, 1,
         seq_len * sizeof(MASK_TYPE));  // all 1s

  // Allocate memory for position embedding
  MALLOC(weights->preprocess.pos_embedding, DATA_TYPE, seq_len * dim);

  // Initialize position embedding (dim, seq_len)
  int num_pos_feats = dim / 2;  // Because it's split between X and Y

  double temperature = 10000.0;
  double eps = 1e-6;
  double scale = 2.0 * M_PI;

  for (int y = 0; y < interpolate_h; ++y) {
    for (int x = 0; x < interpolate_w; ++x) {
      int pos_index = y * interpolate_w + x;

      // Normalized positional encodings
      double y_norm = (double)(y + 1) / (double)(interpolate_h + eps);
      double x_norm = (double)(x + 1) / (double)(interpolate_w + eps);

      y_norm *= scale;
      x_norm *= scale;

      for (int i = 0; i < num_pos_feats; ++i) {
        double div_term = pow(temperature, (2.0 * (i / 2)) / num_pos_feats);

        double pos_x = x_norm / div_term;
        double pos_y = y_norm / div_term;

        // sin and cos interleaved - pos_x and pos_y
        if (i % 2 == 0) {
          weights->preprocess
              .pos_embedding[pos_index + (i + num_pos_feats) * seq_len] =
              (DATA_TYPE)sin(pos_x);
          weights->preprocess.pos_embedding[pos_index + (i)*seq_len] =
              (DATA_TYPE)sin(pos_y);
        } else {
          weights->preprocess
              .pos_embedding[pos_index + (i + num_pos_feats) * seq_len] =
              (DATA_TYPE)cos(pos_x);
          weights->preprocess.pos_embedding[pos_index + (i)*seq_len] =
              (DATA_TYPE)cos(pos_y);
        }
      }
    }
  }
  return 0;
}

/**
 * Finds a tensor by name in the DETR weight file.
 *
 * @param name The name of the tensor to find.
 * @param detr The DETR structure containing the weight file.
 * @return A pointer to the tensor data, or NULL if not found.
 */
void* find_tensor(const char* name, DETR* detr) {
  char* name_start = (char*)((size_t)detr->weight_mmap +
                             (size_t)detr->file_header.name_offset);
  void* data_start = (char*)((size_t)detr->weight_mmap +
                             (size_t)detr->file_header.data_offset);
  TensorInfo* info = (TensorInfo*)((size_t)detr->weight_mmap +
                                   (size_t)detr->file_header.info_offset);
  int nt = detr->file_header.num_tensor;
  char* tname;
  int data_size;
  void* r_ptr = NULL;

  for (int i = 0; i < nt; i++) {
    tname = name_start + info[i].name_offset;
    if (strcmp(tname, name) == 0) {
      r_ptr = data_start + info[i].data_offset;
      DEBUG_LOG("%s (%ld, +%u * %lu)", name,
                ((size_t)r_ptr - (size_t)detr->weight_mmap), info[i].data_size,
                sizeof(DATA_TYPE));
      break;
    }
  }
  if (r_ptr == NULL)
    fprintf(stderr, "Tensor not found: %s\n", name);
  return r_ptr;
}

/**
 * Lists all tensors in the DETR weight file.
 *
 * @param detr The DETR structure containing the weight file.
 */
void list_tensor(DETR* detr) {
  char* name_start = (char*)((size_t)detr->weight_mmap +
                             (size_t)detr->file_header.name_offset);
  TensorInfo* info = (TensorInfo*)((size_t)detr->weight_mmap +
                                   (size_t)detr->file_header.info_offset);
  int nt = detr->file_header.num_tensor;
  char* name;
  int data_size;

  for (int i = 0; i < nt; i++) {
    name = name_start + info[i].name_offset;
    data_size = info[i].data_size;
    printf("[%3d] %64s (%d)\n", i, name, data_size);
  }
}

/**
 * Loads the weights from a file into the DETR structure.
 *
 * @param filename The path to the weight file.
 * @param detr The DETR structure to populate.
 * @return 0 on success, or a non-zero value on failure.
 */
int load_weight(const char* filename, DETR* detr) {
  DETRConfig* config = &(detr->config);
  DETRWeights* weights = &(detr->weights);

  /*pre process*/
  if (generate_mask_and_pos_embedding(detr)) {
    fprintf(stderr, "generate mask and pos embedding failed\n");
    return -1;
  }

  printf("load weights file: %s\n", filename);

  /* open file */
  detr->weight_fp = fopen(filename, "rb");
  if (!detr->weight_fp) {
    fprintf(stderr, "open file failed: %s\n", filename);
    return 1;
  }

  /* map */
  fseek(detr->weight_fp, 0, SEEK_END);       // seek to end of file
  detr->mmap_size = ftell(detr->weight_fp);  // get current file pointer
  fseek(detr->weight_fp, 0, SEEK_SET);       // seek back to beginning of file

  // Map file into memory
  detr->weight_mmap = mmap(NULL, detr->mmap_size, PROT_READ, MAP_PRIVATE,
                           fileno(detr->weight_fp), 0);
  if (detr->weight_mmap == MAP_FAILED) {
    fprintf(stderr, "mmap\n");
    fclose(detr->weight_fp);
    return 1;
  }

  // parse header
  TensorFile* h = (TensorFile*)detr->weight_mmap;
  detr->file_header = *h;

  printf("version: %d\n", detr->file_header.version);
  printf("num_tensor: %d\n", detr->file_header.num_tensor);
  printf("pack_method: %d\n", detr->file_header.pack_method);
  printf("info_offset: %d\n", detr->file_header.info_offset);
  printf("name_offset: %d\n", detr->file_header.name_offset);
  printf("data_offset: %d\n", detr->file_header.data_offset);
  printf("total size: %ld\n", detr->mmap_size);

  /* point weights */
  int n_encoder_layers = config->transformer.n_encoder_layers;
  int n_decoder_layers = config->transformer.n_decoder_layers;
  int dim = config->transformer.dim;
  DATA_TYPE* in_proj_weight;
  DATA_TYPE* in_proj_bias;

  // transformer

  weights->encoder.pos_embedding =
      weights->preprocess.pos_embedding;  // (dim, seq_len)
  // weights for matmuls. note dim == n_heads * head_size

  weights->encoder.att_mask =
      weights->preprocess.att_mask;  // (seq_len)

  MALLOC(weights->encoder.layer, EncoderLayerWeights, n_encoder_layers);
  for (int i = 0; i < n_encoder_layers; i++) {
    // in_proj_weight, bias
    FIND_TENSOR(detr, in_proj_weight, DATA_TYPE,
                "transformer.encoder.layers.%d.self_attn.in_proj_weight", i);

    weights->encoder.layer[i].wq =
        in_proj_weight;  // (layer, dim, n_heads * head_size)
    weights->encoder.layer[i].wk =
        in_proj_weight + dim * dim;  // (layer, dim,  n_heads * head_size)
    weights->encoder.layer[i].wv =
        in_proj_weight + 2 * dim * dim;  // (layer, dim,  n_heads * head_size)

    FIND_TENSOR(detr, in_proj_bias, DATA_TYPE,
                "transformer.encoder.layers.%d.self_attn.in_proj_bias", i);

    weights->encoder.layer[i].bq = in_proj_bias;            // (layer, dim)
    weights->encoder.layer[i].bk = in_proj_bias + dim;      // (layer, dim)
    weights->encoder.layer[i].bv = in_proj_bias + 2 * dim;  // (layer, dim)

    // out proj
    FIND_TENSOR(detr, weights->encoder.layer[i].wo, DATA_TYPE,
                "transformer.encoder.layers.%d.self_attn.out_proj.weight",
                i);  // (layer, dim,  dim)

    FIND_TENSOR(detr, weights->encoder.layer[i].bo, DATA_TYPE,
                "transformer.encoder.layers.%d.self_attn.out_proj.bias",
                i);  // (layer, dim)

    // weights for layernorm
    FIND_TENSOR(detr, weights->encoder.layer[i].wln1, DATA_TYPE,
                "transformer.encoder.layers.%d.norm1.weight",
                i);  // (layer, dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].bln1, DATA_TYPE,
                "transformer.encoder.layers.%d.norm1.bias", i);  // (layer, dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].wln2, DATA_TYPE,
                "transformer.encoder.layers.%d.norm2.weight",
                i);  // (layer, dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].bln2, DATA_TYPE,
                "transformer.encoder.layers.%d.norm2.bias", i);  // (layer, dim)
    // weights for ffn
    FIND_TENSOR(detr, weights->encoder.layer[i].w1, DATA_TYPE,
                "transformer.encoder.layers.%d.linear1.weight",
                i);  // (layer, dim, hidden_dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].b1, DATA_TYPE,
                "transformer.encoder.layers.%d.linear1.bias",
                i);  // (layer, hidden_dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].w2, DATA_TYPE,
                "transformer.encoder.layers.%d.linear2.weight",
                i);  // (layer, hidden_dim, dim)
    FIND_TENSOR(detr, weights->encoder.layer[i].b2, DATA_TYPE,
                "transformer.encoder.layers.%d.linear2.bias",
                i);  // (layer, dim)
  }
  // position embedding
  weights->decoder.pos_embedding =
  weights->preprocess.pos_embedding;  // (dim, seq_len)

  weights->decoder.att_mask =
  weights->preprocess.att_mask;  // (seq_len)

  // query position embedding
  FIND_TENSOR(detr, weights->decoder.query_pos_embedding, DATA_TYPE,
    "query_embed.weight");  // (seq_len, dim)

  FIND_TENSOR(detr, weights->decoder.wln, DATA_TYPE,
    "transformer.decoder.norm.weight");  // (dim)

  FIND_TENSOR(detr, weights->decoder.bln, DATA_TYPE,
    "transformer.decoder.norm.bias");  // (dim)

  MALLOC(weights->decoder.layer, DecoderLayerWeights, n_decoder_layers);
  for (int i = 0; i < n_decoder_layers; i++) {
    // weights for matmuls. note dim == n_heads * head_size
    FIND_TENSOR(detr, in_proj_weight, DATA_TYPE,
                "transformer.decoder.layers.%d.self_attn.in_proj_weight", i);
    weights->decoder.layer[i].wq =
        in_proj_weight;  // (layer, dim, n_heads * head_size)
    weights->decoder.layer[i].wk =
        in_proj_weight + dim * dim;  // (layer, dim,  n_heads * head_size)
    weights->decoder.layer[i].wv =
        in_proj_weight + 2 * dim * dim;  // (layer, dim,  n_heads * head_size)

    FIND_TENSOR(detr, in_proj_bias, DATA_TYPE,
                "transformer.decoder.layers.%d.self_attn.in_proj_bias", i);
    weights->decoder.layer[i].bq = in_proj_bias;            // (layer, dim)
    weights->decoder.layer[i].bk = in_proj_bias + dim;      // (layer, dim)
    weights->decoder.layer[i].bv = in_proj_bias + 2 * dim;  // (layer, dim)

    FIND_TENSOR(detr, weights->decoder.layer[i].wo, DATA_TYPE,
                "transformer.decoder.layers.%d.self_attn.out_proj.weight",
                i);  //  (layer, dim,  dim)

    FIND_TENSOR(detr, weights->decoder.layer[i].bo, DATA_TYPE,
                "transformer.decoder.layers.%d.self_attn.out_proj.bias",
                i);  // (layer, dim)

    // weights for matmuls. note dim == n_heads * head_size
    FIND_TENSOR(detr, in_proj_weight, DATA_TYPE,
                "transformer.decoder.layers.%d.multihead_attn.in_proj_weight",
                i);

    weights->decoder.layer[i].wq2 =
        in_proj_weight;  // (layer, dim, n_heads * head_size)
    weights->decoder.layer[i].wk2 =
        in_proj_weight + dim * dim;  // (layer, dim,  n_heads * head_size)
    weights->decoder.layer[i].wv2 =
        in_proj_weight + 2 * dim * dim;  // (layer, dim,  n_heads * head_size)

    FIND_TENSOR(detr, in_proj_bias, DATA_TYPE,
                "transformer.decoder.layers.%d.multihead_attn.in_proj_bias", i);

    weights->decoder.layer[i].bq2 = in_proj_bias;            // (layer, dim)
    weights->decoder.layer[i].bk2 = in_proj_bias + dim;      // (layer, dim)
    weights->decoder.layer[i].bv2 = in_proj_bias + 2 * dim;  // (layer, dim)

    FIND_TENSOR(detr, weights->decoder.layer[i].wo2, DATA_TYPE,
                "transformer.decoder.layers.%d.multihead_attn.out_proj.weight",
                i);

    FIND_TENSOR(detr, weights->decoder.layer[i].bo2, DATA_TYPE,
                "transformer.decoder.layers.%d.multihead_attn.out_proj.bias",
                i);
    // weights for layernorm
    FIND_TENSOR(detr, weights->decoder.layer[i].wln1, DATA_TYPE,
                "transformer.decoder.layers.%d.norm1.weight",
                i);  // (layer, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].bln1, DATA_TYPE,
                "transformer.decoder.layers.%d.norm1.bias", i);  // (layer, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].wln2, DATA_TYPE,
                "transformer.decoder.layers.%d.norm2.weight",
                i);  // (layer, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].bln2, DATA_TYPE,
                "transformer.decoder.layers.%d.norm2.bias", i);  // (layer, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].wln3, DATA_TYPE,
                "transformer.decoder.layers.%d.norm3.weight",
                i);  // (layer, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].bln3, DATA_TYPE,
                "transformer.decoder.layers.%d.norm3.bias", i);  // (layer, dim)
    // weights for ffn
    FIND_TENSOR(detr, weights->decoder.layer[i].w1, DATA_TYPE,
                "transformer.decoder.layers.%d.linear1.weight",
                i);  // (layer, dim, hidden_dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].b1, DATA_TYPE,
                "transformer.decoder.layers.%d.linear1.bias",
                i);  // (layer, hidden_dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].w2, DATA_TYPE,
                "transformer.decoder.layers.%d.linear2.weight",
                i);  // (layer, hidden_dim, dim)
    FIND_TENSOR(detr, weights->decoder.layer[i].b2, DATA_TYPE,
                "transformer.decoder.layers.%d.linear2.bias",
                i);  // (layer, dim)
  }

  // output embed
  FIND_TENSOR(detr, weights->outputembed.class_w, DATA_TYPE,
              "class_embed.weight");  // (num_classes, dim)
  FIND_TENSOR(detr, weights->outputembed.class_b, DATA_TYPE,
              "class_embed.bias");  // (num_classes)
  FIND_TENSOR(detr, weights->outputembed.bbox_w1, DATA_TYPE,
              "bbox_embed.layers.0.weight");  // (dim, dim)
  FIND_TENSOR(detr, weights->outputembed.bbox_b1, DATA_TYPE,
              "bbox_embed.layers.0.bias");  // (dim)
  FIND_TENSOR(detr, weights->outputembed.bbox_w2, DATA_TYPE,
              "bbox_embed.layers.1.weight");  // (dim, dim)
  FIND_TENSOR(detr, weights->outputembed.bbox_b2, DATA_TYPE,
              "bbox_embed.layers.1.bias");  // (dim)
  FIND_TENSOR(detr, weights->outputembed.bbox_w3, DATA_TYPE,
              "bbox_embed.layers.2.weight");  // (4, dim)
  FIND_TENSOR(detr, weights->outputembed.bbox_b3, DATA_TYPE,
              "bbox_embed.layers.2.bias");  // (4)

  // resnet50
  FIND_TENSOR(detr, weights->resnet50.conv1.weight, DATA_TYPE,
              "backbone.0.body.conv1.weight");
  weights->resnet50.conv1.bias = NULL;  // reserved for future use
  FIND_TENSOR(detr, weights->resnet50.bn1.weight, DATA_TYPE,
              "backbone.0.body.bn1.weight");
  FIND_TENSOR(detr, weights->resnet50.bn1.bias, DATA_TYPE,
              "backbone.0.body.bn1.bias");
  FIND_TENSOR(detr, weights->resnet50.bn1.running_mean, DATA_TYPE,
              "backbone.0.body.bn1.running_mean");
  FIND_TENSOR(detr, weights->resnet50.bn1.running_var, DATA_TYPE,
              "backbone.0.body.bn1.running_var");

  int num_resblock = config->resnet50.num_resblock;
  MALLOC(weights->resnet50.resblock, ResidualWeights, num_resblock);
  for (int i = 0; i < num_resblock; i++) {
    ResidualWeights* rb = &weights->resnet50.resblock[i];

    int num_bottleneck = config->resnet50.resblock[i].num_bottleneck;

    // Free the conv array inside each resblock
    FIND_TENSOR(detr, rb->downsample.weight, DATA_TYPE,
                "backbone.0.body.layer%d.0.downsample.0.weight", i + 1);
    rb->downsample.bias = NULL;  // reserved for future use
    FIND_TENSOR(detr, rb->bn_downsample.weight, DATA_TYPE,
                "backbone.0.body.layer%d.0.downsample.1.weight", i + 1);
    FIND_TENSOR(detr, rb->bn_downsample.bias, DATA_TYPE,
                "backbone.0.body.layer%d.0.downsample.1.bias", i + 1);
    FIND_TENSOR(detr, rb->bn_downsample.running_mean, DATA_TYPE,
                "backbone.0.body.layer%d.0.downsample.1.running_mean", i + 1);
    FIND_TENSOR(detr, rb->bn_downsample.running_var, DATA_TYPE,
                "backbone.0.body.layer%d.0.downsample.1.running_var", i + 1);

    MALLOC(rb->conv, ConvWeights, num_bottleneck * CONV_OP_PER_BUTTLENECK);
    MALLOC(rb->bn, BatchNormWeights, num_bottleneck * CONV_OP_PER_BUTTLENECK);
    for (int j = 0; j < num_bottleneck; j++) {
      for (int k = 0; k < CONV_OP_PER_BUTTLENECK; k++) {
        FIND_TENSOR(detr, rb->conv[CONV_ID(j, k)].weight, DATA_TYPE,
                    "backbone.0.body.layer%d.%d.conv%d.weight", i + 1, j,
                    k + 1);
        rb->conv[CONV_ID(j, k)].bias = NULL;  // reserved for future use
        FIND_TENSOR(detr, rb->bn[CONV_ID(j, k)].weight, DATA_TYPE,
                    "backbone.0.body.layer%d.%d.bn%d.weight", i + 1, j, k + 1);
        FIND_TENSOR(detr, rb->bn[CONV_ID(j, k)].bias, DATA_TYPE,
                    "backbone.0.body.layer%d.%d.bn%d.bias", i + 1, j, k + 1);
        FIND_TENSOR(detr, rb->bn[CONV_ID(j, k)].running_mean, DATA_TYPE,
                    "backbone.0.body.layer%d.%d.bn%d.running_mean", i + 1, j,
                    k + 1);
        FIND_TENSOR(detr, rb->bn[CONV_ID(j, k)].running_var, DATA_TYPE,
                    "backbone.0.body.layer%d.%d.bn%d.running_var", i + 1, j,
                    k + 1);
      }
    }
  }

  FIND_TENSOR(detr, weights->resnet50.input_proj.weight, DATA_TYPE,
              "input_proj.weight");
  FIND_TENSOR(detr, weights->resnet50.input_proj.bias, DATA_TYPE,
              "input_proj.bias");
  return 0;
}

/**
 * Frees the memory allocated for the DETR weights.
 *
 * @param detr The DETR structure containing the weights to free.
 */
void free_weights(DETR* detr) {
  DETRConfig* config = &(detr->config);
  DETRWeights* weights = &(detr->weights);

  if (!weights)
    return;

  if(weights->preprocess.att_mask) {
    free(weights->preprocess.att_mask);
    weights->preprocess.att_mask = NULL;
  }
  if(weights->preprocess.pos_embedding) {
    free(weights->preprocess.pos_embedding);
    weights->preprocess.pos_embedding = NULL;
  }

  if (weights->encoder.layer) {
    free(weights->encoder.layer);
    weights->encoder.layer = NULL;
  }

  if (weights->decoder.layer) {
    free(weights->decoder.layer);
    weights->decoder.layer = NULL;
  }

  // Loop through each resblock
  for (int i = 0; i < config->resnet50.num_resblock; i++) {
    ResidualWeights* rb = &weights->resnet50.resblock[i];

    // Free the conv array inside each resblock
    if (rb->conv) {
      free(rb->conv);
      rb->conv = NULL;
    }
    if (rb->bn) {
      free(rb->bn);
      rb->bn = NULL;
    }
  }

  // Free the resblock array itself
  if (weights->resnet50.resblock) {
    free(weights->resnet50.resblock);
    weights->resnet50.resblock = NULL;
  }

  /* unmap */
  if (detr->weight_mmap)
    munmap(detr->weight_mmap, detr->mmap_size);

  /* close file */
  if (detr->weight_fp)
    fclose(detr->weight_fp);
}

/**
 * Loads an input tensor from a file.
 *
 * @param r The ConvolutionTensor to load the data into.
 * @param filename The name of the tensor to load.
 */
void load_input_tensor(ConvolutionTensor* r, const char* filename){
  FILE* fp = fopen(filename, "rb");
  if (!fp) {
    fprintf(stderr, "Failed to open file: %s\n", filename);
    return;
  }
  size_t read_size = fread(r->x, sizeof(DATA_TYPE), CONV_SIZE(*r), fp);
  if (read_size != CONV_SIZE(*r)) {
    fprintf(stderr, "Failed to read tensor data from file: %s\n", filename);
    fclose(fp);
    return;
  }
  fclose(fp);
  DEBUG_LOG("Loaded input tensor from file: %s", filename);
}

/**
  * Saves the output tensor to a file.
  *
  * @param r The OutputTensor to save.
  * @param boxes_filename The name of the boxes tensor to save.
  * @param logits_filename The name of the logits tensor to save.
  */

void save_output_tensor(OutputTensor* r, const char* boxes_filename, const char* logits_filename){
  FILE* fp = fopen(logits_filename, "wb");
  if (!fp) {
    fprintf(stderr, "Failed to open file: %s\n", logits_filename);
    return;
  }
  size_t write_size = fwrite(r->classes, sizeof(DATA_TYPE),
                              r->num_boxes * r->num_classes, fp);
  if (write_size != r->num_boxes * r->num_classes) {
    fprintf(stderr, "Failed to write tensor data to file: %s\n", logits_filename);
    fclose(fp);
    return;
  }
  fclose(fp);

  fp = fopen(boxes_filename, "wb");
  if (!fp) {
    fprintf(stderr, "Failed to open file: %s\n", boxes_filename);
    return;
  }
  write_size = fwrite(r->bbox, sizeof(DATA_TYPE), r->num_boxes * BBOX_COORDS,
                      fp);
  if (write_size != r->num_boxes * BBOX_COORDS) {
    fprintf(stderr, "Failed to write tensor data to file: %s\n", boxes_filename);
    fclose(fp);
    return;
  }
  fclose(fp);
  DEBUG_LOG("Saved output logits tensor to file: %s", logits_filename);
  DEBUG_LOG("Saved output boxes tensor to file: %s", boxes_filename);
}

/*
 * ============================
 * Initialization and Cleanup
 * ============================
 */

/**
 * Initializes the DETR model by loading configuration and weights.
 *
 * @param detr The DETR structure to initialize.
 * @param config_file The path to the configuration file.
 * @param weights_file The path to the weights file.
 * @return 0 on success, or a non-zero value on failure.
 */
int init_detr(DETR* detr, const char* config_file, const char* weights_file) {
  int r;
  r = load_config(config_file, detr);
  if (r) {
    fprintf(stderr, "[DETR INIT] load config failed.\n");
    return r;
  }
  r = load_weight(weights_file, detr);
  if (r) {
    fprintf(stderr, "[DETR INIT] load weights failed.\n");
    return r;
  }

  DEBUG_LOG("DETR initialized with config and weights.");
  return 0;
}

/**
 * Frees the memory allocated for the DETR model.
 *
 * @param detr The DETR structure to free.
 */
void free_detr(DETR* detr) {
  free_weights(detr);
  free_config(detr);
  DEBUG_LOG("DETR freed");
}

/*
 * ============================
 * Tensor Memory Management
 * ============================
 */

/**
 * Allocates memory for a ConvolutionTensor.
 *
 * @param r The ConvolutionTensor to allocate memory for.
 */
void malloc_conv2D_tensor(ConvolutionTensor* r) {
  if (r->x != NULL) {
    free(r->x);
    r->x = NULL;
  }

  MALLOC(r->x, DATA_TYPE, CONV_SIZE(*r));
}

/**
 * Frees the memory allocated for a ConvolutionTensor.
 *
 * @param r The ConvolutionTensor to free.
 */
void free_conv2D_tensor(ConvolutionTensor* r) {
  if (r->x == NULL)
    return;
  free(r->x);
  r->x = NULL;
}

/**
 * Allocates memory for an OutputTensor.
 *
 * @param r The OutputTensor to allocate memory for.
 */
void malloc_output_tensor(OutputTensor* r) {
  MALLOC(
      r->classes, DATA_TYPE,
      r->num_boxes * r->num_classes);  // Allocate memory for class predictions
  MALLOC(r->bbox, DATA_TYPE,
         r->num_boxes *
             BBOX_COORDS);  // Allocate memory for bounding box coordinates
}

/**
 * Frees the memory allocated for an OutputTensor.
 *
 * @param r The OutputTensor to free.
 */
void free_output_tensor(OutputTensor* r) {
  if (r->classes != NULL) {
    free(r->classes);
    r->classes = NULL;
  }
  if (r->bbox != NULL) {
    free(r->bbox);
    r->bbox = NULL;
  }
}

/*
 * ============================
 * Runstate Management
 * ============================
 */

/**
 * Allocates memory for an EncoderRunState.
 *
 * @param config The TransformerConfig structure.
 * @param r The EncoderRunState to allocate memory for.
 */
void malloc_encoder_run_state(const TransformerConfig* config,
                              EncoderRunState* r) {
  int dim = config->dim;
  int hidden_dim = config->hidden_dim;
  int seq_len = config->encoder_seq_len;
  int n_heads = config->n_heads;

  MALLOC(r->x, DATA_TYPE, dim * seq_len);
  MALLOC(r->xb, DATA_TYPE, dim * seq_len);
  MALLOC(r->xb2, DATA_TYPE, dim * seq_len);
  MALLOC(r->hb, DATA_TYPE, hidden_dim * seq_len);
  MALLOC(r->hb2, DATA_TYPE, hidden_dim * seq_len);
  MALLOC(r->q, DATA_TYPE, dim * seq_len);
  MALLOC(r->k, DATA_TYPE, dim * seq_len);
  MALLOC(r->v, DATA_TYPE, dim * seq_len);
  MALLOC(r->att, DATA_TYPE, n_heads * seq_len * seq_len);
}

/**
 * Frees the memory allocated for an EncoderRunState.
 *
 * @param r The EncoderRunState to free.
 */
void free_encoder_run_state(EncoderRunState* r) {
  free(r->x);
  r->x = NULL;
  free(r->xb);
  r->xb = NULL;
  free(r->xb2);
  r->xb2 = NULL;
  free(r->hb);
  r->hb = NULL;
  free(r->hb2);
  r->hb2 = NULL;
  free(r->q);
  r->q = NULL;
  free(r->k);
  r->k = NULL;
  free(r->v);
  r->v = NULL;
  free(r->att);
  r->att = NULL;
}

/**
 * Allocates memory for a DecoderRunState.
 *
 * @param config The TransformerConfig structure.
 * @param r The DecoderRunState to allocate memory for.
 */
void malloc_decoder_run_state(const TransformerConfig* config,
                              DecoderRunState* r) {
  int dim = config->dim;
  int hidden_dim = config->hidden_dim;
  int seq_len = config->decoder_seq_len;
  int encoder_seq_len = config->encoder_seq_len;
  int n_heads = config->n_heads;

  MALLOC(r->x, DATA_TYPE, dim * seq_len);
  MALLOC(r->xb, DATA_TYPE, dim * seq_len);
  MALLOC(r->xb2, DATA_TYPE, dim * seq_len);
  MALLOC(r->hb, DATA_TYPE, hidden_dim * seq_len);
  MALLOC(r->hb2, DATA_TYPE, hidden_dim * seq_len);
  MALLOC(r->q, DATA_TYPE, dim * seq_len);
  if (seq_len < encoder_seq_len) {
    MALLOC(r->k, DATA_TYPE, dim * encoder_seq_len);  // select bigger size
    MALLOC(r->v, DATA_TYPE, dim * encoder_seq_len);  // select bigger size
    MALLOC(r->att, DATA_TYPE, n_heads * seq_len * encoder_seq_len);
  } else {
    MALLOC(r->k, DATA_TYPE, dim * seq_len);  // select bigger size
    MALLOC(r->v, DATA_TYPE, dim * seq_len);  // select bigger size
    MALLOC(r->att, DATA_TYPE, n_heads * seq_len * seq_len);
  }

  MALLOC(r->f, DATA_TYPE, dim * encoder_seq_len);
  MALLOC(r->f_embed, DATA_TYPE, dim * encoder_seq_len);
}

/**
 * Frees the memory allocated for a DecoderRunState.
 *
 * @param r The DecoderRunState to free.
 */
void free_decoder_run_state(DecoderRunState* r) {
  free(r->x);
  r->x = NULL;
  free(r->xb);
  r->xb = NULL;
  free(r->xb2);
  r->xb2 = NULL;
  free(r->hb);
  r->hb = NULL;
  free(r->hb2);
  r->hb2 = NULL;
  free(r->q);
  r->q = NULL;
  free(r->k);
  r->k = NULL;
  free(r->v);
  r->v = NULL;
  free(r->att);
  r->att = NULL;
  free(r->f);
  r->f = NULL;
  free(r->f_embed);
  r->f_embed = NULL;
}

/**
 * Allocates memory for an OutputRunState.
 *
 * @param config The DETRConfig structure.
 * @param r The OutputRunState to allocate memory for.
 */
void malloc_output_run_state(const DETRConfig* config, OutputRunState* r) {
  int dim = config->transformer.dim;
  int num_boxes = config->num_boxes;
  int num_classes = config->num_classes;

  MALLOC(r->x, DATA_TYPE, dim * num_boxes);
  MALLOC(r->xb, DATA_TYPE, dim * num_boxes);
  MALLOC(r->classes, DATA_TYPE, num_classes * num_boxes);
  MALLOC(r->bbox, DATA_TYPE, BBOX_COORDS * num_boxes);
}

/**
 * Frees the memory allocated for an OutputRunState.
 *
 * @param r The OutputRunState to free.
 */
void free_output_run_state(OutputRunState* r) {
  free(r->x);
  r->x = NULL;
  free(r->xb);
  r->xb = NULL;
  free(r->classes);
  r->classes = NULL;
  free(r->bbox);
  r->bbox = NULL;
}

/*
 * ============================
 * Model Inference
 * ============================
 */

/**
 * Runs the forward pass of the DETR model.
 *
 * @param detr The DETR structure containing the model.
 * @param image The input image as a ConvolutionTensor.
 * @param result The output result as an OutputTensor.
 */
void forward(DETR* detr, ConvolutionTensor* image, OutputTensor* result) {
  STATISTICS_INIT_CSV;
  // placeholder
  ConvolutionTensor resnet50_out = CONVTENSOR_INITIALIZER;
  EncoderRunState encoder_runstate = {NULL};
  DecoderRunState decoder_runstate = {NULL};
  OutputRunState output_runstate = {NULL};
  size_t feature_size;
  TransformerConfig* t_cfg = &(detr->config.transformer);
  DETRConfig* cfg = &(detr->config);

  // Run Resnet50 backbone
  DEBUG_LOG("--------------------- Resnet50 Backbone");
  forward_resnet50(&(detr->config.resnet50), &(detr->weights.resnet50), image,
                   &resnet50_out);  // Run ResNet50

  // Copy features from backbone to enocder
  malloc_encoder_run_state(t_cfg, &encoder_runstate);
  memcpy(encoder_runstate.x, resnet50_out.x,
         CONV_SIZE(resnet50_out) * sizeof(DATA_TYPE));  // Copy
  free_conv2D_tensor(&resnet50_out);                    // release memory

  // Run Transformer Encoder
  DEBUG_LOG("--------------------- Transformer - Encoder");
  forward_encoder(t_cfg, &(detr->weights.encoder),
                  &encoder_runstate);

  // Copy features from encoder to deocder
  malloc_decoder_run_state(t_cfg, &decoder_runstate);
  feature_size = t_cfg->encoder_seq_len * t_cfg->dim * sizeof(DATA_TYPE);
  memcpy(decoder_runstate.f, encoder_runstate.x,
         feature_size);                               // Copy
  free_encoder_run_state(&encoder_runstate);          // release memory

  // Init decoder tokens to zeros
  feature_size = t_cfg->decoder_seq_len * t_cfg->dim * sizeof(DATA_TYPE);
  memset(decoder_runstate.x, 0, feature_size);

  // Run Transformer Decoder
  DEBUG_LOG("--------------------- Transformer - Decoder");
  forward_decoder(t_cfg, &(detr->weights.decoder),
                  &decoder_runstate);

  // Copy features from decoder to output embed
  malloc_output_run_state(cfg, &output_runstate);
  feature_size = detr->config.num_boxes * t_cfg->dim * sizeof(DATA_TYPE);
  memcpy(output_runstate.x, decoder_runstate.x,
         feature_size);                               // Copy
  free_decoder_run_state(&decoder_runstate);          // release memory


  // Run Output Embedded
  DEBUG_LOG("--------------------- Output Embedded");
  forward_output(&(detr->config), &(detr->weights.outputembed),
                 &output_runstate);

  // copy output with Transpose (dim, n) -> (n ,dim)
  for (int nb = 0; nb < result->num_boxes; nb++) {
    for (int nc = 0; nc < result->num_classes; nc++) {
      result->classes[nb * result->num_classes + nc] =
          output_runstate.classes[nc * result->num_boxes + nb];
    }
    for (int bc = 0; bc < BBOX_COORDS; bc++) {
      result->bbox[nb * BBOX_COORDS + bc] =
          output_runstate.bbox[bc * result->num_boxes + nb];
    }
  }

  free_output_run_state(&output_runstate);
}

/**
 * Runs the forward pass of the ResNet50 backbone.
 *
 * @param config The ResNet50Config structure.
 * @param weights The ResNet50Weights structure.
 * @param image The input image as a ConvolutionTensor.
 * @param result The output result as a ConvolutionTensor.
 */
void forward_resnet50(ResNet50Config* config, ResNet50Weights* weights,
                      ConvolutionTensor* image, ConvolutionTensor* result) {
  // tensor placeholder
  ConvolutionTensor r = CONVTENSOR_INITIALIZER;
  ConvolutionTensor r_1 = CONVTENSOR_INITIALIZER;
  ConvolutionTensor r_2 = CONVTENSOR_INITIALIZER;
  ConvolutionTensor downsample = CONVTENSOR_INITIALIZER;

  // 7*7 conv 2D
  DEBUG_LOG("---------------------conv1");
  conv2D(&r_1, image, &(config->conv1), &(weights->conv1));
  DUMP_TENSOR(r_1.x, DATA_TYPE, CONV_SIZE(r_1), "%s.%s", BACKBONE_NAME, "conv1");
  batchnorm2D(&r_1, &r_1, &(weights->bn1));  // with batchnorm2D
  relu(r_1.x, r_1.x, CONV_SIZE(r_1));
  maxpooling2D(&r, &r_1, &(config->maxpool));

  // residul blocks
  for (int rb = 0; rb < config->num_resblock; rb++) {  // residual block
    for (int nb = 0; nb < config->resblock[rb].num_bottleneck;
         nb++) {  // num of bottleneck
      DEBUG_LOG("---------------------resblock [layer%d.%d]", rb + 1, nb);
      // Bottleneck
      conv2D(&r_1, &r, &(config->resblock[rb].conv[CONV_ID(nb, 0)]),
             &(weights->resblock[rb].conv[CONV_ID(nb, 0)]));
      DUMP_TENSOR(r_1.x, DATA_TYPE, CONV_SIZE(r_1), "%s.layer%d.%d.conv1", BACKBONE_NAME, rb + 1,
                  nb);

      batchnorm2D(
          &r_1, &r_1,
          &(weights->resblock[rb].bn[CONV_ID(nb, 0)]));  // with batchnorm2D

      relu(r_1.x, r_1.x, CONV_SIZE(r_1));

      conv2D(&r_2, &r_1, &(config->resblock[rb].conv[CONV_ID(nb, 1)]),
             &(weights->resblock[rb].conv[CONV_ID(nb, 1)]));
      DUMP_TENSOR(r_2.x, DATA_TYPE, CONV_SIZE(r_2), "%s.layer%d.%d.conv2", BACKBONE_NAME, rb + 1,
             nb);
      batchnorm2D(
          &r_2, &r_2,
          &(weights->resblock[rb].bn[CONV_ID(nb, 1)]));  // with batchnorm2D

      relu(r_2.x, r_2.x, CONV_SIZE(r_2));

      conv2D(&r_1, &r_2, &(config->resblock[rb].conv[CONV_ID(nb, 2)]),
             &(weights->resblock[rb].conv[CONV_ID(nb, 2)]));
      DUMP_TENSOR(r_1.x, DATA_TYPE, CONV_SIZE(r_1), "%s.layer%d.%d.conv3", BACKBONE_NAME, rb + 1,
                  nb);
      batchnorm2D(
          &r_1, &r_1,
          &(weights->resblock[rb].bn[CONV_ID(nb, 2)]));  // with batchnorm2D

      // downsample
      if (nb == 0) {
        conv2D(&downsample, &r, &(config->resblock[rb].downsample),
               &(weights->resblock[rb].downsample));
        DUMP_TENSOR(downsample.x, DATA_TYPE, CONV_SIZE(downsample),
                    "%s.layer%d.0.downsample.0", BACKBONE_NAME, rb + 1);
        batchnorm2D(
            &downsample, &downsample,
            &(weights->resblock[rb].bn_downsample));  // with batchnorm2D
        DUMP_TENSOR(downsample.x, DATA_TYPE, CONV_SIZE(downsample),
                    "%s.layer%d.0.downsample", BACKBONE_NAME, rb + 1);
        CONV_SHAPE_COPY(r, r_1);
        malloc_conv2D_tensor(&r);
        add(r.x, r_1.x, downsample.x,
            CONV_SIZE(r_1));  // add(input1, intput2, output)

        // free tensor
        free_conv2D_tensor(&downsample);
      } else {
        add(r.x, r_1.x, r.x, CONV_SIZE(r_1));  // add(input1, intput2, output)
      }

      relu(r.x, r.x, CONV_SIZE(r));

      // free tensor
      free_conv2D_tensor(&r_1);
      free_conv2D_tensor(&r_2);
    }
  }
  DEBUG_LOG("---------------------input_proj");
  // 1*1 conv 2D (projection)
  conv2D(result, &r, &(config->input_proj), &(weights->input_proj));
  DUMP_TENSOR(result->x, DATA_TYPE, CONV_SIZE(*result), "input_proj");

  // free tensor
  free_conv2D_tensor(&r);
  free_conv2D_tensor(&r_1);
  free_conv2D_tensor(&r_2);
  free_conv2D_tensor(&downsample);
}

/**
 * Runs the forward pass of the Transformer encoder.
 *
 * @param c The TransformerConfig structure.
 * @param w The EncoderWeights structure.
 * @param r The EncoderRunState structure.
 */
void forward_encoder(TransformerConfig* c, EncoderWeights* ew,
                     EncoderRunState* r) {
  int dim = c->dim;
  int hidden_dim = c->hidden_dim;
  int head_size = dim / c->n_heads;
  int seq_len = c->encoder_seq_len;
  int token_size = seq_len * dim;

  EncoderLayerWeights* w = ew->layer;

  DATA_TYPE head_size_sqrt = sqrtf(head_size);

  for (int l = 0; l < c->n_encoder_layers; l++) {
    DEBUG_LOG("---------------------layer = %d", l);
    // pos embedding
    add(r->xb, r->x, ew->pos_embedding, token_size);
    // q, k ,v
    gemm(r->q, r->xb, w[l].wq, w[l].bq, seq_len, dim, dim);
    gemm(r->k, r->xb, w[l].wk, w[l].bk, seq_len, dim, dim);
    gemm(r->v, r->x, w[l].wv, w[l].bv, seq_len, dim, dim);

    // multi-head attention
    multihead_attention(r->xb, r->q, r->k, r->v,
        r->att, ew->att_mask, c->n_heads, dim, seq_len, seq_len);

    // output gemm
    gemm(r->xb2, r->xb, w[l].wo, w[l].bo, seq_len, dim, dim);
    DUMP_TENSOR(r->xb2, DATA_TYPE, token_size,
      "%s.layers.%d.self_attn", ENCODER_NAME, l);

    // residual connection back to x
    add(r->xb, r->x, r->xb2, token_size);
    layernorm(r->x, r->xb, w[l].wln1, w[l].bln1, seq_len, dim);
    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
      "%s.layers.%d.norm1", ENCODER_NAME, l);
    // ffn
    gemm(r->hb, r->x, w[l].w1, w[l].b1, seq_len, dim, hidden_dim);
    DUMP_TENSOR(r->hb, DATA_TYPE, seq_len * hidden_dim,
        "%s.layers.%d.linear1", ENCODER_NAME, l);
    relu(r->hb2, r->hb, seq_len * hidden_dim);
    gemm(r->xb, r->hb2, w[l].w2, w[l].b2, seq_len, hidden_dim, dim);
    DUMP_TENSOR(r->xb, DATA_TYPE, token_size,
      "%s.layers.%d.linear2", ENCODER_NAME, l);
    // residual connection
    add(r->x, r->x, r->xb, token_size);
    layernorm(r->x, r->x, w[l].wln2, w[l].bln2, seq_len, dim);
    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
      "%s.layers.%d.norm2", ENCODER_NAME, l);

    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
        "%s.layers.%d", ENCODER_NAME, l);
  }
}

/**
 * Runs the forward pass of the Transformer decoder.
 *
 * @param c The TransformerConfig structure.
 * @param w The DecoderWeights structure.
 * @param r The DecoderRunState structure.
 */
void forward_decoder(TransformerConfig* c, DecoderWeights* dw,
                     DecoderRunState* r) {
  int dim = c->dim;
  int hidden_dim = c->hidden_dim;
  int head_size = dim / c->n_heads;
  int seq_len = c->decoder_seq_len;
  int encoder_seq_len = c->encoder_seq_len;
  int token_size = seq_len * dim;
  int encoder_token_size = encoder_seq_len * dim;
  MASK_TYPE* att_mask = dw->att_mask;

  DecoderLayerWeights* w = dw->layer;

  DATA_TYPE head_size_sqrt = sqrtf(head_size);

  // decoder feature pos embedding
  add(r->f_embed, r->f, dw->pos_embedding, encoder_token_size);

  for (int l = 0; l < c->n_decoder_layers; l++) {
    DEBUG_LOG("---------------------layer = %d", l);
    // query pos embedding
    add(r->xb, r->x, dw->query_pos_embedding, token_size);
    // q, k ,v
    gemm(r->q, r->xb, w[l].wq, w[l].bq, seq_len, dim, dim);
    gemm(r->k, r->xb, w[l].wk, w[l].bk, seq_len, dim, dim);
    gemm(r->v, r->x, w[l].wv, w[l].bv, seq_len, dim, dim);

    // multi-head attention 1
    multihead_attention(r->xb, r->q, r->k, r->v,
      r->att, NULL, c->n_heads, dim, seq_len, seq_len);

    // output gemm
    gemm(r->xb2, r->xb, w[l].wo, w[l].bo, seq_len, dim, dim);
    DUMP_TENSOR(r->xb2, DATA_TYPE, token_size,
      "%s.layers.%d.self_attn", DECODER_NAME, l);

    // residual connection back to x
    add(r->xb, r->x, r->xb2, token_size);
    layernorm(r->x, r->xb, w[l].wln1, w[l].bln1, seq_len, dim);
    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
      "%s.layers.%d.norm1", DECODER_NAME, l);

    // query pos embedding
    add(r->xb, r->x, dw->query_pos_embedding, token_size);

    // q, k ,v
    gemm(r->q, r->xb, w[l].wq2, w[l].bq2, seq_len, dim, dim);
    gemm(r->k, r->f_embed, w[l].wk2, w[l].bk2, encoder_seq_len, dim, dim);
    gemm(r->v, r->f, w[l].wv2, w[l].bv2, encoder_seq_len, dim, dim);

    // multi-head attention 2
    multihead_attention(r->xb, r->q, r->k, r->v,
      r->att, NULL, c->n_heads, dim, seq_len, encoder_seq_len);

    // output gemm
    gemm(r->xb2, r->xb, w[l].wo2, w[l].bo2, seq_len, dim, dim);
    DUMP_TENSOR(r->xb2, DATA_TYPE, token_size,
      "%s.layers.%d.multihead_attn", DECODER_NAME, l);

    // residual connection back to x
    add(r->xb, r->x, r->xb2, token_size);
    layernorm(r->x, r->xb, w[l].wln2, w[l].bln2, seq_len, dim);
    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
      "%s.layers.%d.norm2", DECODER_NAME, l);

    // ffn
    gemm(r->hb, r->x, w[l].w1, w[l].b1, seq_len, dim, hidden_dim);
    DUMP_TENSOR(r->hb, DATA_TYPE, seq_len * hidden_dim,
      "%s.layers.%d.linear1", DECODER_NAME, l);

    relu(r->hb2, r->hb, seq_len * hidden_dim);
    gemm(r->xb, r->hb2, w[l].w2, w[l].b2, seq_len, hidden_dim, dim);
    DUMP_TENSOR(r->xb, DATA_TYPE, token_size,
      "%s.layers.%d.linear2", DECODER_NAME, l);

    // residual connection
    add(r->x, r->x, r->xb, token_size);
    layernorm(r->x, r->x, w[l].wln3, w[l].bln3, seq_len, dim);
    DUMP_TENSOR(r->x, DATA_TYPE, token_size,
      "%s.layers.%d.norm3", DECODER_NAME, l);
  }
  // norm
  layernorm(r->x, r->x, dw->wln, dw->bln, seq_len, dim);
  DUMP_TENSOR(r->x, DATA_TYPE, token_size,
    "%s.norm", DECODER_NAME);
}

/**
 * Runs the forward pass of the output embedding layer.
 *
 * @param c The DETRConfig structure.
 * @param w The OutputEmbedWeights structure.
 * @param r The OutputRunState structure.
 */
void forward_output(DETRConfig* c, OutputEmbedWeights* w, OutputRunState* r) {
  int num_classes = c->num_classes;
  int num_boxes = c->num_boxes;
  int dim = c->transformer.dim;

  // classes
  gemm(r->classes, r->x, w->class_w, w->class_b, num_boxes, dim, num_classes);
  DUMP_TENSOR(r->classes, DATA_TYPE, num_boxes * num_classes,
    "class_embed");

  // bbox
  gemm(r->xb, r->x, w->bbox_w1, w->bbox_b1, num_boxes, dim, dim);
  DUMP_TENSOR(r->xb, DATA_TYPE, num_boxes * dim,
    "bbox_embed.layers.0");

  relu(r->x, r->xb, num_boxes * dim);
  gemm(r->xb, r->x, w->bbox_w2, w->bbox_b2, num_boxes, dim, dim);
  DUMP_TENSOR(r->xb, DATA_TYPE, num_boxes * dim,
    "bbox_embed.layers.1");

  relu(r->x, r->xb, num_boxes * dim);
  gemm(r->bbox, r->x, w->bbox_w3, w->bbox_b3, num_boxes, dim, BBOX_COORDS);
  DUMP_TENSOR(r->bbox, DATA_TYPE, num_boxes * BBOX_COORDS,
    "bbox_embed.layers.2");

  sigmoid(r->bbox, r->bbox, num_boxes * BBOX_COORDS);
}

/*
 * ============================
 * Operations
 * ============================
 */

/**
 * Performs a 2D convolution operation.
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
  // Perform convolution (channels, height, width)
  for (int oh = 0; oh < output_height; ++oh) {
    for (int ow = 0; ow < output_width; ++ow) {
      for (int oc = 0; oc < config->out_channels; ++oc) {
        DATA_TYPE sum = 0.0f;
        // apply bias if present
        if (weights->bias) {
          sum = weights->bias[oc];
        }
        for (int r = 0; r < config->kernel_size; ++r) {
          for (int c = 0; c < config->kernel_size; ++c) {
            int ih = oh * config->stride + r - config->padding;
            int iw = ow * config->stride + c - config->padding;

            if (ih >= 0 && ih < input->height && iw >= 0 && iw < input->width) {
              for (int ic = 0; ic < config->in_channels; ++ic) {
                uint32_t x_idx =
                    CONVTENSOR_INDEX(input->height, input->width, ic, ih, iw);
                uint32_t w_idx =
                    CONVWEIGHT_INDEX(config->in_channels, config->kernel_size,
                                     config->kernel_size, oc, ic, r, c);
                // 1 MACs
                sum += (input->x[x_idx] * weights->weight[w_idx]);
                // statistics
                STATISTICS_INC_MAC(stat, 1);
                STATISTICS_INC_MEMORY_READ(stat, 2 * sizeof(DATA_TYPE));
                STATISTICS_INC_MEMORY_WRITE(stat, 1 * sizeof(DATA_TYPE));
              }
            }
          }
        }
        // store output
        output->x[CONVTENSOR_INDEX(output_height, output_width, oc, oh, ow)] =
            sum;
      }
    }
  }
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Performs a 2D max pooling operation.
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
  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int ch = 0; ch < input->channels; ++ch) {

        DATA_TYPE max_value = -__FLT_MAX__;

        for (int r = 0; r < config->kernel_size; ++r) {
          for (int c = 0; c < config->kernel_size; ++c) {
            int in_h = h * config->stride + r - config->padding;
            int in_w = w * config->stride + c - config->padding;

            if (in_h >= 0 && in_h < input->height && in_w >= 0 &&
                in_w < input->width) {

              DATA_TYPE value = input->x[CONVTENSOR_INDEX(
                  input->height, input->width, ch, in_h, in_w)];


              STATISTICS_INC_MEMORY_READ(stat, 1 * sizeof(DATA_TYPE));

              if (value > max_value) {
                max_value = value;
              }
              // statistics - compare
              STATISTICS_INC_NON_LINEAR_OP(stat, 1);
            }
          }
        }

        output->x[CONVTENSOR_INDEX(height, width, ch, h, w)] = max_value;
        // statistics
        STATISTICS_INC_MEMORY_WRITE(stat, 1 * sizeof(DATA_TYPE));
      }
    }
  }
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Applies layer normalization.
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
  for (int i = 0; i < n; i++) {
    DATA_TYPE mean = 0;
    DATA_TYPE var = 0;

    // Compute mean
    for (int j = 0; j < dim; j++) {
      mean += x[j * n + i];
    }
    STATISTICS_INC_MEMORY_READ(stat, 2 * dim * sizeof(DATA_TYPE)); // x, mean
    STATISTICS_INC_MAC(stat, dim);
    STATISTICS_INC_MEMORY_WRITE(stat, dim * sizeof(DATA_TYPE)); // mean

    mean /= dim;
    STATISTICS_INC_DIV(stat, 1);

    // Compute variance
    for (int j = 0; j < dim; j++) {
      DATA_TYPE diff = x[j * n + i] - mean;
      var += diff * diff;
    }
    STATISTICS_INC_MEMORY_READ(stat, 2 * dim * sizeof(DATA_TYPE)); // x, var
    STATISTICS_INC_MAC(stat, dim);
    STATISTICS_INC_MEMORY_WRITE(stat, dim * sizeof(DATA_TYPE)); // var

    var /= dim;
    STATISTICS_INC_DIV(stat, 1);

    DATA_TYPE inv_std = 1.0f / sqrtf(var + eps);

    // Normalize + affine transform
    for (int j = 0; j < dim; j++) {
      DATA_TYPE norm = (x[j * n + i] - mean) * inv_std;
      out[j * n + i] = norm * w[j] + b[j];
    }
    STATISTICS_INC_MEMORY_READ(stat, 3 * dim * sizeof(DATA_TYPE)); // x, w, b
    STATISTICS_INC_MAC(stat, dim * 2);
    STATISTICS_INC_MEMORY_WRITE(stat, dim * sizeof(DATA_TYPE)); // out
  }
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Applies batch normalization.
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
  for (int c = 0; c < C; c++) {
    DATA_TYPE gamma = bn->weight[c];
    DATA_TYPE beta = bn->bias[c];
    DATA_TYPE running_var = bn->running_var[c];
    DATA_TYPE running_mean = bn->running_mean[c];

    DATA_TYPE scale = gamma / sqrtf(running_var + eps);
    DATA_TYPE bias = beta - running_mean * scale;

    STATISTICS_INC_MEMORY_READ(stat, 4 * sizeof(DATA_TYPE)); // gamma, beta, mean, var
    STATISTICS_INC_NON_LINEAR_OP(stat, 1); // sqrtf
    STATISTICS_INC_ADD(stat, 1); // scale
    STATISTICS_INC_DIV(stat, 1); // scale
    STATISTICS_INC_MAC(stat, 1); // bias

    for (int h = 0; h < H; h++) {
      for (int w = 0; w < W; w++) {
        int index = CONVTENSOR_INDEX(H, W, c, h, w);
        output->x[index] = input->x[index] * scale + bias;
      }
    }
    STATISTICS_INC_MEMORY_READ(stat, H*W*sizeof(DATA_TYPE)); // input->x
    STATISTICS_INC_MEMORY_WRITE(stat, H*W*sizeof(DATA_TYPE)); // output->x
    STATISTICS_INC_MAC(stat, H*W);
  }
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Performs a general matrix multiplication (GEMM) operation.
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
  for(int o = 0; o < od; o++) {
    for(int nidx = 0; nidx < n; nidx++) {
      out[o * n + nidx] = b[o];
      for(int i = 0; i < id; i++) {
        out[o * n + nidx] += w[o * id + i] * x[i * n + nidx];
        STATISTICS_INC_MEMORY_READ(stat, 2 * sizeof(DATA_TYPE)); // w, x
        STATISTICS_INC_MEMORY_WRITE(stat, sizeof(DATA_TYPE)); // out
        STATISTICS_INC_MAC(stat, 1);
      }
    }
  }
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Adds two arrays element-wise.
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
  int i;
  for (i = 0; i < size; i++) {
    out[i] = x[i] + y[i];
  }
  STATISTICS_INC_MEMORY_READ(stat, 2 * size * sizeof(DATA_TYPE)); // x, y
  STATISTICS_INC_MEMORY_WRITE(stat, size * sizeof(DATA_TYPE)); // out
  STATISTICS_INC_ADD(stat, 2 * size);
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Performs a multi-head attention operation.
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

  int h;
  for (h = 0; h < n_heads; h++) {
    // get q,k,v of this head
    DATA_TYPE* Q = qx + h * head_size * q_len;
    DATA_TYPE* K = kx + h * head_size * kv_len;
    DATA_TYPE* V = vx + h * head_size * kv_len;
    DATA_TYPE* ATT = att + h * q_len * kv_len;
    DATA_TYPE* OUT = out + h * head_size * q_len;

    // q,k -> attention score
    for (int q = 0; q < q_len; q++) {
      for (int k = 0; k < kv_len; k++) {
        // skip if mask is 0(false);
        if (use_att_mask && (!att_mask[q] || !att_mask[k])) {
          ATT[q * kv_len + k] =
              -__FLT_MAX__;  // Use large negative value for masked positions
          STATISTICS_INC_MEMORY_WRITE(stat_self_attn, sizeof(DATA_TYPE));
          continue;
        }

        DATA_TYPE score = 0;
        for (int d = 0; d < head_size; d++) {
          score += Q[d * q_len + q] * K[d * kv_len + k];
        }
        STATISTICS_INC_MAC(stat_self_attn, head_size);
        STATISTICS_INC_MEMORY_READ(stat_self_attn, 2 * head_size * sizeof(DATA_TYPE));

        ATT[q * kv_len + k] = score / head_size_sqrt;
        STATISTICS_INC_DIV(stat_self_attn, 1); // div
        STATISTICS_INC_MEMORY_WRITE(stat_self_attn, sizeof(DATA_TYPE));
      }
      softmax(ATT + q * kv_len, ATT + q * kv_len, kv_len);  // softmax
      STATISTICS_INC_MAC(stat_softmax, 2 * kv_len); // /, -
      STATISTICS_INC_NON_LINEAR_OP(stat_softmax, kv_len); // exp
      STATISTICS_INC_MEMORY_READ(stat_softmax, 3 * kv_len * sizeof(DATA_TYPE));
      STATISTICS_INC_MEMORY_WRITE(stat_softmax, 2 * kv_len * sizeof(DATA_TYPE));
    }

    // output of attention = att @ V
    for (int i = 0; i < q_len; i++) {
      for (int d = 0; d < head_size; d++) {
        DATA_TYPE sum = 0;
        for (int j = 0; j < kv_len; j++) {
          sum += ATT[i * kv_len + j] * V[d * kv_len + j];
        }
        STATISTICS_INC_MAC(stat_self_attn, kv_len);
        STATISTICS_INC_MEMORY_READ(stat_self_attn, 2 * kv_len * sizeof(DATA_TYPE));

        OUT[d * q_len + i] = sum;
        STATISTICS_INC_MEMORY_WRITE(stat_self_attn, sizeof(DATA_TYPE));
      }
    }
  }
  STATISTICS_APPEND_CSV(stat_softmax);
  STATISTICS_APPEND_CSV(stat_self_attn);
}

/*
 * ============================
 * Activation Functions
 * ============================
 */

/**
 * Applies the ReLU activation function.
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
  STATISTICS_INC_MEMORY_READ(stat, size * sizeof(DATA_TYPE)); // x
  STATISTICS_INC_MEMORY_WRITE(stat, size * sizeof(DATA_TYPE)); // out
  STATISTICS_INC_NON_LINEAR_OP(stat, size); // compare
  STATISTICS_APPEND_CSV(stat);
}

/**
 * Applies the softmax activation function.
 *
 * @param out The output array.
 * @param x The input array.
 * @param size The size of the array.
 */
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
}

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
  STATISTICS_INC_MEMORY_READ(stat, size * sizeof(DATA_TYPE)); // x
  STATISTICS_INC_MEMORY_WRITE(stat, size * sizeof(DATA_TYPE)); // out
  STATISTICS_INC_NON_LINEAR_OP(stat, size);
  STATISTICS_INC_ADD(stat, size);
  STATISTICS_INC_DIV(stat, size);
  STATISTICS_APPEND_CSV(stat);
}
