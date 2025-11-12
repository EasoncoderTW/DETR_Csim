#ifndef UTILS_H
#define UTILS_H

/*
* ============================
* DEFINES
* ============================
*/

#define DATA_TYPE float
#define MASK_TYPE uint8_t

#define SOFTMAX_NORMAL 0
#define SOFTMAX_SOLE 1

#ifndef SOFTMAX_METHOD
#define SOFTMAX_METHOD SOFTMAX_NORMAL
#endif

#define LAYERNORM_NORMAL 0
#define LAYERNORM_SOLE 1

#ifndef LAYERNORM_METHOD
#define LAYERNORM_METHOD LAYERNORM_NORMAL
#endif

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

//**************************************
// DEBUG API
//**************************************
void dump_tensor(const char* name, const DATA_TYPE* tensor, int size);

//**************************************
// fp16 clip
//**************************************
#ifdef FP16
float fp16_clip(float x_val);
#endif

#endif // UTILS_H