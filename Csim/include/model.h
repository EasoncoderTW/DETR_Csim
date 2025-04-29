#ifndef MODEL_H
#define MODEL_H

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

#define DATA_TYPE float
#define MASK_TYPE uint8_t
#define CONVTENSOR_INITIALIZER \
  { NULL, 0, 0, 0 }
#define OUTPUTTENSOR_INITIALIZER \
  { NULL, NULL, 0, 0 }

//**************************************
// DETR model configuration
//**************************************
typedef struct {
  int dim;               // transformer dimension
  int hidden_dim;        // for ffn layers
  int n_encoder_layers;  // number of encoder layers
  int n_decoder_layers;  // number of decoder layers
  int n_heads;           // number of query heads
  int encoder_seq_len;   // max sequence length
  int decoder_seq_len;   // max sequence length
} TransformerConfig;

typedef struct {
  int in_channels;   // input channels (C)
  int out_channels;  // output channels (M)
  int kernel_size;   // kernel size (R/S)
  int stride;        // stride (U)
  int padding;       // padding (P)
} ConvConfig;

typedef struct {
  int kernel_size;  // kernel size (R/S)
  int stride;       // stride (U)
  int padding;      // padding (P)
} MaxPoolConfig;

typedef struct {
  int num_bottleneck;
  ConvConfig downsample;
  ConvConfig* conv;
} ResidualConfig;

typedef struct {
  int num_resblock;
  ConvConfig conv1;          // first conv layer
  MaxPoolConfig maxpool;     // only one maxpooling layer
  ResidualConfig* resblock;  // residual blocks
  ConvConfig input_proj;     // final conv layer
} ResNet50Config;

typedef struct {
  TransformerConfig transformer;  // transformer encoder config
  ResNet50Config resnet50;        // ResNet50 config
  int num_classes;                // number of classes (C)
  int num_boxes;                  // number of boxes (B)
  int image_height;               // image height (H)
  int image_width;                // image width (W)
  int image_channels;             // image channels (C)
} DETRConfig;

//**************************************
// Weights
//**************************************

typedef struct {
  DATA_TYPE*
      weight;       // (out_channels, in_channels, kernel_height, kernel_width)
  DATA_TYPE* bias;  // (out_channels) // reserved for future use
} ConvWeights;

typedef struct {
  DATA_TYPE*
      weight;       // (out_channels, in_channels, kernel_height, kernel_width)
  DATA_TYPE* bias;  // (out_channels)
  DATA_TYPE* running_mean;  // (out_channels)
  DATA_TYPE* running_var;   // (out_channels)
} BatchNormWeights;

typedef struct {
  ConvWeights downsample;
  BatchNormWeights bn_downsample;
  ConvWeights* conv;
  BatchNormWeights* bn;
} ResidualWeights;

typedef struct {
  ConvWeights conv1;
  BatchNormWeights bn1;
  ResidualWeights* resblock;  // residual blocks
  ConvWeights input_proj;
} ResNet50Weights;

typedef struct {
  // weights for matmuls. note dim == n_heads * head_size
  DATA_TYPE* wq;  // (layer, dim, n_heads * head_size)
  DATA_TYPE* bq;  // (layer, dim)
  DATA_TYPE* wk;  // (layer, dim,  n_heads * head_size)
  DATA_TYPE* bk;  // (layer, dim)
  DATA_TYPE* wv;  // (layer, dim,  n_heads * head_size)
  DATA_TYPE* bv;  // (layer, dim)
  DATA_TYPE* wo;  // (layer, dim,  dim)
  DATA_TYPE* bo;  // (layer, dim)
  // weights for layernorm
  DATA_TYPE* wln1;  // (layer, dim)
  DATA_TYPE* bln1;  // (layer, dim)
  DATA_TYPE* wln2;  // (layer, dim)
  DATA_TYPE* bln2;  // (layer, dim)
  // weights for ffn
  DATA_TYPE* w1;  // (layer, dim, hidden_dim)
  DATA_TYPE* b1;  // (layer, hidden_dim)
  DATA_TYPE* w2;  // (layer, hidden_dim, dim)
  DATA_TYPE* b2;  // (layer, dim)
} EncoderLayerWeights;

typedef struct{
  // position embedding
  DATA_TYPE* pos_embedding;  // (dim, seq_len)
  // attention mask
  MASK_TYPE* att_mask;  // (seq_len)
  EncoderLayerWeights* layer;  // layers
}EncoderWeights;

typedef struct {
  // weights for matmuls. note dim == n_heads * head_size
  DATA_TYPE* wq;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bq;  // (layer, n_heads , head_size)
  DATA_TYPE* wk;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bk;  // (layer, n_heads , head_size)
  DATA_TYPE* wv;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bv;  // (layer, n_heads , head_size)
  DATA_TYPE* wo;  // (layer, dim,  dim)
  DATA_TYPE* bo;  // (layer, dim)
  // weights for matmuls. note dim == n_heads * head_size
  DATA_TYPE* wq2;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bq2;  // (layer, n_heads , head_size)
  DATA_TYPE* wk2;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bk2;  // (layer, n_heads , head_size)
  DATA_TYPE* wv2;  // (layer, n_heads , head_size, dim)
  DATA_TYPE* bv2;  // (layer, n_heads , head_size)
  DATA_TYPE* wo2;  // (layer, dim, dim)
  DATA_TYPE* bo2;  // (layer, dim)
  // weights for layernorm
  DATA_TYPE* wln1;  // (layer, dim)
  DATA_TYPE* bln1;  // (layer, dim)
  DATA_TYPE* wln2;  // (layer, dim)
  DATA_TYPE* bln2;  // (layer, dim)
  DATA_TYPE* wln3;  // (layer, dim)
  DATA_TYPE* bln3;  // (layer, dim)
  // weights for ffn
  DATA_TYPE* w1;  // (layer, hidden_dim, dim)
  DATA_TYPE* b1;  // (layer, hidden_dim)
  DATA_TYPE* w2;  // (layer, dim, hidden_dim)
  DATA_TYPE* b2;  // (layer, dim)
} DecoderLayerWeights;

typedef struct {
  // position embedding
  DATA_TYPE* pos_embedding;  // (dim, seq_len)
  // query position embedding
  DATA_TYPE* query_pos_embedding;  // (dim, seq_len)
  // attention mask
  MASK_TYPE* att_mask;  // (seq_len)
  DATA_TYPE* wln;
  DATA_TYPE* bln;
  DecoderLayerWeights* layer;  // layers
} DecoderWeights;

typedef struct {
  DATA_TYPE* class_w;  // (num_classes, dim)
  DATA_TYPE* class_b;  // (num_classes)
  DATA_TYPE* bbox_w1;  // (dim, dim)
  DATA_TYPE* bbox_b1;  // (dim)
  DATA_TYPE* bbox_w2;  // (dim, dim)
  DATA_TYPE* bbox_b2;  // (dim)
  DATA_TYPE* bbox_w3;  // (4, dim)
  DATA_TYPE* bbox_b3;  // (4)
} OutputEmbedWeights;

typedef struct {
  DATA_TYPE* pos_embedding;  // position embedding (dim, seq_len)
  MASK_TYPE* att_mask;       // attention mask (seq_len)
} PreprocessWeights;

typedef struct {
  PreprocessWeights preprocess;    // preprocessing weights
  ResNet50Weights resnet50;        // backbone
  EncoderWeights encoder;         // transformer encoder
  DecoderWeights decoder;         // transformer decoder
  OutputEmbedWeights outputembed;  // Output Embedding
} DETRWeights;

//**************************************
// DETR model tensor
//**************************************

typedef struct {
  DATA_TYPE* x;  // (channels, height, width)
  int height;    // height of the activation
  int width;     // width of the activation
  int channels;  // input channels (C)
} ConvolutionTensor;

typedef struct {
  DATA_TYPE* classes;  // (num_classes, seq_len)
  DATA_TYPE* bbox;     // (4, seq_len)
  int num_boxes;       // sequence length
  int num_classes;     // number of classes (C)
} OutputTensor;

//**************************************
// DETR model runstate
//**************************************

typedef struct {
  DATA_TYPE* x;    // activation at current time stamp (dim, seq_len)
  DATA_TYPE* xb;   // same, but inside a residual branch (dim, seq_len)
  DATA_TYPE* xb2;  // an additional buffer just for convenience (dim, seq_len)
  DATA_TYPE* hb;   // buffer in the ffn (hidden_dim, seq_len)
  DATA_TYPE* hb2;  // buffer in the ffn (hidden_dim, seq_len)
  DATA_TYPE* q;    // query (dim, seq_len)
  DATA_TYPE* k;    // key (dim, seq_len)
  DATA_TYPE* v;    // value (dim, seq_len)
  DATA_TYPE* att;  // buffer for scores/attention (n_heads, seq_len, seq_len)
} EncoderRunState;

typedef struct {
  DATA_TYPE* x;    // activation at current time stamp (dim, seq_len)
  DATA_TYPE* xb;   // same, but inside a residual branch (dim, seq_len)
  DATA_TYPE* xb2;  // an additional buffer just for convenience (dim, seq_len)
  DATA_TYPE* hb;   // buffer in the ffn (hidden_dim, seq_len)
  DATA_TYPE* hb2;  // buffer in the ffn (hidden_dim, seq_len)
  DATA_TYPE* q;    // query (dim, seq_len)
  DATA_TYPE* k;    // key (dim, seq_len)
  DATA_TYPE* v;    // value (dim, seq_len)
  DATA_TYPE* att;  // buffer for scores/attention values (n_heads, seq_len)
  DATA_TYPE* f;    // features from encoder (dim, encoder_seq_len)
  DATA_TYPE* f_embed;  // features from encoder (dim, encoder_seq_len)
} DecoderRunState;

typedef struct {
  DATA_TYPE* x;        // (dim, seq_len)
  DATA_TYPE* xb;       // (dim, seq_len)
  DATA_TYPE* classes;  // (num_classes, seq_len)
  DATA_TYPE* bbox;     // (4, seq_len)
} OutputRunState;

//**************************************
// DETR model weight file header
//**************************************

typedef struct {
  uint32_t id;
  uint32_t data_type;
  uint32_t data_offset;
  uint32_t data_size;
  uint32_t name_offset;
  uint32_t name_size;
} TensorInfo;

typedef struct {
  uint32_t version;
  uint32_t num_tensor;
  uint32_t pack_method;
  uint32_t info_offset;
  uint32_t name_offset;
  uint32_t data_offset;
} TensorFile;

//**************************************
// DETR model structure
//**************************************

typedef struct {
  DETRConfig config;       // model configuration
  DETRWeights weights;     // model weights
  FILE* weight_fp;         // file pointer
  void* weight_mmap;       // memory mapped file pointer
  size_t mmap_size;        // size of the memory mapped file
  TensorFile file_header;  // file header for weights search
} DETR;

//**************************************
// DEBUG API
//**************************************
void dump_tensor(const char* name, const DATA_TYPE* tensor, int size);

//**************************************
// JSON and Config Parsing
//**************************************
int extract_int_value(const char* key, const char* json);
char* load_json_file(const char* filename);
void parse_conv_config(const char* section, ConvConfig* config);
void parse_maxpool_config(const char* section, MaxPoolConfig* config);

int load_config(const char* filename, DETR* detr);
void free_config(DETR* detr);

//**************************************
// Tensor and Weights Management
//**************************************
void* find_tensor(const char* name, DETR* detr);
int generate_mask_and_pos_embedding(DETR* detr);

int load_weights(const char* filename, DETR* detr);
void free_weights(DETR* detr);

void load_input_tensor(ConvolutionTensor* r, const char* filename);
void save_output_tensor(OutputTensor* r, const char* boxes_filename,
                      const char* logits_filename);

//**************************************
// DETR Model API
//**************************************
void print_conv_config(int level, const char* name, const ConvConfig* config);
void print_config(DETRConfig* config);

int init_detr(DETR* detr, const char* config_file, const char* weights_file);
void free_detr(DETR* detr);

void print_result(OutputRunState* result);

//**************************************
// Tensor Memory Management
//**************************************
void malloc_conv2D_tensor(ConvolutionTensor* r);
void free_conv2D_tensor(ConvolutionTensor* r);

void malloc_output_tensor(OutputTensor* r);
void free_output_tensor(OutputTensor* r);

//**************************************
// Runstate Memory Management
//**************************************
void malloc_encoder_run_state(const TransformerConfig* config,
                              EncoderRunState* r);
void free_encoder_run_state(EncoderRunState* r);

void malloc_decoder_run_state(const TransformerConfig* config,
                              DecoderRunState* r);
void free_decoder_run_state(DecoderRunState* r);

void malloc_output_run_state(const DETRConfig* config, OutputRunState* r);
void free_output_run_state(OutputRunState* r);

//**************************************
// Model Inference
//**************************************
void forward(DETR* detr, ConvolutionTensor* image, OutputTensor* result);
void forward_resnet50(ResNet50Config* config, ResNet50Weights* weights,
                      ConvolutionTensor* image, ConvolutionTensor* result);
void forward_encoder(TransformerConfig* c, EncoderWeights* w,
                     EncoderRunState* r);
void forward_decoder(TransformerConfig* c, DecoderWeights* w,
                     DecoderRunState* r);
void forward_output(DETRConfig* c, OutputEmbedWeights* w, OutputRunState* r);

//**************************************
// Operations
//**************************************
void conv2D(ConvolutionTensor* output, const ConvolutionTensor* input,
            const ConvConfig* config, const ConvWeights* weights);
void maxpooling2D(ConvolutionTensor* output, const ConvolutionTensor* input,
                  const MaxPoolConfig* config);
void layernorm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n,
               int dim);
void batchnorm2D(ConvolutionTensor* output, const ConvolutionTensor* input,
                 BatchNormWeights* bn);
void gemm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n,
          int id, int od);
void add(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* y, int size);

//**************************************
// Activation Functions
//**************************************
void relu(DATA_TYPE* out, DATA_TYPE* x, int size);
void softmax(DATA_TYPE* out, DATA_TYPE* x, int size);
void sigmoid(DATA_TYPE* out, DATA_TYPE* x, int size);

#endif  // MODEL_H
