#ifndef MODEL_H
#define MODEL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#define DATA_TYPE float
#define DATA_TYPE_SIZE sizeof(DATA_TYPE)
#define SQRT sqrtf

#define CONV_SIZE(runstate) ((runstate)->height * (runstate)->width * (runstate)->channels)

#define CONV_OP_PER_BUTTLENECK 3
#define CONV_ID(resblock_id, conv_id) ((conv_id) + CONV_OP_PER_BUTTLENECK*(resblock_id))

#define RESBLOCK 4
#define ENCODER_LAYER 6
#define DECODER_LAYER 6

#define BBOX_COORDS 4

/*
* Config
*/
typedef struct{
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_encoder_layers; // number of encoder layers
    int n_decoder_layers; // number of decoder layers
    int n_heads; // number of query heads
    int encoder_seq_len; // max sequence length
    int decoder_seq_len; // max sequence length
} TransformerConfig; // 28bytes

typedef struct{
    int in_channels; // input channels (C)
    int out_channels; // output channels (M)
    int kernel_size; // kernel size (R/S)
    int stride; // stride (U)
    int padding; // padding (P)
} ConvConfig; // 20bytes

typedef struct{
    int num_bottleneck;
    ConvConfig shortcut;
    ConvConfig* conv;
} ResidualConfig; // 20bytes

typedef struct{
    int num_resblock;
    ConvConfig conv1; // first conv layer
    ResidualConfig *resblock; // residual blocks
    ConvConfig conv2; // final conv layer
} ResNet50Config; // 600bytes

typedef struct{
    TransformerConfig tranformer; // transformer encoder config
    ResNet50Config resnet50; // ResNet50 config
    int num_classes; // number of classes (C)
    int num_boxes; // number of boxes (B)
} DETRConfig; // 636bytes

/*
* Weights
*/

/* Convolution weight */
typedef struct{
    DATA_TYPE* weight; // (out_channels, in_channels, kernel_height, kernel_width)
    DATA_TYPE* bias;   // (out_channels)
} ConvWeights;

/* BatchNorm weight */
typedef struct{
    DATA_TYPE* weight; // (out_channels, in_channels, kernel_height, kernel_width)
    DATA_TYPE* bias;   // (out_channels)
} BatchNormWeights;

typedef struct{
    int num_bottleneck;
    ConvWeights shortcut;
    BatchNormWeights bn_shortcut;
    ConvWeights* conv;
    BatchNormWeights* bn;
} ResidualWeights;

/* ResNet50 */
typedef struct{
    ConvWeights conv1;
    BatchNormWeights* bn1;
    ResidualWeights *resblock; // residual blocks
    ConvWeights conv2;
}ResNet50Weights;


/* transformer encoder weight */
typedef struct{
    // position embedding
    DATA_TYPE* pos_embedding;    // (seq_len, dim)
    // weights for matmuls. note dim == n_heads * head_size
    DATA_TYPE* wq; // (layer, dim, n_heads * head_size)
    DATA_TYPE* bq; // (layer, dim)
    DATA_TYPE* wk; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bk; // (layer, dim)
    DATA_TYPE* wv; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bv; // (layer, dim)
    DATA_TYPE* wo; // (layer, dim,  dim)
    DATA_TYPE* bo; // (layer, dim)
    // weights for layernorm
    DATA_TYPE* wln1;  // (layer, dim)
    DATA_TYPE* bln1;  // (layer, dim)
    DATA_TYPE* wln2;  // (layer, dim)
    DATA_TYPE* bln2;  // (layer, dim)
    // weights for ffn
    DATA_TYPE* w1; // (layer, dim, hidden_dim)
    DATA_TYPE* b1; // (layer, hidden_dim)
    DATA_TYPE* w2; // (layer, hidden_dim, dim)
    DATA_TYPE* b2; // (layer, dim)
} EncoderWeights;

/* transformer decoder weight */
typedef struct{
    // position embedding
    DATA_TYPE* pos_embedding;    // (seq_len, dim)
    // weights for matmuls. note dim == n_heads * head_size
    DATA_TYPE* wq; // (layer, dim, n_heads * head_size)
    DATA_TYPE* bq; // (layer, dim)
    DATA_TYPE* wk; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bk; // (layer, dim)
    DATA_TYPE* wv; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bv; // (layer, dim)
    DATA_TYPE* wo; //  (layer, dim,  dim)
    DATA_TYPE* bo; // (layer, dim)
    // query position embedding
    DATA_TYPE* query_pos_embedding;    // (seq_len, dim)
    // weights for matmuls. note dim == n_heads * head_size
    DATA_TYPE* wq2; // (layer, dim, n_heads * head_size)
    DATA_TYPE* bq2; // (layer, dim)
    DATA_TYPE* wk2; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bk2; // (layer, dim)
    DATA_TYPE* wv2; // (layer, dim,  n_heads * head_size)
    DATA_TYPE* bv2; // (layer, dim)
    DATA_TYPE* wo2; // (layer, dim, dim)
    DATA_TYPE* bo2; // (layer, dim)
    // weights for layernorm
    DATA_TYPE* wln1;  // (layer, dim)
    DATA_TYPE* bln1;  // (layer, dim)
    DATA_TYPE* wln2;  // (layer, dim)
    DATA_TYPE* bln2;  // (layer, dim)
    DATA_TYPE* wln3;  // (layer, dim)
    DATA_TYPE* bln3;  // (layer, dim)
    // weights for ffn
    DATA_TYPE* w1; // (layer, dim, hidden_dim)
    DATA_TYPE* b1; // (layer, hidden_dim)
    DATA_TYPE* w2; // (layer, hidden_dim, dim)
    DATA_TYPE* b2; // (layer, dim)
} DecoderWeights;

/* Output Embedding */
typedef struct{
    DATA_TYPE* class_w; // (num_classes, dim)
    DATA_TYPE* class_b; // (num_classes)
    DATA_TYPE* bbox_w1; // (dim, dim)
    DATA_TYPE* bbox_b1; // (dim)
    DATA_TYPE* bbox_w2; // (dim, dim)
    DATA_TYPE* bbox_b2; // (dim)
    DATA_TYPE* bbox_w3; // (4, dim)
    DATA_TYPE* bbox_b3; // (4)
}OutputEmbedWeights;

/* DETR weights */
typedef struct{
    ResNet50Weights resnet50; // backbone
    EncoderWeights *encoder; // transformer encoder
    DecoderWeights *decoder; // transformer decoder
    OutputEmbedWeights *outputembed; // Output Embedding
    DATA_TYPE* query_pos_embedding; // (seq_len, dim)
} DETRWeights;

/*
* Activations
*/

/* Convolution RunState */
typedef struct{
    DATA_TYPE* x; // (channels, height, width)
    int height; // height of the activation
    int width; // width of the activation
    int channels; // input channels (C)
} ConvolutionRunState;

/* Transformer Encoder RunState */
typedef struct {
    // current wave of activations
    DATA_TYPE *x; // activation at current time stamp (dim, seq_len)
    DATA_TYPE *xb; // same, but inside a residual branch (dim, seq_len)
    DATA_TYPE *xb2; // an additional buffer just for convenience (dim, seq_len)
    DATA_TYPE *hb; // buffer for hidden dimension in the ffn (hidden_dim, seq_len)
    DATA_TYPE *hb2; // buffer for hidden dimension in the ffn (hidden_dim, seq_len)
    DATA_TYPE *q; // query (dim, seq_len)
    DATA_TYPE *k; // key (dim, seq_len)
    DATA_TYPE *v; // value (dim, seq_len)
    DATA_TYPE *att; // buffer for scores/attention values (n_heads, seq_len, seq_len)
    bool *att_mask; // attention mask
} EncoderRunState;

/* Transformer Decoder RunState */
typedef struct {
    // current wave of activations
    DATA_TYPE *x; // activation at current time stamp (dim, seq_len)
    DATA_TYPE *xb; // same, but inside a residual branch (dim, seq_len)
    DATA_TYPE *xb2; // an additional buffer just for convenience (dim, seq_len)
    DATA_TYPE *hb; // buffer for hidden dimension in the ffn (hidden_dim, seq_len)
    DATA_TYPE *hb2; // buffer for hidden dimension in the ffn (hidden_dim, seq_len)
    DATA_TYPE *q; // query (dim, seq_len)
    DATA_TYPE *k; // key (dim, seq_len)
    DATA_TYPE *v; // value (dim, seq_len)
    DATA_TYPE *att; // buffer for scores/attention values (n_heads, seq_len)
    DATA_TYPE *f; // features from encoder (dim, encoder_seq_len)
    DATA_TYPE *f_embed; // features from encoder (dim, encoder_seq_len)
    DATA_TYPE *target; // target to output embed (dim, seq_len)
} DecoderRunState;

/* Transformer Output RunState */
typedef struct {
    // current wave of activations
    DATA_TYPE *x; // (num_boxes, dim)
    DATA_TYPE *xb; // (num_boxes, dim)
    DATA_TYPE *classes; // (num_boxes, num_classes)
    DATA_TYPE *bbox; // (num_boxes, 4)
} OutputRunState;

/*
* DETR
*/

typedef struct{
    DETRConfig config;
    DETRWeights weights;
} DETR;

/*
 * Functions
 */

void print_config(DETR* detr);
void init_detr(DETR* detr, const char* config_file, const char* weights_file);
void free_detr(DETR* detr);

void print_result(OutputRunState* result);

/* malloc and free of ConvolutionRunState */
void malloc_conv2D_run_state(ConvolutionRunState* r);
void free_conv2D_run_state(ConvolutionRunState* r);
/* malloc and free of EncoderRunState */
void malloc_encoder_run_state(const TransformerConfig* config, EncoderRunState* r);
void free_encoder_run_state(EncoderRunState* r);
/* malloc and free of DecoderRunState */
void malloc_decoder_run_state(const TransformerConfig* config, DecoderRunState* r);
void free_decoder_run_state(DecoderRunState* r);
/* malloc and free of OutputRunState */
void malloc_output_run_state(const TransformerConfig* config, OutputRunState* r);
void free_output_run_state(OutputRunState* r);

/*
 * model inference
 */
void forward(DETR* detr, ConvolutionRunState* image, OutputRunState* result);
void forward_resnet50(ResNet50Config* config, ResNet50Weights* weights, ConvolutionRunState* image, ConvolutionRunState* result);
void forward_encoder(TransformerConfig* c, EncoderWeights* w, EncoderRunState* r);
void forward_decoder(TransformerConfig* c, DecoderWeights* w, DecoderRunState* r);
void forward_output(DETRConfig* c, OutputEmbedWeights* w, OutputRunState* r);
/*
 *  Operations
 */
 void conv2D(ConvolutionRunState *output, const ConvolutionRunState *input, const ConvConfig *config, const ConvWeights *weights);
 void maxpooling2D(ConvolutionRunState *output, const ConvolutionRunState *input, int stride);
 void layernorm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n, int dim);
 void batchnorm2D(ConvolutionRunState *output, const ConvolutionRunState *input, BatchNormWeights *bn);
 void gemm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n, int id, int od);
 void add(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* y, int size);

 /*
 *  Activation Function
 */
void relu(DATA_TYPE* out, DATA_TYPE* x, int size);
void softmax(DATA_TYPE* out, DATA_TYPE* x, int size);
void sigmoid(DATA_TYPE* out, DATA_TYPE* x, int size);


#endif // MODEL_H
