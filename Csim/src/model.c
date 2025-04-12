#include "model.h"
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

/*
 * Utils
 */

// Helper to extract integer values from `"key": value` format
int extract_int_value(const char* key, const char* json) {
    char* pos = strstr(json, key);
    if (!pos) return 0;
    pos = strchr(pos, ':');
    if (!pos) return 0;
    return atoi(pos + 1);
}

// Extract a ConvConfig block from the JSON
void parse_conv_config(const char* section, ConvConfig* config) {
    config->in_channels = extract_int_value("\"in_channels\"", section);
    config->out_channels = extract_int_value("\"out_channels\"", section);
    config->kernel_size = extract_int_value("\"kernel_size\"", section);
    config->stride = extract_int_value("\"stride\"", section);
    config->padding = extract_int_value("\"padding\"", section);
}

// Extract a MaxPoolConfig block from the JSON
void parse_maxpool_config(const char* section, MaxPoolConfig* config) {
    config->kernel_size = extract_int_value("\"kernel_size\"", section);
    config->stride = extract_int_value("\"stride\"", section);
    config->padding = extract_int_value("\"padding\"", section);
}


// Load entire JSON file into memory
char* load_json_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    rewind(f);

    char* buffer = malloc(len + 1);
    fread(buffer, 1, len, f);
    buffer[len] = '\0';
    fclose(f);
    return buffer;
}

int load_config(const char* filename, DETR* detr) {

    DETRConfig *config = &(detr->config);

    printf("load config file: %s\n", filename);
    char* json = load_json_file(filename);
    if (!json) return -1;

    // Top-level config values
    config->num_classes = extract_int_value("\"num_classes\"", json);
    config->num_boxes = extract_int_value("\"num_boxes\"", json);

    // Transformer config
    char* t = strstr(json, "\"transformer\"");
    config->transformer.dim = extract_int_value("\"dim\"", t);
    config->transformer.hidden_dim = extract_int_value("\"hidden_dim\"", t);
    config->transformer.n_encoder_layers = extract_int_value("\"n_encoder_layers\"", t);
    config->transformer.n_decoder_layers = extract_int_value("\"n_decoder_layers\"", t);
    config->transformer.n_heads = extract_int_value("\"n_heads\"", t);
    config->transformer.encoder_seq_len = extract_int_value("\"encoder_seq_len\"", t);
    config->transformer.decoder_seq_len = extract_int_value("\"decoder_seq_len\"", t);

    // ResNet50 top conv1
    char* resnet = strstr(json, "\"resnet50\"");
    char* conv1 = strstr(resnet, "\"conv1\"");
    parse_conv_config(conv1, &config->resnet50.conv1);

    //maxpool
    char* maxpool = strstr(resnet, "\"maxpool\"");
    parse_maxpool_config(maxpool, &config->resnet50.maxpool);

    // ResBlock array (only handles 1 for simplicity)
    config->resnet50.num_resblock = extract_int_value("\"num_resblock\"", resnet);
    config->resnet50.resblock = malloc(sizeof(ResidualConfig) * config->resnet50.num_resblock);
    char* resblock = strstr(resnet, "\"resblock\"");
    if (!resblock) return -2;

    for (int i = 0; i < config->resnet50.num_resblock; i++) {
        // For simplicity, we’ll parse only one resblock from index 0
        ResidualConfig* rb = &config->resnet50.resblock[i];
        rb->num_bottleneck = extract_int_value("\"num_bottleneck\"", resblock);

        // Parse shortcut
        char* shortcut = strstr(resblock, "\"shortcut\"");
        parse_conv_config(shortcut, &rb->shortcut);

        // Parse bottleneck convs (3 max)
        rb->conv = (ConvConfig*)malloc(sizeof(ConvConfig) * rb->num_bottleneck * CONV_OP_PER_BUTTLENECK);
        char* conv_array = strstr(resblock, "\"conv\"");
        for (int j = 0; j < rb->num_bottleneck*CONV_OP_PER_BUTTLENECK; j++) {
            // crude jump to next object
            char* conv = strchr(conv_array, '{');
            if (!conv) break;
            conv_array = strchr(conv + 1, '}');
            if (!conv_array) break;

            size_t block_len = conv_array - conv + 1;
            char* conv_block = (char*)malloc(block_len + 1);
            strncpy(conv_block, conv, block_len);
            conv_block[block_len] = '\0';

            parse_conv_config(conv_block, &(rb->conv[j]));
            free(conv_block);
        }
        resblock = conv_array; // update string
    }


    // conv2
    char* conv2 = strstr(resnet, "\"conv2\"");
    parse_conv_config(conv2, &config->resnet50.conv2);

    free(json);
    return 0;
}

void free_config(DETR* detr)
{
    DETRConfig *config = &(detr->config);
    if (!config) return;

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

DATA_TYPE *fake_weights;

int load_weight(const char* filename, DETR* detr) {

    DETRConfig *config = &(detr->config);
    DETRWeights *weights = &(detr->weights);

    printf("load weights file: %s\n", filename);

    /* open file */
    // TBD

    /* map */
    // TBD


    // fake weights
    fake_weights = (DATA_TYPE*)malloc(sizeof(DATA_TYPE) * 1024 * 1024 * 10); // 40MB
    if(!fake_weights){
        printf("fake_weights failed\n");
        return -1;
    }

    /* point weights */
    int n_encoder_layers = config->transformer.n_encoder_layers;
    int n_decoder_layers = config->transformer.n_decoder_layers;

    // transformer
    weights->encoder = (EncoderWeights*)malloc(sizeof(EncoderWeights) * n_encoder_layers);
    for(int i=0;i<n_encoder_layers;i++){
        weights->encoder[i].pos_embedding = fake_weights;    // (seq_len, dim)
        // weights for matmuls. note dim == n_heads * head_size
        weights->encoder[i].wq = fake_weights; // (layer, dim, n_heads * head_size)
        weights->encoder[i].bq = fake_weights; // (layer, dim)
        weights->encoder[i].wk = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->encoder[i].bk = fake_weights; // (layer, dim)
        weights->encoder[i].wv = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->encoder[i].bv = fake_weights; // (layer, dim)
        weights->encoder[i].wo = fake_weights; // (layer, dim,  dim)
        weights->encoder[i].bo = fake_weights; // (layer, dim)
        // weights for layernorm
        weights->encoder[i].wln1 = fake_weights;  // (layer, dim)
        weights->encoder[i].bln1 = fake_weights;  // (layer, dim)
        weights->encoder[i].wln2 = fake_weights;  // (layer, dim)
        weights->encoder[i].bln2 = fake_weights;  // (layer, dim)
        // weights for ffn
        weights->encoder[i].w1 = fake_weights; // (layer, dim, hidden_dim)
        weights->encoder[i].b1 = fake_weights; // (layer, hidden_dim)
        weights->encoder[i].w2 = fake_weights; // (layer, hidden_dim, dim)
        weights->encoder[i].b2 = fake_weights; // (layer, dim)
    }
    weights->decoder = (DecoderWeights*)malloc(sizeof(DecoderWeights) * n_decoder_layers);
    for(int i=0;i<n_decoder_layers;i++){
        // position embedding
        weights->decoder[i].pos_embedding = fake_weights;    // (seq_len, dim)
        // weights for matmuls. note dim == n_heads * head_size
        weights->decoder[i].wq = fake_weights; // (layer, dim, n_heads * head_size)
        weights->decoder[i].bq = fake_weights; // (layer, dim)
        weights->decoder[i].wk = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->decoder[i].bk = fake_weights; // (layer, dim)
        weights->decoder[i].wv = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->decoder[i].bv = fake_weights; // (layer, dim)
        weights->decoder[i].wo = fake_weights; //  (layer, dim,  dim)
        weights->decoder[i].bo = fake_weights; // (layer, dim)
        // query position embedding
        weights->decoder[i].query_pos_embedding = fake_weights;    // (seq_len, dim)
        // weights for matmuls. note dim == n_heads * head_size
        weights->decoder[i].wq2 = fake_weights; // (layer, dim, n_heads * head_size)
        weights->decoder[i].bq2 = fake_weights; // (layer, dim)
        weights->decoder[i].wk2 = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->decoder[i].bk2 = fake_weights; // (layer, dim)
        weights->decoder[i].wv2 = fake_weights; // (layer, dim,  n_heads * head_size)
        weights->decoder[i].bv2 = fake_weights; // (layer, dim)
        weights->decoder[i].wo2 = fake_weights; // (layer, dim, dim)
        weights->decoder[i].bo2 = fake_weights; // (layer, dim)
        // weights for layernorm
        weights->decoder[i].wln1 = fake_weights;  // (layer, dim)
        weights->decoder[i].bln1 = fake_weights;  // (layer, dim)
        weights->decoder[i].wln2 = fake_weights;  // (layer, dim)
        weights->decoder[i].bln2 = fake_weights;  // (layer, dim)
        weights->decoder[i].wln3 = fake_weights;  // (layer, dim)
        weights->decoder[i].bln3 = fake_weights;  // (layer, dim)
        // weights for ffn
        weights->decoder[i].w1 = fake_weights; // (layer, dim, hidden_dim)
        weights->decoder[i].b1 = fake_weights; // (layer, hidden_dim)
        weights->decoder[i].w2 = fake_weights; // (layer, hidden_dim, dim)
        weights->decoder[i].b2 = fake_weights; // (layer, dim)
    }

    // output embed
    weights->outputembed.class_w = fake_weights; // (num_classes, dim)
    weights->outputembed.class_b = fake_weights; // (num_classes)
    weights->outputembed.bbox_w1 = fake_weights; // (dim, dim)
    weights->outputembed.bbox_b1 = fake_weights; // (dim)
    weights->outputembed.bbox_w2 = fake_weights; // (dim, dim)
    weights->outputembed.bbox_b2 = fake_weights; // (dim)
    weights->outputembed.bbox_w3 = fake_weights; // (4, dim)
    weights->outputembed.bbox_b3 = fake_weights; // (4)

    // resnet50
    weights->resnet50.conv1.weight = fake_weights;
    weights->resnet50.conv1.bias = fake_weights;
    weights->resnet50.bn1.weight = fake_weights;
    weights->resnet50.bn1.bias = fake_weights;

    int num_resblock = config->resnet50.num_resblock;
    weights->resnet50.resblock = (ResidualWeights*)malloc(num_resblock * sizeof(ResidualWeights));
    for (int i = 0; i < num_resblock; i++) {
        ResidualWeights* rb = &weights->resnet50.resblock[i];

        int num_bottleneck = config->resnet50.resblock[i].num_bottleneck;

        // Free the conv array inside each resblock
        rb->shortcut.weight = fake_weights;
        rb->shortcut.bias = fake_weights;
        rb->bn_shortcut.weight = fake_weights;
        rb->bn_shortcut.bias = fake_weights;

        rb->conv = (ConvWeights*)malloc(sizeof(ConvWeights) * num_bottleneck * CONV_OP_PER_BUTTLENECK);
        rb->bn = (BatchNormWeights*)malloc(sizeof(BatchNormWeights) * num_bottleneck * CONV_OP_PER_BUTTLENECK);
        for(int j = 0;j < num_bottleneck * CONV_OP_PER_BUTTLENECK; j++){
            rb->conv[j].weight = fake_weights;
            rb->conv[j].bias = fake_weights;
            rb->bn[j].weight = fake_weights;
            rb->bn[j].bias = fake_weights;
        }
    }

    weights->resnet50.conv2.weight = fake_weights;
    weights->resnet50.conv2.bias = fake_weights;
}

void free_weights(DETR* detr)
{
    DETRConfig *config = &(detr->config);
    DETRWeights *weights = &(detr->weights);

    if (!weights) return;

    if (weights->encoder){
        free(weights->encoder);
        weights->encoder = NULL;
    }

    if (weights->decoder){
        free(weights->decoder);
        weights->decoder = NULL;
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
    // TBD

    /* close file */
    // TBD

    free(fake_weights);
}

/*
 *  API
 */

void print_conv_config(int level, const char* name, const ConvConfig *config){
    for(int l=0;l<level;l++)printf("  ");
    printf("Conv2D[%s]: in_channels= %3d, out_channels= %3d, kernel_size= %d, stride= %d, padding= %d\n", name,
        config->in_channels,
        config->out_channels,
        config->kernel_size,
        config->stride,
        config->padding);
}

void print_config(DETRConfig *config)
{
    char buffer[64];
    printf("==================== DETR Configuration ====================\n");

    // Transformer
    printf("Transformer:\n");
    printf("  Dimension: %d\n", config->transformer.dim);
    printf("  Hidden Dimension: %d\n", config->transformer.hidden_dim);
    printf("  Encoder Layers: %d\n", config->transformer.n_encoder_layers);
    printf("  Decoder Layers: %d\n", config->transformer.n_decoder_layers);
    printf("  Heads: %d\n", config->transformer.n_heads);
    printf("  Encoder Sequence Length: %d\n", config->transformer.encoder_seq_len);
    printf("  Decoder Sequence Length: %d\n", config->transformer.decoder_seq_len);

    // ResNet50
    printf("\nResNet50:\n");\
    print_conv_config(1, "conv1", &(config->resnet50.conv1));
    printf("  Maxpool2D: kernel_size= %d, stride= %d, padding= %d\n",
        config->resnet50.conv1.kernel_size,
        config->resnet50.conv1.stride,
        config->resnet50.conv1.padding);
    for (int i = 0; i < config->resnet50.num_resblock; i++) {
        printf("  Residual Block %d:\n", i);
        printf("    num_bottleneck: %d\n",config->resnet50.resblock[i].num_bottleneck);
        print_conv_config(2, "shortcut", &(config->resnet50.resblock[i].shortcut));
        printf("    [\n");
        for (int j = 0; j < config->resnet50.resblock[i].num_bottleneck; j++) {
            for(int k=0; k < CONV_OP_PER_BUTTLENECK;k++){
                sprintf(buffer, "%d-%d",j,k);
                print_conv_config(3, buffer, &(config->resnet50.resblock[i].conv[CONV_ID(j,k)]));
            }
        }
        printf("\n    ]\n");
    }
    print_conv_config(1, "conv2", &(config->resnet50.conv2));

    printf("\nOutput Config:\n");
    printf("  Number of Classes: %d\n", config->num_classes);
    printf("  Number of Boxes: %d\n", config->num_boxes);
    printf("============================================================\n");
}


int init_detr(DETR* detr, const char* config_file, const char* weights_file)
{
    // Load configuration and weights from files
    // This is a placeholder for actual implementation
    int r;
    r = load_config(config_file, detr);
    if(r) return r;
    r = load_weight(weights_file, detr);
    if(r) return r;

    printf("DETR initialized with config and weights.\n");
}

void free_detr(DETR* detr)
{
    free_weights(detr);
    free_config(detr);
    printf("DETR freed.\n");
}

/*
 *  Runstate
 */

/* malloc and free of ConvolutionRunState */
void malloc_conv2D_run_state(ConvolutionRunState* r) {
    if(r->x != NULL) free(r->x);

    r->x = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * CONV_SIZE(*r));
    if(r->x == NULL){
        fprintf(stderr, "Molloc failed for ConvolutionRunState\n.");
        exit(EXIT_FAILURE);
    }
}

void free_conv2D_run_state(ConvolutionRunState* r) {
    if(r->x == NULL) return;
    free(r->x);
    r->x = NULL;
}

/* malloc and free of EncoderRunState */
void malloc_encoder_run_state(const TransformerConfig* config, EncoderRunState* r) {

    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int seq_len = config->encoder_seq_len;
    int n_heads = config->n_heads;

    r->x = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->xb = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->xb2 = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->hb = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * hidden_dim * seq_len);
    r->hb2 = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * hidden_dim * seq_len);
    r->q = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->k = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->v = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->att = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len * seq_len);
    r->att_mask = (int*)malloc(sizeof(int) * seq_len * seq_len);
}

void free_encoder_run_state(EncoderRunState* r) {
    free(r->x);
    free(r->xb);
    free(r->xb2);
    free(r->hb);
    free(r->hb2);
    free(r->q);
    free(r->k);
    free(r->v);
    free(r->att);
    free(r->att_mask);
}

/* malloc and free of DecoderRunState */
void malloc_decoder_run_state(const TransformerConfig* config, DecoderRunState* r) {

    int dim = config->dim;
    int hidden_dim = config->hidden_dim;
    int seq_len = config->decoder_seq_len;
    int encoder_seq_len = config->encoder_seq_len;
    int n_heads = config->n_heads;

    r->x = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->xb = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->xb2 = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    r->hb = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * hidden_dim * seq_len);
    r->hb2 = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * hidden_dim * seq_len);
    r->q = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
    if(seq_len < encoder_seq_len){
        r->k = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * encoder_seq_len); // select bigger size
        r->v = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * encoder_seq_len); // select bigger size
        r->att = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len * encoder_seq_len);
    }else{
        r->k = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len); // select bigger size
        r->v = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len); // select bigger size
        r->att = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len * seq_len);
    }

    r->f = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * encoder_seq_len);
    r->f_embed = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * encoder_seq_len);
    r->target = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len);
}

void free_decoder_run_state(DecoderRunState* r) {
    free(r->x);
    free(r->xb);
    free(r->xb2);
    free(r->hb);
    free(r->hb2);
    free(r->q);
    free(r->k);
    free(r->v);
    free(r->att);
    free(r->f);
    free(r->f_embed);
    free(r->target);
}
/* malloc and free of OutputRunState */
void malloc_output_run_state(const DETRConfig* config, OutputRunState* r) {
    int dim = config->transformer.dim;
    int num_boxes = config->num_boxes;
    int num_classes = config->num_classes;

    r->x = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * num_boxes);
    r->xb = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * num_boxes);
    r->classes = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * num_classes * num_boxes);
    r->bbox = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * BBOX_COORDS * num_boxes);
}

void free_output_run_state(OutputRunState* r) {
    free(r->x);
    free(r->xb);
    free(r->classes);
    free(r->bbox);
}

/*
 * model inference
 */
void forward(DETR* detr, ConvolutionRunState* image, OutputRunState* result) {
    printf("forward\n");
    // placeholder
    ConvolutionRunState resnet50_out = {NULL, 0, 0, 0};
    EncoderRunState encoder_runstate;
    DecoderRunState decoder_runstate;
    size_t feature_size;
    TransformerConfig *t_cfg = &(detr->config.transformer);

    // init runstate
    malloc_encoder_run_state(t_cfg ,&encoder_runstate);
    malloc_decoder_run_state(t_cfg ,&decoder_runstate);

    /* CNN backbone */
    forward_resnet50(&(detr->config.resnet50), &(detr->weights.resnet50), image, &resnet50_out); // Run ResNet50
    memcpy(resnet50_out.x,encoder_runstate.x,CONV_SIZE(resnet50_out)*DATA_TYPE_SIZE); // Transpose
    free_conv2D_run_state(&resnet50_out); // release memory

    /* Transformer - Encoder */
    forward_encoder(t_cfg, detr->weights.encoder, &encoder_runstate); // Run Transformer Encoder
    feature_size = t_cfg->encoder_seq_len * t_cfg->dim * DATA_TYPE_SIZE;
    memcpy(encoder_runstate.x,decoder_runstate.f,feature_size); // feed to decoder
    /* Transformer - Decoder */
    forward_decoder(t_cfg, detr->weights.decoder, &decoder_runstate); // Run Transformer Decoder
    feature_size = detr->config.num_boxes * t_cfg->dim * DATA_TYPE_SIZE;
    memcpy(result->x,decoder_runstate.x,feature_size); // feed to decoder
    /* Output Embedded */
    forward_output(&(detr->config), &(detr->weights.outputembed), result);

    free_encoder_run_state(&encoder_runstate);
    free_decoder_run_state(&decoder_runstate);
}


void forward_resnet50(ResNet50Config* config, ResNet50Weights* weights, ConvolutionRunState* image, ConvolutionRunState* result){
    printf("forward_resnet50\n");
    // runstate placeholder
    ConvolutionRunState r = {NULL, 0, 0, 0};
    ConvolutionRunState r_1 = {NULL, 0, 0, 0};
    ConvolutionRunState r_2 = {NULL, 0, 0, 0};
    ConvolutionRunState shortcut = {NULL, 0, 0, 0};

    // 7*7 conv 2D
    printf("-------------------------------[conv1]\n");
    conv2D(&r_1, image, &(config->conv1), &(weights->conv1));
    batchnorm2D(&r_1, &r_1, &(weights->bn1)); // with batchnorm2D
    relu(r_1.x, r_1.x, CONV_SIZE(r_1));
    maxpooling2D(&r, &r_1, &(config->maxpool));

    // residul blocks
    for(int rb = 0; rb < config->num_resblock; rb++){ // residual block
        for(int nb = 0;nb < config->resblock[rb].num_bottleneck; nb++){ // num of bottleneck
            printf("------------------------------- resblock [%d - %d]\n", rb, nb);
            // Bottleneck
            conv2D(
                &r_1,
                &r,
                &(config->resblock[rb].conv[CONV_ID(nb, 0)]),
                &(weights->resblock[rb].conv[CONV_ID(nb, 0)]));
            batchnorm2D(&r_1, &r_1, &(weights->resblock[rb].bn[CONV_ID(nb, 0)])); // with batchnorm2D

            relu(r_1.x, r_1.x, CONV_SIZE(r_1));

            conv2D(
                &r_2,
                &r_1,
                &(config->resblock[rb].conv[CONV_ID(nb, 1)]),
                &(weights->resblock[rb].conv[CONV_ID(nb, 1)]));
            batchnorm2D(&r_2, &r_2, &(weights->resblock[rb].bn[CONV_ID(nb, 1)])); // with batchnorm2D

            relu(r_2.x, r_2.x, CONV_SIZE(r_2));

            conv2D(
                &r_1,
                &r_2,
                &(config->resblock[rb].conv[CONV_ID(nb, 2)]),
                &(weights->resblock[rb].conv[CONV_ID(nb, 2)]));
            batchnorm2D(&r_1, &r_1, &(weights->resblock[rb].bn[CONV_ID(nb, 2)])); // with batchnorm2D

            // Shortcut
            if(nb == 0){
                conv2D(
                    &shortcut,
                    &r,
                    &(config->resblock[rb].shortcut),
                    &(weights->resblock[rb].shortcut));
                batchnorm2D(&shortcut, &shortcut, &(weights->resblock[rb].bn_shortcut)); // with batchnorm2D

                CONV_SHAPE_COPY(r,r_1);
                malloc_conv2D_run_state(&r);
                add(r.x, r_1.x, shortcut.x,CONV_SIZE(r_1)); // add(input1, intput2, output)
            }else{
                add(r.x, r_1.x, r.x,CONV_SIZE(r_1)); // add(input1, intput2, output)
            }

            relu(r.x, r.x, CONV_SIZE(r));
        }
    }
    printf("-------------------------------[conv2]\n");
    // 1*1 conv 2D (projection)
    conv2D(result, &r, &(config->conv2), &(weights->conv2));

    // free placeholder
    free_conv2D_run_state(&r);
    free_conv2D_run_state(&r_1);
    free_conv2D_run_state(&r_2);
    free_conv2D_run_state(&shortcut);
}

void forward_encoder(TransformerConfig* c, EncoderWeights* w, EncoderRunState* r){
    printf("forward_encoder\n");
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int head_size = dim / c->n_heads;
    int seq_len = c->encoder_seq_len;
    int *att_mask = r->att_mask;
    int token_size = seq_len * dim;

    DATA_TYPE head_size_sqrt = SQRT(head_size);

    for(unsigned long long l = 0; l < c->n_encoder_layers; l++) {
        printf("------------------------------- encoder[%lld]\n",l);
        // pos embedding
        add(r->xb, r->x, w[l].pos_embedding, token_size);
        // q, k ,v
        gemm(r->q, r->xb, w[l].wq, w[l].bq, seq_len, dim, dim);
        gemm(r->k, r->xb, w[l].wk, w[l].bk, seq_len, dim, dim);
        gemm(r->v, r->x, w[l].wv, w[l].bv, seq_len, dim, dim);

        // multi-head attention. iterate over all heads
        int h;
        //#pragma omp parallel for private(h)
        for (h = 0; h < c->n_heads; h++) {
            // get q,k,v of this head
            DATA_TYPE* Q = r->q + h * head_size * seq_len;
            DATA_TYPE* K = r->k + h * head_size * seq_len;
            DATA_TYPE* V = r->v + h * head_size * seq_len;
            DATA_TYPE* att = r->att + h * head_size * seq_len * seq_len;
            DATA_TYPE* xb = r->xb + h * head_size * seq_len;
            // q,k -> attention score
            for(int i = 0; i < seq_len; i++){
                for(int j = 0; j < seq_len; j++){
                    // skip if mask is 0(false);
                    if(!att_mask[i * seq_len + j]){
                        att[i * seq_len + j] = 0;
                        continue;
                    }

                    DATA_TYPE score = 0;
                    for (int d = 0; d < head_size; d++){
                        score += Q[d * seq_len + i] * K[d * seq_len + j];
                    }
                    att[i * seq_len + j] = score / head_size_sqrt;
                }
            }
            // softmax
            softmax(att, att, head_size * seq_len);
            // output of attention = att @ V
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_size; d++) {
                    DATA_TYPE sum = 0;
                    for (int j = 0; j < seq_len; j++) {
                        sum += att[i * seq_len + j] * V[d * seq_len + j];
                    }
                    xb[d * seq_len + i] = sum;
                }
            }
        }
        // output gemm
        gemm(r->xb2, r->xb, w[l].wo, w[l].bo, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l].wln1, w[l].bln1, seq_len, dim);
        // ffn
        gemm(r->hb, r->x, w[l].w1, w[l].b1, seq_len, dim, hidden_dim);
        relu(r->hb2, r->hb, seq_len*hidden_dim);
        gemm(r->xb, r->hb2, w[l].w2, w[l].b2, seq_len, hidden_dim, dim);
        // residual connection
        add(r->x, r->x, r->xb, token_size);
        layernorm(r->x, r->x, w[l].wln2, w[l].bln2, seq_len, dim);
    }
}

void forward_decoder(TransformerConfig* c, DecoderWeights* w, DecoderRunState* r){
    printf("forward_decoder\n");
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int head_size = dim / c->n_heads;
    int seq_len = c->decoder_seq_len;
    int encoder_seq_len = c->encoder_seq_len;
    int token_size = seq_len * dim;
    int encoder_token_size = encoder_seq_len * dim;

    DATA_TYPE head_size_sqrt = SQRT(head_size);

    // decoder feature pos embedding
    add(r->f_embed, r->f, w[0].pos_embedding, encoder_token_size);

    for(unsigned long long l = 0; l < c->n_decoder_layers; l++) {
        printf("------------------------------- decoder[%lld]\n",l);
        // query pos embedding
        add(r->xb, r->x, w[l].query_pos_embedding, token_size);
        // q, k ,v
        gemm(r->q, r->xb, w[l].wq, w[l].bq, seq_len, dim, dim);
        gemm(r->k, r->xb, w[l].wk, w[l].bk, seq_len, dim, dim);
        gemm(r->v, r->x, w[l].wv, w[l].bv, seq_len, dim, dim);

        // multi-head attention 1. iterate over all heads
        int h;
        //#pragma omp parallel for private(h)
        for (h = 0; h < c->n_heads; h++) {
            // get q,k,v of this head
            DATA_TYPE* Q = r->q + h * head_size * seq_len;
            DATA_TYPE* K = r->k + h * head_size * seq_len;
            DATA_TYPE* V = r->v + h * head_size * seq_len;
            DATA_TYPE* att = r->att + h * head_size * seq_len * seq_len;
            DATA_TYPE* xb = r->xb + h * head_size * seq_len;
            // q,k -> attention score
            for(int i = 0; i < seq_len; i++){
                for(int j = 0; j < seq_len; j++){
                    DATA_TYPE score = 0;
                    for (int d = 0; d < head_size; d++){
                        score += Q[d * seq_len + i] * K[d * seq_len + j];
                    }
                    att[i * seq_len + j] = score / head_size_sqrt;
                }
            }
            // softmax
            softmax(att, att, head_size * seq_len);
            // output of attention = att @ V
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_size; d++) {
                    DATA_TYPE sum = 0;
                    for (int j = 0; j < seq_len; j++) {
                        sum += att[i * seq_len + j] * V[d * seq_len + j];
                    }
                    xb[d * seq_len + i] = sum;
                }
            }
        }
        // output gemm
        gemm(r->xb2, r->xb, w[l].wo, w[l].bo, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l].wln1, w[l].bln1, seq_len, dim);

        // query pos embedding
        add(r->xb, r->x, w[l].query_pos_embedding, token_size);

        // q, k ,v
        gemm(r->q, r->xb, w[l].wq2, w[l].bq2, seq_len, dim, dim);
        gemm(r->k, r->f_embed, w[l].wk2, w[l].bk2, encoder_seq_len, dim, dim);
        gemm(r->v, r->f, w[l].wv2, w[l].bv2, encoder_seq_len, dim, dim);

        // multi-head attention 2. iterate over all heads
        //#pragma omp parallel for private(h)
        for (h = 0; h < c->n_heads; h++) {
            // get q,k,v of this head
            DATA_TYPE* Q = r->q + h * head_size * seq_len;
            DATA_TYPE* K = r->k + h * head_size * encoder_seq_len;
            DATA_TYPE* V = r->v + h * head_size * encoder_seq_len;
            DATA_TYPE* att = r->att + h * head_size * seq_len * encoder_seq_len;
            DATA_TYPE* xb = r->xb + h * head_size * seq_len;
            // q,k -> attention score
            for(int i = 0; i < seq_len; i++){
                for(int j = 0; j < encoder_seq_len; j++){
                    DATA_TYPE score = 0;
                    for (int d = 0; d < head_size; d++){
                        score += Q[d * seq_len + i] * K[d * encoder_seq_len + j];
                    }
                    att[i * encoder_seq_len + j] = score / head_size_sqrt;
                }
            }
            // softmax
            softmax(att, att, head_size * seq_len);
            // output of attention = att @ V
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_size; d++) {
                    DATA_TYPE sum = 0;
                    for (int j = 0; j < encoder_seq_len; j++) {
                        sum += att[i * encoder_seq_len + j] * V[d * encoder_seq_len + j];
                    }
                    xb[d * seq_len + i] = sum;
                }
            }
        }
        // output gemm
        gemm(r->xb2, r->xb, w[l].wo2, w[l].bo2, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l].wln1, w[l].bln1, seq_len, dim);
        // ffn
        gemm(r->hb, r->x, w[l].w1,w[l].b1, seq_len, dim, hidden_dim);
        relu(r->hb2, r->hb, seq_len*hidden_dim);
        gemm(r->xb, r->hb2, w[l].w2, w[l].b2, seq_len, hidden_dim, dim);
        // residual connection
        add(r->x, r->x, r->xb, token_size);
        layernorm(r->x, r->x, w[l].wln2, w[l].bln2, seq_len, dim);


    }
    // copy target
    memcpy(r->x, r->target, token_size*DATA_TYPE_SIZE);
}

void forward_output(DETRConfig* c, OutputEmbedWeights* w, OutputRunState* r){
    printf("forward_output\n");
    int num_classes = c->num_classes;
    int num_boxes = c->num_boxes;
    int dim = c->transformer.dim;
    // classes
    gemm(r->classes, r->x, w->class_w, w->class_b, num_boxes, dim, num_classes);
    // bbox
    gemm(r->xb, r->x, w->bbox_w1, w->bbox_b1, num_boxes, dim, dim);
    relu(r->x, r->xb, num_boxes*dim);
    gemm(r->xb, r->x, w->bbox_w2, w->bbox_b2, num_boxes, dim, dim);
    relu(r->x, r->xb, num_boxes*dim);
    gemm(r->bbox, r->x, w->bbox_w3, w->bbox_b3, num_boxes, dim, BBOX_COORDS);
    sigmoid(r->bbox, r->bbox, num_boxes*BBOX_COORDS);
}

/*
 *  Operations
 */
 void conv2D(ConvolutionRunState *output, const ConvolutionRunState *input, const ConvConfig *config, const ConvWeights *weights) {
    printf("conv2D\n");
    assert(weights->weight != NULL);
    assert(weights->bias != NULL);
    assert(input->x != NULL);
    assert(config->in_channels == input->channels);

    // calculate dimensions from input and output
    int output_height = (input->height + 2 * config->padding - config->kernel_size) / config->stride + 1;
    int output_width = (input->width + 2 * config->padding - config->kernel_size) / config->stride + 1;

    // prepare output ConvolutionRunState
    output->height = output_height;
    output->width = output_width;
    output->channels = config->out_channels;
    malloc_conv2D_run_state(output);

    // Perform convolution
    for (int oc = 0; oc < config->out_channels; oc++) {
        for (int ih = -config->padding; ih <= input->height + config->padding - config->kernel_size; ih += config->stride) {
            for (int iw = -config->padding; iw <= input->width + config->padding - config->kernel_size; iw += config->stride) {
                DATA_TYPE sum = 0;
                for (int ic = 0; ic < config->in_channels; ic++) {
                    int kh = -config->padding;
                    for (int ki = 0; ki < config->kernel_size; ki++, kh++) {
                        if (ih + kh >= 0 && ih + kh < input->height && iw + ki >= 0 && iw + ki < input->width) {
                            int idx_input = ic * input->height * input->width +
                                            (ih + kh) * input->width +
                                            (iw + ki);
                            int idx_weight = oc * config->kernel_size * config->kernel_size +
                                            ic * config->kernel_size * config->kernel_size +
                                            kh * config->kernel_size +
                                            ki;
                            sum += input->x[idx_input] * weights->weight[idx_weight];
                        }
                    }
                }
                int oh = (ih - config->padding) / config->stride;
                int ow = (iw - config->padding) / config->stride;
                if (oh >= 0 && oh < output->height && ow >= 0 && ow < output->width) {
                    int idx_output =    oc * output->height * output->width +
                                        oh * output->width +
                                        ow;
                    output->x[idx_output] = sum + weights->bias[oc];
                }
            }
        }
    }

}

void maxpooling2D(ConvolutionRunState *output, const ConvolutionRunState *input, const MaxPoolConfig* config) {
    printf("maxpooling2D\n");
    assert(input->x != NULL);

    int padded_height = input->height + 2 * config->padding;
    int padded_width = input->width + 2 * config->padding;

    int height = (padded_height - config->kernel_size) / config->stride + 1;
    int width  = (padded_width - config->kernel_size) / config->stride + 1;

    output->height = height;
    output->width = width;
    output->channels = input->channels;
    malloc_conv2D_run_state(output);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int ch = 0; ch < input->channels; ++ch) {

                DATA_TYPE max_value = -__FLT_MAX__;

                for (int r = 0; r < config->kernel_size; ++r) {
                    for (int c = 0; c < config->kernel_size; ++c) {
                        int in_h = h * config->stride + r - config->padding;
                        int in_w = w * config->stride + c - config->padding;

                        if (in_h >= 0 && in_h < input->height &&
                            in_w >= 0 && in_w < input->width) {

                            DATA_TYPE value = input->x[(in_h * input->width + in_w) * input->channels + ch];
                            if (value > max_value) {
                                max_value = value;
                            }
                        }
                    }
                }

                output->x[(h * width + w) * output->channels + ch] = max_value;
            }
        }
    }
}


void layernorm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n, int dim) {
    printf("layernorm\n");
    const DATA_TYPE eps = 1e-5;

    for (int i = 0; i < n; i++) {
        DATA_TYPE mean = 0;
        DATA_TYPE var = 0;

        // Compute mean
        for (int j = 0; j < dim; j++) {
            mean += x[i * dim + j];
        }
        mean /= dim;

        // Compute variance
        for (int j = 0; j < dim; j++) {
            DATA_TYPE diff = x[i * dim + j] - mean;
            var += diff * diff;
        }
        var /= dim;
        DATA_TYPE inv_std = 1.0f / SQRT(var + eps);

        // Normalize + affine transform
        for (int j = 0; j < dim; j++) {
            DATA_TYPE norm = (x[i * dim + j] - mean) * inv_std;
            out[i * dim + j] = norm * w[j] + b[j];
        }
    }
}

void batchnorm2D(ConvolutionRunState *output, const ConvolutionRunState *input, BatchNormWeights *bn){
    printf("batchnorm2D\n");
    int C = input->channels;
    int H = input->height;
    int W = input->width;

    for (int c = 0; c < C; c++) {
        DATA_TYPE gamma = bn->weight[c];
        DATA_TYPE beta = bn->bias[c];
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int index = c * H * W + h * W + w;
                output->x[index] = gamma * input->x[index] + beta;
            }
        }
    }
}


void gemm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n, int id, int od){
    printf("gemm\n");
    // W (id, od) @ x(id, n) -> xout (od, n)
    int i;
    #pragma omp parallel for private(i)
    for (int i = 0; i < n * od; i++) {
        int o = i / n; // output dimension index
        int batch = i % n; // batch index
        out[i] = b[o];
        for (int j = 0; j < id; j++) {
            out[i] += w[j + o*n] * x[j * n + batch];
        }
    }
}

void add(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* y, int size) {
    printf("add\n");
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = x[i] + y[i];
    }
}

/*
 *  Activation Function
 */

void relu(DATA_TYPE* out, DATA_TYPE* x, int size) {
    printf("relu\n");
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = (x[i] < 0)? 0 : x[i];
    }
}

void softmax(DATA_TYPE* out, DATA_TYPE* x, int size) {
    printf("softmax\n");
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
    #pragma omp parallel for private(i)
    for (int i = 0; i < size; i++) {
        out[i] /= sum;
    }
}

void sigmoid(DATA_TYPE* out, DATA_TYPE* x, int size) {
    printf("sigmoid\n");
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = 1 / (1 + expf(-x[i]));
    }
}
