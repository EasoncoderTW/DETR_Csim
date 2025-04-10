#include "model.h"


void print_config(DETR *detr)
{
    printf("Transformer Config:\n");
    printf("  Dimension: %d\n", detr->config.tranformer.dim);
    printf("  Hidden Dimension: %d\n", detr->config.tranformer.hidden_dim);
    printf("  Encoder Layers: %d\n", detr->config.tranformer.n_encoder_layers);
    printf("  Decoder Layers: %d\n", detr->config.tranformer.n_decoder_layers);
    printf("  Heads: %d\n", detr->config.tranformer.n_heads);
    printf("  Key/Value Heads: %d\n", detr->config.tranformer.n_kv_heads);
    printf("  Sequence Length: %d\n", detr->config.tranformer.seq_len);

    printf("\nResNet50 Config:\n");\
    for (int i = 0; i < RESIDUAL_BLOCKS; i++) {
        printf("  Residual Block %d:\n", i);
        printf("    Conv[shortcut]: in_channels= %3d, out_channels= %3d, kernel_size= %d, stride= %d, padding= %d\n", j,
            detr->config.resnet50.resblock[i].shortcut.in_channels,
            detr->config.resnet50.resblock[i].shortcut.out_channels,
            detr->config.resnet50.resblock[i].shortcut.kernel_size,
            detr->config.resnet50.resblock[i].shortcut.stride,
            detr->config.resnet50.resblock[i].shortcut.padding);
        for(int j = 0;j < config->resnet50.resblock[i].num_bottleneck; j++){ // num of bottleneck
            printf("    Conv[%d]: in_channels= %3d, out_channels= %3d, kernel_size= %d, stride= %d, padding= %d\n", j,
                detr->config.resnet50.resblock[i].conv[j].in_channels,
                detr->config.resnet50.resblock[i].conv[j].out_channels,
                detr->config.resnet50.resblock[i].conv[j].kernel_size,
                detr->config.resnet50.resblock[i].conv[j].stride,
                detr->config.resnet50.resblock[i].conv[j].padding);
            // Repeat for other conv layers in the residual block
        }
    }

    printf("\nDETR Config:\n");
    printf("  Number of Classes: %d\n", detr->config.num_classes);
    printf("  Number of Boxes: %d\n", detr->config.num_boxes);
}

void init_detr(DETR* detr, const char* config_file, const char* weights_file)
{
    // Load configuration and weights from files
    // This is a placeholder for actual implementation
    memset(detr, 0, sizeof(DETR));

    printf("DETR initialized with config and weights.\n");
}

void free_detr(DETR* detr)
{

    printf("DETR freed.\n");
}

/* malloc and free of ConvolutionRunState */
void malloc_conv2D_run_state(ConvolutionRunState* r) {
    if(r->x != NULL) free(r->x);

    r->x = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * CONV_SIZE(r));
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
    r->att = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * n_heads * seq_len * seq_len);
    r->att_mask = (bool*)malloc(sizeof(bool) * seq_len * seq_len);
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
    }else{
        r->k = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len); // select bigger size
        r->v = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * dim * seq_len); // select bigger size
    }

    r->att = (DATA_TYPE*)malloc(DATA_TYPE_SIZE * n_heads * seq_len * seq_len);

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
    int dim = config->tranformer.dim;
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
    // placeholder
    ConvolutionRunState resnet50_out = {NULL, 0, 0, 0, 0};
    EncoderRunState encoder_runstate;
    DecoderRunState decoder_runstate;
    size_t feature_size;

    // init runstate
    malloc_encoder_run_state(&(detr->config.tranformer) ,&encoder_runstate);
    malloc_decoder_run_state(&(detr->config.tranformer) ,&decoder_runstate);

    /* CNN backbone */
    forward_resnet50(&(detr->config.resnet50), &(detr->weights.resnet50), image, &resnet50_out); // Run ResNet50
    memcpy(resnet50_out.x,encoder_runstate.x,CONV_SIZE(resnet50_out)*DATA_TYPE_SIZE); // Transpose
    free_conv2D_run_state(&resnet50_out); // release memory

    /* Transformer - Encoder */
    forward_encoder(&(detr->config.tranformer), &(detr->weights.encoder), &encoder_runstate); // Run Transformer Encoder
    feature_size = detr->config.encoder_seq_len*detr->config.dim*DATA_TYPE_SIZE;
    memcpy(encoder_runstate.x,decoder_runstate.f,feature_size); // feed to decoder
    /* Transformer - Decoder */
    forward_decoder(&(detr->config.tranformer), &(detr->weights.decoder), &decoder_runstate); // Run Transformer Decoder
    feature_size = detr->config.num_boxes*detr->config.dim*DATA_TYPE_SIZE;
    memcpy(result->x,decoder_runstate.x,feature_size); // feed to decoder
    /* Output Embedded */
    forward_output(&(detr->config), &(detr->weights.outputembed), result);

    free_encoder_run_state(&encoder_runstate);
    free_decoder_run_state(&decoder_runstate);
}


void forward_resnet50(ResNet50Config* config, ResNet50Weights* weights, ConvolutionRunState* image, ConvolutionRunState* result){
    // runstate placeholder
    ConvolutionRunState r = {NULL, 0, 0, 0, 0};
    ConvolutionRunState r_1 = {NULL, 0, 0, 0, 0};
    ConvolutionRunState r_2 = {NULL, 0, 0, 0, 0};
    ConvolutionRunState shortcut = {NULL, 0, 0, 0, 0};

    // 7*7 conv 2D
    conv2D(&r_1, image, &(config->conv1), &(weights->conv1));
    batchnorm2D(&r_1, &r_1, &(weights->bn1)); // with batchnorm2D
    relu(&(r_1->x), &(r_1->x), CONV_SIZE(r_1));
    maxpooling2D(&r, &r_1, 3); // stride = 3, placeholder

    // residul blocks
    for(int rb = 0; rb < config->num_resblock; rb++){ // residual block
        for(int nb = 0;nb < config->resblock[rb].num_bottleneck; nb++){ // num of bottleneck
            // Bottleneck
            conv2D(
                &r_1,
                &r,
                &(config->resblock[rb].conv[CONV_ID(nb, 0)]),
                &(weights->resblock[rb].conv[CONV_ID(nb, 0)]));
            batchnorm2D(&r_1, &r_1, &(weights->resblock[rb].bn[CONV_ID(nb, 0)])); // with batchnorm2D

            relu(&(r_1->x), &(r_1->x), CONV_SIZE(r_1));

            conv2D(
                &r_2,
                &r_1,
                &(config->resblock[rb].conv[CONV_ID(nb, 1)]),
                &(weights->resblock[rb].conv[CONV_ID(nb, 1)]));
            batchnorm2D(&r_2, &r_2, &(weights->resblock[rb].bn[CONV_ID(nb, 1)])); // with batchnorm2D

            relu(&(r_2->x), &(r_2->x), CONV_SIZE(r_2));

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
                batchnorm2D(&shortcut, &shortcut, &(weights->bn_shortcut)); // with batchnorm2D

                add(&(r->x), &(r_1->x), &(shortcut->x),CONV_SIZE(r_1)); // add(input1, intput2, output)
            }else{
                add(&(r->x), &(r_1->x), &(r->x),CONV_SIZE(r_1)); // add(input1, intput2, output)
            }

            relu(&(r->x), &(r->x), CONV_SIZE(r));
        }
    }
    // 1*1 conv 2D (projection)
    conv2D(&result, &r, &(config->conv2), &(weights->conv2));

    // free placeholder
    free_conv2D_run_state(&r);
    free_conv2D_run_state(&r_1);
    free_conv2D_run_state(&r_2);
    free_conv2D_run_state(&shortcut);
}

void forward_encoder(TransformerConfig* c, EncoderWeights* w, EncoderRunState* r){
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int head_size = dim / c->n_heads;
    int seq_len = c->encoder_seq_len;
    bool *att_mask = r->att_mask;
    int token_size = seq_len * dim;

    DATA_TYPE head_size_sqrt = SQRT(head_size);

    for(unsigned long long l = 0; l < c->n_encoder_layers; l++) {
        // pos embedding
        add(r->xb, r->x, w[l]->pos_embedding, token_size);
        // q, k ,v
        gemm(r->q, r->xb, w[l]->wq, w[l]->bq, seq_len, dim, dim);
        gemm(r->k, r->xb, w[l]->wk, w[l]->bk, seq_len, dim, dim);
        gemm(r->v, r->x, w[l]->wv, w[l]->bv, seq_len, dim, dim);

        // multi-head attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
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
                        score += q[d * seq_len + i] * k[d * seq_len + j];
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
        gemm(r->xb2, r->xb, w[l]->wo, w[l]->bo, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l]->wln1, w[l]->bln1, seq_len, dim);
        // ffn
        gemm(r->hb, r->x, w[l]->w1, w[l]->b1, seq_len, dim, hidden_dim);
        relu(r->hb2, r->hb, seq_len*hidden_dim);
        gemm(r->xb, r->hb2, w[l]->w2, w[l]->b2, seq_len, hidden_dim, dim);
        // residual connection
        add(r->x, r->x, r->xb, token_size);
        layernorm(r->x, r->x, w[l]->wln2, w[l]->bln2, seq_len, dim);
    }
}

void forward_decoder(TransformerConfig* c, DecoderWeights* w, DecoderRunState* r){
    int dim = c->dim;
    int hidden_dim = c->hidden_dim;
    int head_size = dim / c->n_heads;
    int seq_len = c->decoder_seq_len;
    int encoder_seq_len = c->encoder_seq_len;
    bool *att_mask = r->att_mask;
    int token_size = seq_len * dim;
    int encoder_token_size = encoder_seq_len * dim;

    DATA_TYPE head_size_sqrt = SQRT(head_size);

    // encoder feature pos embedding
    add(r->f_embed, r->f, w[l]->pos_embedding, encoder_token_size);

    for(unsigned long long l = 0; l < c->n_decoder_layers; l++) {
        // query pos embedding
        add(r->xb, r->x, w[l]->query_pos_embedding, token_size);
        // q, k ,v
        gemm(r->q, r->xb, w[l]->wq, w[l]->bq, seq_len, dim, dim);
        gemm(r->k, r->xb, w[l]->wk, w[l]->bk, seq_len, dim, dim);
        gemm(r->v, r->x, w[l]->wv, w[l]->bv, seq_len, dim, dim);

        // multi-head attention 1. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
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
                        score += q[d * seq_len + i] * k[d * seq_len + j];
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
        gemm(r->xb2, r->xb, w[l]->wo, w[l]->bo, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l]->wln1, w[l]->bln1, seq_len, dim);

        // query pos embedding
        add(r->xb, r->x, w[l]->query_pos_embedding, token_size);

        // q, k ,v
        gemm(r->q, r->xb, w[l]->wq2, w[l]->bq2, seq_len, dim, dim);
        gemm(r->k, r->f_embed, w[l]->wk2, w[l]->bk2, encoder_seq_len, dim, dim);
        gemm(r->v, r->f, w[l]->wv2, w[l]->bv2, encoder_seq_len, dim, dim);

        // multi-head attention 2. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
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
                        score += q[d * seq_len + i] * k[d * encoder_seq_len + j];
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
        gemm(r->xb2, r->xb, w[l]->wo2, w[l]->bo2, seq_len, dim, dim);

        // residual connection back to x
        add(r->xb, r->x, r->xb2, token_size);
        layernorm(r->x, r->xb, w[l]->wln1, w[l]->bln1, seq_len, dim);
        // ffn
        gemm(r->hb, r->x, r->w1,r->b1, seq_len, dim, hidden_dim);
        relu(r->hb2, r->hb, seq_len*hidden_dim);
        gemm(r->xb, r->hb2, r->w2, r->b2, seq_len, hidden_dim, dim);
        // residual connection
        add(r->x, r->x, r->xb, token_size);
        layernorm(r->x, r->x, w[l]->wln2, w[l]->bln2, seq_len, dim);

        // copy target (concat)
        DATA_TYPE *target = r->target + l*token_size;
        memcpy(r->x, target, token_size*DATA_TYPE_SIZE);
    }
}

void forward_output(DETRConfig* c, OutputEmbedWeights* w, OutputRunState* r){
    int num_classes = c->num_classes;
    int num_boxes = c->num_boxes;
    // classes
    gemm(r->classes, r->x, w->class_w, w->class_b, num_boxes, dim, num_classes);
    // bbox
    gemm(r->xb, r->x, w->bbox_w1, w->bbox_b1, num_boxes, dim, dim);
    relu(r->x, r->xb, num_boxes*dim);
    gemm(r->xb, r->x, w->bbox_w2, w->bbox_b2, num_boxes, dim, dim);
    relu(r->x, r->xb, num_boxes*dim);
    gemm(r->num_boxes, r->x, w->bbox_w3, w->bbox_b3, num_boxes, dim, BBOX_COORDS);
    sigmoid(r->num_boxes, r->num_boxes, num_boxes*BBOX_COORDS);
}

/*
 *  Operations
 */
 void conv2D(ConvolutionRunState *output, const ConvolutionRunState *input, const ConvConfig *config, const ConvWeights *weights) {
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

void maxpooling2D(ConvolutionRunState *output, const ConvolutionRunState *input, int stride){

    assert(input->x != NULL);

    int height = input->height / stride;
    int width = input->width / stride;

    // prepare output ConvolutionRunState
    output->height = height;
    output->width = width;
    output->channels = input->channels;
    malloc_conv2D_run_state(output);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            int start_h = h * stride;
            int start_w = w * stride;

            // Find the maximum value in the current pooling region
            DATA_TYPE max_value = input->x[(start_h + 0) * input->width * input->channels +
                                                (start_w + 0) * input->channels];

            for (int r = 1; r < config->kernel_size && start_h + r < input->height; ++r) {
                for (int c = 1; c < config->kernel_size && start_w + c < input->width; ++c) {
                    DATA_TYPE value = input->x[(start_h + r) * input->width * input->channels +
                                                    (start_w + c) * input->channels];
                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }

            // Store the maximum value in the output activation
            output->x[h * width * output->channels + w * output->channels] = max_value;
        }
    }

}

void layernorm(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* w, DATA_TYPE* b, int n, int dim) {
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
    // W (id, od) @ x(id, n) -> xout (od, n)
    int i;
    #pragma omp parallel for private(i)
    for (int i = 0; i < n * od; i++) {
        int o = i / n; // output dimension index
        int b = i % n; // batch index
        out[i] = b[o];
        for (int j = 0; j < id; j++) {
            out[i] += w[j + o*n] * x[j * n + b];
        }
    }
}

void add(DATA_TYPE* out, DATA_TYPE* x, DATA_TYPE* y, int size) {
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
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = (x[i] < 0)? 0 : x[i];
    }
}

void softmax(DATA_TYPE* out, DATA_TYPE* x, int size) {
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
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < size; i++) {
        out[i] = 1 / (1 + expf(-x[i]));
    }
}
