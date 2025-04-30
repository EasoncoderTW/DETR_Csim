# DETR pure C simulation

## File structure
```
DETR
├── Csim
│   ├── README.md
│   ├── detr.c
│   ├── src
│   │   └── model.c
│   ├── Makefile
│   └── include
│       └── model.h
└── Python
```
## Data Structures
```graphviz
digraph structs_dependency {
    rankdir=LR;

    DETR [label="DETR" shape=box];
    DETRConfig [label="DETRConfig" shape=box];
    DETRWeights [label="DETRWeights" shape=box];
    TransformerConfig [label="TransformerConfig" shape=box];
    ResNet50Config [label="ResNet50Config" shape=box];
    ResNet50Weights [label="ResNet50Weights" shape=box];
    EncoderWeights [label="EncoderWeights" shape=box];
    DecoderWeights [label="DecoderWeights" shape=box];
    OutputEmbedWeights [label="OutputEmbedWeights" shape=box];
    ResidualConfig [label="ResidualConfig" shape=box];
    ConvConfig [label="ConvConfig" shape=box];
    ConvWeights [label="ConvWeights" shape=box];
    BatchNormWeights [label="BatchNormWeights" shape=box];

    // Dependencies
    DETR -> DETRConfig;
    DETR -> DETRWeights;
    DETRConfig -> TransformerConfig;
    DETRConfig -> ResNet50Config;
    DETRWeights -> ResNet50Weights;
    DETRWeights -> EncoderWeights;
    DETRWeights -> DecoderWeights;
    DETRWeights -> OutputEmbedWeights;
    ResNet50Config -> ConvConfig;
    ResNet50Config -> ResidualConfig;
    ResidualConfig -> ConvConfig;
    ResNet50Weights -> ConvWeights;
    ResNet50Weights -> BatchNormWeights;
}

```
## Call stacks
```
forward
├── malloc_encoder_run_state
├── malloc_decoder_run_state
├── forward_resnet50
│   ├── conv2D
│   │   └── malloc_conv2D_run_state
│   ├── BatchNorm
│   ├── relu
│   ├── maxpooling2D
│   │   └── malloc_conv2D_run_state
│   ├── add
│   └── free_conv2D_run_state
├── memcpy (multiple calls)
├── forward_encoder
│   ├── add
│   ├── gemm
│   ├── softmax
│   ├── relu
│   └── layernorm
├── forward_decoder
│   ├── add
│   ├── gemm
│   ├── softmax
│   ├── relu
│   ├── layernorm
│   └── memcpy
├── forward_output
│   ├── gemm
│   ├── relu
│   └── sigmoid
├── free_encoder_run_state
└── free_decoder_run_state
```

## weight binary file format
- custom format (.bin)
