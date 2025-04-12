#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

int main(int argc, char* argv[])
{
    DETR detr;
    ConvolutionRunState image;
    OutputRunState result;

    init_detr(&detr, "config.json", "weights.bin");
    print_config(&(detr.config));

    image.channels = 3;
    image.height = 800;
    image.width = 800;

    malloc_conv2D_run_state(&image);
    malloc_output_run_state(&(detr.config), &result);

    forward(&detr, &image, &result);

    free_conv2D_run_state(&image);
    free_output_run_state(&result);
    free_detr(&detr);

    // DETRConfig config;

    // if(load_config("config.json", &config))
    // {
    //     printf("load config error.");
    // }

    // print_config(&config);

    // free_config(&config);

    return 0;
}
