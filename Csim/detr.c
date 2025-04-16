#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

int main(int argc, char* argv[]) {
  if (argc != 3) {
    fprintf(stderr, "argc not correct");
    fprintf(stderr, "Usage: %s [config (.json)] [weights (.bin)]\n", argv[0]);
    return -1;
  }

  DETR detr;
  ConvolutionTensor image = CONVTENSOR_INITIALIZER;
  OutputTensor result = OUTPUTTENSOR_INITIALIZER;

  if (init_detr(&detr, argv[1], argv[2])) {
    return -1;
  }

  image.channels = 3;
  image.height = 800;
  image.width = 800;

  result.num_classes = 92;
  result.num_boxes = 100;

  printf("init\n");
  malloc_conv2D_tensor(&image);
  malloc_output_tensor(&result);

  printf("forward\n");
  forward(&detr, &image, &result);

  printf("free\n");
  free_conv2D_tensor(&image);
  free_output_tensor(&result);
  free_detr(&detr);
  return 0;

  // DETRConfig config;

  // if(load_config("config.json", &config))
  // {
  //     printf("load config error.");
  // }

  // print_config(&config);

  // free_config(&config);

  /*
    for (int i = 0;i<100; i++)
    {
        int max_id = 0;
        for(int c = 0;c < 92;c++){
            if(result.classes[i *92 + c] < result.classes[i *92 + max_id]){
                max_id = c;
            }
        }
        printf("[%3d], (%.3f, %.3f, %.3f, %.3lf)\n",
            max_id,
            result.bbox[i * 4],
            result.bbox[i * 4 + 1],
            result.bbox[i * 4 + 2],
            result.bbox[i * 4 + 3]
        );
    }
        */
}
