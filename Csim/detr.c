#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"

enum {
  PROGRMAM_NAME = 0,
  CONFIG_JSON,
  WEIGHT_BIN,
  INPUT_BIN,
  OUT_BOXES_BIN,
  OUT_SCORES_BIN,
  NUM_ARGS,
};

const char* class_anme[92] = {
  "N/A","person","bicycle","car","motorcycle","airplane","bus","train",
  "truck","boat","traffic light","fire hydrant","street sign",
  "stop sign","parking meter","bench","bird","cat","dog","horse",
  "sheep","cow","elephant","bear","zebra","giraffe","hat","backpack",
  "umbrella","shoe","eye glasses","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
  "skateboard","surfboard","tennis racket","bottle","plate","wine glass",
  "cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
  "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","mirror","dining table","window","desk","toilet",
  "door","tv","laptop","mouse","remote","keyboard","cell phone","microwave",
  "oven","toaster","sink","refrigerator","blender","book","clock","vase",
  "scissors","teddy bear","hair drier","toothbrush"};

int main(int argc, char* argv[]) {
  if (argc != NUM_ARGS) {
    fprintf(stderr, "argc not correct");
    fprintf(stderr, "Usage: %s [config (.json)] [weights (.bin)] [input (.bin))] [output boxes (.bin)] [output scores (.bin)]\n", argv[0]);
    return -1;
  }

  DETR detr;
  ConvolutionTensor image = CONVTENSOR_INITIALIZER;
  OutputTensor result = OUTPUTTENSOR_INITIALIZER;

  if (init_detr(&detr, argv[CONFIG_JSON], argv[WEIGHT_BIN])) {
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

  printf("load input tensor\n");
  load_input_tensor(&image, argv[INPUT_BIN]);

  printf("forward\n");
  forward(&detr, &image, &result);


  for (int i = 0;i<100; i++)
  {
    softmax(result.classes + i * 92, result.classes + i * 92, 92);

    int max_id = 0;
    for(int c = 0;c < 92;c++){
      if(result.classes[i *92 + c] > result.classes[i *92 + max_id]){
        max_id = c;
      }
    }
    printf("%3d. [%3d], %8.3f (%.3f, %.3f, %.3f, %.3lf) ",
      i,
      max_id,
      result.classes[i *92 + max_id],
      result.bbox[i * 4],
      result.bbox[i * 4 + 1],
      result.bbox[i * 4 + 2],
      result.bbox[i * 4 + 3]
    );
    if (result.classes[i *92 + max_id] > 0.7 && max_id != 91) {
      printf(" <-- %s\n", class_anme[max_id]);
    }else{
      printf("\n");
    }
  }

  printf("save output tensor\n");
  save_output_tensor(&result, argv[OUT_BOXES_BIN], argv[OUT_SCORES_BIN]);

  printf("free\n");
  free_conv2D_tensor(&image);
  free_output_tensor(&result);
  free_detr(&detr);
  return 0;
}
