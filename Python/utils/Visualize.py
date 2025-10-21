import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import argparse

# COCO classes
COCO_CLASSES = [
    "N/A",'person','bicycle','car','motorcycle','airplane','bus','train',
    'truck','boat','traffic light','fire hydrant','street sign',
    'stop sign','parking meter','bench','bird','cat','dog','horse',
    'sheep','cow','elephant','bear','zebra','giraffe','hat','backpack',
    'umbrella','shoe','eye glasses','handbag','tie','suitcase','frisbee',
    'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
    'skateboard','surfboard','tennis racket','bottle','plate','wine glass',
    'cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange',
    'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
    'potted plant','bed','mirror','dining table','window','desk','toilet',
    'door','tv','laptop','mouse','remote','keyboard','cell phone','microwave',
    'oven','toaster','sink','refrigerator','blender','book','clock','vase',
    'scissors','teddy bear','hair drier','toothbrush']

def boxes_cxcywh_to_xyxy(boxes):
    x_c, y_c, w, h = boxes
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def rescale_bboxes(boxes, size):
    img_w, img_h = size
    boxes = boxes * np.array([img_w, img_h, img_w, img_h])
    return boxes

def run_visualize_detections(args):
    image_path = args.image_path
    output_path = args.output_path
    score_path = args.score_path
    boxes_path = args.boxes_path
    threshold = args.threshold
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Load scores(prob) and boxes
    prob = np.fromfile(score_path, dtype=np.float32).reshape(-1, 92)
    boxes = np.fromfile(boxes_path, dtype=np.float32).reshape(-1, 4)

    # argmax to get the highest scoring class for each detection
    print(prob.shape, boxes.shape)

    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=15)
    valid_boxes = 0
    for p, (cx, cy, w, h), c in zip(prob, boxes.tolist(), COLORS * 100):
        cl = p.argmax()
        if p[cl] < threshold or cl == len(COCO_CLASSES):
            continue

        xmin, ymin, xmax, ymax = rescale_bboxes(boxes_cxcywh_to_xyxy([cx, cy, w, h]), image.size)
        draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(int(255 * x) for x in c), width=3)

        text = f'{COCO_CLASSES[cl]}: {p[cl]:0.2f}'
        text_size = 15
        draw.text((xmin, ymin - text_size), text, fill=tuple(int(255 * x) for x in c), font=font)
        valid_boxes += 1
        print(f"Detection {cl}: Class {COCO_CLASSES[cl]}, Score {p[cl]:.2f}, Box {[xmin, ymin, xmax, ymax]}")

    image.save(output_path)
    print(f"Total valid detections: {valid_boxes}")
    print(f"Detections visualized and saved to {output_path}")

def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Visualization utilities for object detection.", add_help=add_help)
    parser.add_argument('-i', '--image_path', required=True, help="Path to the input image.")
    parser.add_argument('-o', '--output_path', required=True, help="Path to save the output image with detections.")
    parser.add_argument('-s', '--score_path', required=True, help="Path to the detection score file.")
    parser.add_argument('-b', '--boxes_path', required=True, help="Path to the detection boxes file.")
    parser.add_argument('-t', '--threshold', type=float, default=0.7, help="Confidence threshold for displaying detections.")
    return parser

def main(args):
    run_visualize_detections(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
