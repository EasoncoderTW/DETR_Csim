from typing import Any, Callable
import torch
import struct
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict

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

class ModelInference(object):
    def __init__(self, torch_model,**kargs):
        self.model = torch_model.cpu()
        self.kargs = kargs
        self.model_input = None
        self.model_output = None
        self.raw_input = None
        self.final_output = None
        self.kargs = kargs
        # Register forward hooks for each layer if verbose is True
        if self.kargs.get('verbose', False):
            self.register_forward_hook()

    # Register forward hook to get the tensor output of each layer
    def register_forward_hook(self):
        layers = list(self.model.named_modules())
        # Register forward hooks for each layer
        for layer_name, layer in tqdm(layers, desc="Registering forward hooks", total=len(layers)):
            # Avoid registering hooks for layers without parameters (e.g., Sequential containers)
            if len(list(layer.parameters())) > 0:
                try:
                    # Register a forward hook to print the output shape
                    layer.register_forward_hook(
                        # lambda module, input, output, name=layer_name: print(f"[Layer] {name} output shape: {output.shape}")
                        lambda module, input, output, name=layer_name: self.dump_tensor(output, name)
                    )
                except:
                    continue

    def inference(self, image: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input data using the model.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            np.ndarray: Model output.
        """
        # Preprocess the input data
        self.raw_input = image # Store the raw input data
        input_data = self.preprocess(image)
        self.model_input = input_data # Store the input data
        # Perform inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        # Postprocess the output data
        self.model_output = output # Store the output data
        output = self.postprocess(output)
        return output

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
            (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox):
        # raw_input is the original image
        img_w, img_h = self.raw_input.size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def preprocess(self, image: np.ndarray, **kargs) -> torch.Tensor:
        """
        Preprocess the input data for the model.

        Args:
            input_data (np.ndarray): Input data to preprocess.

        Returns:
            torch.Tensor: Preprocessed input data.
        """
        # Example preprocessing: normalize the input data
        mean = kargs.get('mean', [0.485, 0.456, 0.406])
        std = kargs.get('std', [0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((800, 800)),
            transforms.Normalize(mean=mean, std=std)
        ])
        image_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度
        return image_tensor

    def postprocess(self, output: Any, **kargs) -> Any:
        thresh = kargs.get('threshold', 0.7)
        # keep only predictions with 0.7+ confidence
        probas = output['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > thresh

        boxes = self.rescale_bboxes(output['pred_boxes'][0, keep])
        scores = probas[keep]

        self.final_output = {
            'boxes': boxes,
            'scores': scores
        }
        return self.final_output

    def dump_tensor(self, tensor: Any, filename: str):
        """
        Save a tensor or OrderedDict to a binary file.

        Args:
            tensor (Any): The tensor or OrderedDict to save.
            filename (str): The name of the output file.
        """
        if filename == "":
            filename = "Output"

        #print(f"Saving tensor to {filename}")
        file_path = self.kargs.get('bin_output', "") + "/" + filename

        if isinstance(tensor, torch.Tensor):
            file_path = self.kargs.get('bin_output', "") + "/" + filename
            with open(file_path, "wb") as f:
                if filename.startswith("transformer."):
                    # for transformer
                    if tensor.dim() == 3:
                        tensor = tensor.permute((2,1,0))
                    elif tensor.dim() == 4:
                        tensor = tensor.permute((0,2,3,1))

                if filename.startswith("class_embed") or filename.startswith("bbox_embed"):
                    tensor = tensor[-1].permute((2,1,0))

                print("tensor", filename, "shape", tensor.shape)
                f.write(tensor.numpy().tobytes())

        elif isinstance(tensor, dict):
            print("dict", filename)
            # Handle OrderedDict by iterating through its items
            for key, value in tensor.items():
                if isinstance(value, torch.Tensor):
                    print("key", key, "shape", value.shape)
                    file_path = self.kargs.get('bin_output', "") + "/" + filename+ "." + key
                    with open(file_path, "wb") as f:
                        f.write(value.numpy().tobytes())
        elif isinstance(tensor, tuple):
            print("tuple", filename)
            # Handle tuple by iterating through its elements
            if filename.endswith("_attn"): # for attention
                file_path = self.kargs.get('bin_output', "") + "/" + filename
                tensor_0 = tensor[0].permute((2,1,0))
                print("\ttensor", filename, "shape", tensor_0.shape)
                with open(file_path, "wb") as f:
                    f.write(tensor_0.numpy().tobytes())
                file_path = self.kargs.get('bin_output', "") + "/" + filename + ".attn"
                tensor_1 = tensor[1]#.permute((2,1,0))
                print("\ttensor", filename+".attn", "shape", tensor_1.shape)
                with open(file_path, "wb") as f:
                    f.write(tensor_1.numpy().tobytes())
            else:
                for i, item in enumerate(tensor):
                    if isinstance(item, torch.Tensor):
                        print("\tshape", item.shape)
                        file_path = self.kargs.get('bin_output', "") + "/" + filename
                        with open(file_path, "wb") as f:
                            f.write(item.numpy().tobytes())
        else:
            raise TypeError(f"Unsupported type ({type(tensor)}) for dump_tensor. Must be torch.Tensor or OrderedDict.")

    def dump_input(self):
        """
        Save the input data to a binary file.
        """
        self.dump_tensor(self.model_input, "model_input.bin")
    def dump_output(self):
        """
        Save the output data to a binary file.
        """
        self.dump_tensor(self.model_output['pred_boxes'], "model_pred_boxes.bin")
        self.dump_tensor(self.model_output['pred_logits'], "model_pred_logits.bin")

    def show_final_output(self):
        """
        Print the final output data.
        """
        for box, score in zip(self.final_output['boxes'], self.final_output['scores']):
            arg_max = score.argmax()
            class_name = COCO_CLASSES[arg_max]
            score = score[arg_max]
            box = box.tolist()
            print(f"Class: {class_name} ({arg_max}), Score: {score:.2f}, Box: {box}")

#############
# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

from PIL import ImageDraw, ImageFont
def plot_and_save_results(pil_img, prob, boxes, save_path="output.png"):
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default(size=15)
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        draw.rectangle([xmin, ymin, xmax, ymax], outline=tuple(int(255 * x) for x in c), width=3)
        cl = p.argmax()
        text = f'{COCO_CLASSES[cl]}: {p[cl]:0.2f}'
        text_size = 15
        draw.text((xmin, ymin - text_size), text, fill=tuple(int(255 * x) for x in c), font=font)
    pil_img.save(save_path)

#############
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='DETR Model Inference')
    parser.add_argument(
        "-r", "--repo_or_dir",
        type=str,
        required=True,
        help="Repo name (e.g. 'facebookresearch/detr:main') or local dir"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Model name (e.g. 'detr_resnet50')"
    )

    parser.add_argument(
        "-b", "--bin_output",
        type=str,
        required=True,
        help="Path to output binary file"
    )

    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="Path to the input image"
    )

    # verbose for each layer
    parser.add_argument(
        "-v", "--verbose",
        action='store_true',
        help="Store the output of each layer"
    )


    return parser.parse_args()

if __name__ == "__main__":
    # fixme the random seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)

    # Parse command line arguments
    args = parse_args()

    # Load the image
    image = Image.open(args.image_path).convert("RGB")

    # Load the model
    model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True)

    # Create an instance of ModelInference
    model_inference = ModelInference(model, bin_output=args.bin_output, verbose=args.verbose)

    # Perform inference
    output = model_inference.inference(image)

    # Save the input and output tensors to binary files
    model_inference.dump_input()
    model_inference.dump_output()

    # Show the final output
    model_inference.show_final_output()
    # Plot and save the results
    plot_and_save_results(image, output['scores'], output['boxes'], save_path="./output/output.png")
