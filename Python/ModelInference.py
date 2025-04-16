from typing import Any, Callable
import torch
import struct
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms

BAR_LEN = 30


class ModelInference(object):
    def __init__(self, torch_model, **kargs):
        self.model = torch_model.cpu()
        self.kargs = kargs

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess the input data for the model.

        Args:
            input_data (np.ndarray): Input data to preprocess.

        Returns:
            torch.Tensor: Preprocessed input data.
        """
        # Example preprocessing: normalize the input data
        mean = self.kargs.get('mean', [0.485, 0.456, 0.406])
        std = self.kargs.get('std', [0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((800, 800)),
            transforms.Normalize(mean=mean, std=std)
        ])
        image_tensor = transform(image).unsqueeze(0)  # 增加 batch 維度
        return image_tensor

    def inference(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform inference on the input data using the model.

        Args:
            input_data (np.ndarray): Input data for inference.

        Returns:
            np.ndarray: Model output.
        """
        # Preprocess the input data
        input_data = self.preprocess(input_data)
        # Perform inference
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        return output


import argparse

if __name__ == "__main__":
    # fixme the random seed for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)
    # randomly generate a tensor
    input_data = np.random.rand(800, 800, 3).astype(np.float32)
    # Load the model
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    # Create an instance of ModelInference
    model_inference = ModelInference(model)
    # Perform inference
    output = model_inference.inference(input_data)

    with open("intput.bin", "wb") as f:
        # Write the output to a binary file
        f.write(input_data.tobytes())
    # Print the input_data
    print("Input shape:", input_data.shape)

    output_logits = output['pred_logits'].cpu().numpy()
    output_boxes = output['pred_boxes'].cpu().numpy()
    with open("output_logits.bin", "wb") as f:
        # Write the output to a binary file
        f.write(output_logits.tobytes())
    with open("output_boxes.bin", "wb") as f:
        # Write the output to a binary file
        f.write(output_boxes.tobytes())
    # Print the output
    print("Output logits shape:", output_logits.shape)
    print("Output boxes shape:", output_boxes.shape)