import torch
from .LayerInfo import *
from typing import List, Dict, Any
from torch import nn


class ModelParser:
    def __init__(self, model: nn.Module, input_shape: tuple = (1, 3, 800, 800)):
        self.model: nn.Module = model
        self.shape_params: Dict[str, ShapeParam] = {}
        self.input_shape = input_shape # Example input shape for image models

    def parse(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Parses the model to extract its architecture and parameters.
        Returns a dictionary containing the model's architecture and parameters.
        """

        def hook_fn(module: nn.Module, input, output):
            if isinstance(module, torch.nn.Conv2d):
                param = Conv2DShapeParam(
                    batch_size=input[0].shape[0],
                    input_height=input[0].shape[2],
                    input_width=input[0].shape[3],
                    filter_height=module.kernel_size[0],
                    filter_width=module.kernel_size[1],
                    output_height=output.shape[2],
                    output_width=output.shape[3],
                    input_channels=input[0].shape[1],
                    output_channels=output.shape[1],
                    stride=module.stride[0],
                    padding=module.padding[0]
                )
            elif isinstance(module, torch.nn.BatchNorm2d):
                param = BatchNorm2DShapeParam(
                    batch_size=input[0].shape[0],
                    input_height=input[0].shape[2],
                    input_width=input[0].shape[3],
                    input_channels=input[0].shape[1]
                )
            elif isinstance(module, torch.nn.MultiheadAttention):
                if len(input) == 0 or len(output) == 0:
                    return
                param = MultiHeadAttentionShapeParam(
                    batch_size=input[0].shape[1],  # assuming input is (seq_len, batch, features)
                    input_length=input[0].shape[0],
                    input_dim=input[0].shape[-1],
                    num_heads=module.num_heads,
                    head_dim=module.head_dim
                )
            elif isinstance(module, torch.nn.Linear):
                param = LinearShapeParam(
                    batch_size=input[0].shape[0],
                    in_features=input[0].shape[-1],
                    out_features=output.shape[-1]
                )
            elif isinstance(module, torch.nn.MaxPool2d):
                param = MaxPool2DShapeParam(
                    batch_size=input[0].shape[0],
                    kernel_size=module.kernel_size,
                    stride=module.stride
                )
            elif isinstance(module, torch.nn.ReLU) or isinstance(module, torch.nn.Sigmoid) or isinstance(module, torch.nn.Softmax):
                param = ActivationShapeParam(
                    batch_size=input[0].shape[0],
                    input_length=input[0].shape[2],  # assuming input is (batch, channels, height, width)
                    input_dim=input[0].shape[1]
                )
            elif isinstance(module, torch.nn.LayerNorm):
                param = LayerNormShapeParam(
                    batch_size=input[0].shape[0],
                    input_length=input[0].shape[1],  # assuming input is (batch, features)
                    input_dim=input[0].shape[-1]
                )
            else:
                if verbose:
                    print(f"Skipping module: {module.__class__.__name__}")
                return  # Skip modules that are not of interest

            if param:
                self.shape_params[module] = param

        # Register hooks for each module in the model
        hooks = []
        for module in self.model.modules():
            hooks.append(module.register_forward_hook(hook_fn))

        # Perform a dummy forward pass to trigger the hooks
        dummy_input = torch.randn(self.input_shape)
        self.model(dummy_input)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()
        #!hint<<

    def print_shape_params(self):
        """
        Prints the shape parameters for each layer in the model.
        """
        for module, param in self.shape_params.items():
            print(f"{module.__class__.__name__}: {param.to_dict()}")

def main(args):
    # Example usage
    model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True)

    parser = ModelParser(model)
    parser.parse()
    parser.print_shape_params()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Model Parser')
    parser.add_argument('-r','--repo_or_dir', type=str, required=True, help='Repository or directory of the model')
    parser.add_argument('-m','--model', type=str, required=True, help='Model name to load')

    args = parser.parse_args()

    main(args)