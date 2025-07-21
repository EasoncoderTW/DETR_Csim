import torch
from torch import nn
import os
import argparse
import sys
import torch.jit


class ONNXParser:
    def __init__(self, model: nn.Module, output_path: str):
        self.model = model
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(torch.device('cpu'))  # Ensure the model is on CPU for ON
        self.output_path = output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _print_model(self):
        output_file = os.path.join(self.output_path, "model.txt")
        with open(output_file, "w") as f:
            f.write(str(self.model))

    def export_onnx(self, args, input_names=["input"], output_names=["output"]):
        onnx_file = os.path.join(self.output_path, "model.onnx")
        dynamo_onnx_file = os.path.join(self.output_path, "model_dynamo.onnx")
        artifacts_dir = os.path.join(self.output_path, "artifacts")

        torch.onnx.export(
                self.model,          # 模型（位置參數 1）
                args,                # 輸入參數（位置參數 2）
                f=onnx_file,         # 輸出文件（命名參數）
                opset_version=20,    # 其他命名參數
                export_params=True,
                input_names=input_names,
                output_names=output_names,
            )

        try:
            torch.onnx.export(
                self.model,          # 模型（位置參數 1）
                args,                # 輸入參數（位置參數 2）
                f=dynamo_onnx_file,         # 輸出文件（命名參數）
                opset_version=20,    # 其他命名參數
                dynamo=True,
                report=True,
                verify=True,
                profile=True,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                artifacts_dir=artifacts_dir,
            )
        except:
            print("Failed to trace the model. Please check the input arguments.")

        print(f"ONNX model exported to {onnx_file}")

    def trace_model(self, args):
        """
        Traces the model using the provided arguments and exports the trace to Markdown format (.md).

        Args:
            args: Namespace or configuration object.
        """

        output_md_path = os.path.join(self.output_path, "model_graph.md")

        try:
            # Trace the model
            traced = torch.jit.trace(func=self.model, example_inputs = [*args], strict=False)

            # Print graph for debug or markdown
            graph_str = str(traced.inlined_graph)

            # Write graph to markdown file
            with open(output_md_path, 'w') as f:
                f.write("# Traced Model Graph\n\n")
                f.write("```text\n")
                f.write(graph_str)
                f.write("\n```")

            print(f"\n\n[✓] Model traced and exported to: {output_md_path}\n")
        except Exception as e:
            print(f"[!] Failed to trace model: {e}")
            exit(1)

    def parse(self):
        self._print_model()


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="ONNX Model Exporter", add_help=add_help)
    parser.add_argument( "-r", "--repo_or_dir",type=str,default="facebookresearch/detr:main", help="Repo name (e.g. 'facebookresearch/detr:main') or local dir")
    parser.add_argument("-m", "--model", type=str, default="detr_resnet50", help="Model name (e.g. 'detr_resnet50')")
    parser.add_argument('-o', '--output', type=str, default="./output", help='Output directory for ONNX model')
    parser.add_argument('-s', '--image_size', type=int, nargs=2, default=[800, 800],
                        help='Input image size (height, width) for the model')
    return parser

class NestedTensor(object):
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

class DETR_wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image_tensor, mask_tensor):
        # Wrap the tensors into a NestedTensor object
        samples = NestedTensor(image_tensor, mask_tensor)
        out = self.model(samples)
        return out