import sys
import os
import argparse
import torch

from utils.ModelWeight import ModelWeight, normal_pack
from utils.ModelWeight import run_model_weight
from utils.ModelWeight import get_parser as model_weight_get_parser

from utils.ModelInference import ModelInference
from utils.ModelInference import run_inference
from utils.ModelInference import get_parser as model_inference_get_parser

from utils.CsimVerify import run_csim_verify
from utils.CsimVerify import get_parser as csim_verify_get_parser

from utils.Analyzer import Analyzer
from utils.Analyzer import run_analyzer
from utils.Analyzer import get_parser as analyzer_get_parser

from utils.ONNXParser import ONNXParser, DETR_wrapper
from utils.ONNXParser import get_parser as onnx_parser_get_parser

from utils.Visualize import run_visualize_detections
from utils.Visualize import get_parser as visualize_detections_get_parser

def get_detr_tools_parser():
    parser = argparse.ArgumentParser('DETR tools', add_help=True)
    subparsers = parser.add_subparsers(dest='command')

    # parsers
    parsers = {
        'model_weight': model_weight_get_parser(add_help=False),
        'model_inference': model_inference_get_parser(add_help=False),
        'csim_verify': csim_verify_get_parser(add_help=False),
        'analyzer': analyzer_get_parser(add_help=False),
        'onnx_export': onnx_parser_get_parser(add_help=False),
        'visualize_detections': visualize_detections_get_parser(add_help=False)
    }

    for command, subparser in parsers.items():
        # remove help argument from subparsers
        subparsers.add_parser(command, parents=[subparser])

    return parser

def main(args):
    if args.command == 'model_weight':
        run_model_weight(args)
    elif args.command == 'model_inference':
        run_inference(args)
    elif args.command == 'csim_verify':
        run_csim_verify(args)
    elif args.command == 'analyzer':
        run_analyzer(args)
    elif args.command == 'visualize_detections':
        run_visualize_detections(args)
    elif args.command == 'onnx_export':
        model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True)
        onnx_parser = ONNXParser(DETR_wrapper(model=model), args.output)
        onnx_parser.parse()
        nested_tensor = (torch.randn(1, 3, *args.image_size), torch.ones(1, *args.image_size))
        onnx_parser.trace_model(args=nested_tensor)
        onnx_parser.export_onnx(args=nested_tensor, input_names=["input", "mask"], output_names=["output"])
    else:
        parser = get_detr_tools_parser()
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    parser = get_detr_tools_parser()
    args = parser.parse_args()
    main(args)