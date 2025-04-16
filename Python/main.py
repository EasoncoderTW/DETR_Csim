from ModelDict import *

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Export model weights to binary")

    parser.add_argument(
        "-r", "--repo_or_dir",
        required=True,
        help="Repo name (e.g. 'facebookresearch/detr:main') or local dir"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Model name (e.g. 'detr_resnet50')"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output binary file"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print("Repo or dir:", args.repo_or_dir)
    print("Model:", args.model)
    print("Output path:", args.output)

    model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True)
    MD = ModelDict(model)
    MD.summary()
    MD.save_state_dict(args.output, normal_pack)

