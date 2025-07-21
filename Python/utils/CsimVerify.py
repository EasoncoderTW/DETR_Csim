import os
import argparse
import numpy as np


def verify(csim_path, golden_path, data_type, rtol, atol):
    """
    Compare tensors from csim binary with Python golden.
    :param csim_path: Path to csim binary files.
    :param golden_path: Path to Python golden binary files.
    :param data_type: Data type of the tensors (fp32 or fp16).
    :return: Verification result.
    """
    # Placeholder for verification logic
    for csim_file in sorted(os.listdir(csim_path)):
        csim_tensor = os.path.join(csim_path, csim_file)
        golden_tensor = os.path.join(golden_path, csim_file)
        if not os.path.exists(golden_tensor):
            print(f"Golden tensor '{golden_tensor}' does not exist.")
            continue

        # Load tensors and compare
        # Placeholder for actual tensor loading and comparison logic
        print(f"Comparing {csim_file}:", end=' ')

        with open(csim_tensor, 'rb') as csim_f, open(golden_tensor, 'rb') as golden_f:
            csim_data = np.frombuffer(csim_f.read(), dtype=np.float32 if data_type == 'fp32' else np.float16)
            golden_data = np.frombuffer(golden_f.read(), dtype=np.float32 if data_type == 'fp32' else np.float16)
            # convert to numpy arrays or similar for comparison
            if csim_data.shape != golden_data.shape:
                print(f"Shape mismatch: {csim_data.shape} vs {golden_data.shape}")
                continue
            # fp32 comparison tolerance
            if not np.allclose(csim_data, golden_data, rtol=rtol, atol=atol):
                print("\033[91mMismatch found!\033[0m")
                # Optionally, print the differences
                diff = np.abs(csim_data - golden_data)
                print(f"\t\033[91mMax diff: {np.max(diff)}, Mean diff: {np.mean(diff)}\033[0m")
            else:
                print("\033[92mMatch!\033[0m")


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Verify tensors from csim binary with Python golden.", add_help=add_help)
    parser.add_argument('-c', '--csim', required=True, help="Path to csim binary files.")
    parser.add_argument('-g', '--golden', required=True, help="Path to Python golden binary files.")
    parser.add_argument('-dtype', '--data_type', required=True, choices=['fp32', 'fp16'], help="Data type of the tensors.")
    parser.add_argument('-r', '--rtol', type=float, default=1e-6, help="Relative tolerance for comparison.")
    parser.add_argument('-a', '--atol', type=float, default=1e-3, help="Absolute tolerance for comparison.")
    return parser

def run_csim_verify(args):
    csim_path = args.csim
    golden_path = args.golden

    if not os.path.exists(csim_path):
        raise FileNotFoundError(f"csim path '{csim_path}' does not exist.")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"golden path '{golden_path}' does not exist.")

    print(f"Verifying tensors:")
    print(f"\tcsim path: {csim_path}")
    print(f"\tgolden path: {golden_path}")
    print(f"\tdata type: {args.data_type}")
    print(f"\trtol: {args.rtol}")
    print(f"\tatol: {args.atol}")
    # Add logic to compare tensors here

    verify_result = verify(csim_path, golden_path, args.data_type, args.rtol, args.atol)

def main(args):
    run_csim_verify(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)