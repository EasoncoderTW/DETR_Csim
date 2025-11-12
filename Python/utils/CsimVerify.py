import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def statistic(tensor_name ,tensor, df: pd.DataFrame):
    total_elements = tensor.size
    mean = np.mean(tensor)
    std = np.std(tensor)
    max_val = np.max(tensor)
    min_val = np.min(tensor)
    zeros = np.sum(tensor == 0)
    sparsity = zeros / total_elements * 100

    new_row = pd.DataFrame({
        'Tensor': [tensor_name],
        'Mean': [mean],
        'Std': [std],
        'Max': [max_val],
        'Min': [min_val],
        'Zeros': [zeros],
        'Size': [total_elements],
        'Sparsity(%)': [sparsity]
    })
    return pd.concat([df, new_row], ignore_index=True)


def verify(csim_path, golden_path, data_type, rtol, atol, statistic_enabled=False):
    """
    Compare tensors from csim binary with Python golden.
    :param csim_path: Path to csim binary files.
    :param golden_path: Path to Python golden binary files.
    :param data_type: Data type of the tensors (fp32 or fp16).
    :return: Verification result.
    """
    if statistic_enabled:
        csim_stats_df = pd.DataFrame(columns=['Tensor', 'Mean', 'Std', 'Max', 'Min', 'Zeros', 'Size', 'Sparsity(%)'])
    else:
        csim_stats_df = None

    # Placeholder for verification logic
    min_cosine_similarity = 1.0
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

            if statistic_enabled:
                csim_stats_df = statistic(csim_file, csim_data, csim_stats_df) # df is immutable, so we don't need to return it

            # convert to numpy arrays or similar for comparison
            if csim_data.shape != golden_data.shape:
                print(f"Shape mismatch: {csim_data.shape} vs {golden_data.shape}")
                continue
            # fp32 comparison tolerance
            cos_similarity = np.dot(csim_data, golden_data) / (np.linalg.norm(csim_data) * np.linalg.norm(golden_data))
            min_cosine_similarity = min(min_cosine_similarity, cos_similarity)
            if (not np.allclose(csim_data, golden_data, rtol=rtol, atol=atol)) and cos_similarity < 0.999:
                print("\033[91mMismatch found!\033[0m",end=' ')
                # Optionally, print the differences
                diff = np.abs(csim_data - golden_data)
                #print(f"\t\033[91mMax diff: {np.max(diff)}, Mean diff: {np.mean(diff)}\033[0m")
            else:
                print("\033[92mMatch!\033[0m", end=' ')

            print(f"Cosine Similarity: {cos_similarity:.3f}")

    print(f"Minimum cosine similarity across all tensors: {min_cosine_similarity:.3f}")

    return csim_stats_df


def get_parser(add_help=True):
    parser = argparse.ArgumentParser(description="Verify tensors from csim binary with Python golden.", add_help=add_help)
    parser.add_argument('-c', '--csim', required=True, help="Path to csim binary files.")
    parser.add_argument('-g', '--golden', required=True, help="Path to Python golden binary files.")
    parser.add_argument('-dtype', '--data_type', required=True, choices=['fp32', 'fp16'], help="Data type of the tensors.")
    parser.add_argument('-r', '--rtol', type=float, default=1e-6, help="Relative tolerance for comparison.")
    parser.add_argument('-a', '--atol', type=float, default=1e-3, help="Absolute tolerance for comparison.")
    parser.add_argument('-s', '--statistic', action='store_true', help="Enable output tensor statistics.")
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
    print(f"\tstatistic: {args.statistic}")
    # Add logic to compare tensors here

    verify_result = verify(csim_path, golden_path, args.data_type, args.rtol, args.atol, args.statistic)

    if args.statistic and verify_result is not None:
        csim_path = Path(csim_path)
        stats_output_path = os.path.join(csim_path.parent.absolute(), "csim_tensor_statistics.csv")
        verify_result.to_csv(stats_output_path, index_label='Tensor')
        print(f"Tensor statistics saved to '{stats_output_path}'")

        # print each row
        print("\nTensor Statistics:")
        print(verify_result.to_string(index=False))

        print("Statistics calculation completed.")

def main(args):
    run_csim_verify(args)

if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)