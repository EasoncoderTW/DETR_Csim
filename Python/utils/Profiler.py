import argparse

import torch
from torch.profiler import profile, record_function, ProfilerActivity

def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description='Model Parser')
    parser.add_argument('-r','--repo_or_dir', type=str, required=True, help='Repository or directory of the model')
    parser.add_argument('-m','--model', type=str, required=True, help='Model name to load')
    parser.add_argument('-o', '--output', type=str, default='./output', help='Output dir to save model information')
    parser.add_argument('-i', '--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('-s', '--in_size', type=int, default=800, help='Input size (height and width)')
    parser.add_argument('-d', '--device', type=str, default='cpu', help='Device to run the model on (cpu or cuda)')
    parser.add_argument('-p', '--profile_memory', type=bool, default=True, help='Profile memory usage')


    return parser.parse_args()




def main():
    args = parse_args()
    model = torch.hub.load(args.repo_or_dir, args.model, pretrained=True).to(args.device)

    dummy_input = torch.randn(1, args.in_channels, args.in_size, args.in_size)

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with record_function("model_inference"):
            model(dummy_input)

    # Display the profile results
    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total", row_limit=10, max_shapes_column_width=30
        )
    )

    print(
        prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_memory_usage", row_limit=10, max_shapes_column_width=30
        )
    )

    prof.export_chrome_trace(f"{args.output}/trace.json")
    prof.export_memory_timeline(f"{args.output}/memory_timeline.html")
if __name__ == "__main__":
    main()
