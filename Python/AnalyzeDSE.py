import argparse
from experimental_model.DSEAnalyzer import DSEAnalyzer
from experimental_model.DSEDatabase import DSEDatabase
import pandas as pd
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Analyze DSE results and generate SRAM vs. Latency plot.")
    parser.add_argument("-r","--results_dir", type=str, default="../results", help="Directory containing DSE results")
    parser.add_argument("-d","--db_path", type=str, default="./dse_results.db", help="Path to DSE SQLite DB")
    parser.add_argument("-op","--operator", type=str, default="Conv2D", help="Operator name to filter (e.g., Conv2D)")
    parser.add_argument("-o","--output_dir", type=str, default="./output", help="Output directory for plots")

    args = parser.parse_args()

    dse_database = DSEDatabase(args.db_path)
    dse_database.init_database()

    analyzer = DSEAnalyzer(
        database=dse_database,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        dram_bandwidth=2048//8
    )

    ## test ##
    group_bys = {
        "CONV_TILED_IN_CHANNELS",
        "CONV_TILED_OUT_HEIGHT",
        "CONV_TILED_OUT_CHANNELS",
        "CONV_TILED_OUT_WIDTH"
    }
    ## test sweep ##
    params = [
        {
            "file": "IN_CHANNELS",
            "line": "CONV_TILED_IN_CHANNELS",
            "sweep": {
                "CONV_TILED_OUT_CHANNELS":None,
                "CONV_TILED_OUT_WIDTH":None,
            },
            "fix":{
                "CONV_TILED_OUT_HEIGHT": 128
            }
        },
        {
            "file": "OUT_CHANNELS",
            "line": "CONV_TILED_OUT_CHANNELS",
            "sweep": {
                "CONV_TILED_IN_CHANNELS":None,
                "CONV_TILED_OUT_WIDTH":None,
            },
            "fix":{
                "CONV_TILED_OUT_HEIGHT": 128
            }
        },
        {
            "file": "OUT_WIDTH",
            "line": "CONV_TILED_OUT_WIDTH",
            "sweep": {
                "CONV_TILED_IN_CHANNELS" :[4, 8],
            },
            "fix":{
                "CONV_TILED_OUT_HEIGHT": 128
            }
        }
    ]

    print(args)
    analyzer.analyze(results_dir=args.results_dir, operator_filter=args.operator)
    analyzer.plot_sram_vs_latency(group_bys=list(group_bys))

    for param in params:
        analyzer.plot_Pareto_optimality(
            output_file=f"Pareto_sram_vs_latency_{param.get('file','')}.png",
            parameters=param,
            x="latency",
            y="sram_used",
            x_label="Estimated Latency (cycles)",
            y_label="Max SRAM Used (bytes)",
            title=f"SRAM Usage vs Latency ({args.operator})"
        )

    analyzer.plot_pareto_optimal_solutions()
    pareto_df = analyzer.get_pareto_optimal_solutions(
        x="latency",
        y="sram_used"
    )
    if not pareto_df.empty:
        print("[INFO] Pareto optimal solutions found:")
        print(pareto_df)
        csv_file = f"{args.output_dir}/pareto_optimal_solutions.csv"
        pareto_df.to_csv(csv_file, index=False)

    # best solution
    best_solution_set = pd.DataFrame()
    for ratio in np.logspace(-6, 2, num=17):
        best_solution = analyzer.get_best_pareto_optimal_solution(
            x="latency",
            y="sram_used",
            ratio=ratio
        )
        if best_solution is not None:
            best_solution['ratio'] = ratio
            best_solution_set = pd.concat([best_solution_set, best_solution], ignore_index=True)

    csv_file = f"{args.output_dir}/best_pareto_optimal_solutions.csv"
    best_solution_set.to_csv(csv_file, index=False)

    # print best solutions, latency, sram_used, ratio, hash (better representation)
    for index, row in best_solution_set.iterrows():
        print(
            f"Ratio: {row['ratio']:.2e}, "
            f"Latency: {row['latency']:,.2e}, "
            f"SRAM Used: {format_bytes(row['sram_used'])}, "
            f"Hash: {row['config_hash']}"
        )


def format_bytes(num_bytes):
    if num_bytes >= 1 << 20:
        return f"{num_bytes / (1 << 20):,.2f} MB"
    elif num_bytes >= 1 << 10:
        return f"{num_bytes / (1 << 10):,.2f} KB"
    else:
        return f"{num_bytes:,} B"


if __name__ == "__main__":
    main()
