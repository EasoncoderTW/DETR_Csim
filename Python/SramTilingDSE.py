from experimental_model import *
from argparse import ArgumentParser
from typing import Dict, Any
from itertools import product

class GeneralDSEProducer(DSEProducerBASE):
    def __init__(self):
        super().__init__()

    def produce(self, dse: Dict[str, Any]) -> None:
        sweep_keys = dse.get("sweep_keys", [])
        sweep_space = {
            "CONV_TILED_OUT_HEIGHT": [16, 32, 64, 128],
            "GEMM_TILED_N": [64, 128, 256],
            # Add more sweep spaces here as needed
        }

        sweep_values = [sweep_space[k] for k in sweep_keys if k in sweep_space]
        for combo in product(*sweep_values):
            kwargs = dict(zip(sweep_keys, combo))
            self.produced_dses.append(SramTileParameters(**kwargs))

class GA_DSEProducer(DSEProducerBASE):
    pass
    # TODO: Implement the GA_DSEproducer class with specific methods for generating DSE configurations using Genetic Algorithms

def run_dse(args):

    dse_database = DSEDatabase()
    dse_database.init_database()

    # DSE config defines which keys to sweep
    dse_config = {
        "sweep_keys": ["CONV_TILED_OUT_HEIGHT"]
    }
    dse_producer = GeneralDSEProducer()
    dse_producer.produce(dse_config)
    produced_params: List[SramTileParameters] = dse_producer.get_produced_dses

    # Filter out those already tried
    new_params = [p for p in produced_params if not dse_database.already_tried(p.hash)]

    # Run
    dse_executor = DSEExecutor(parameters=new_params, root_dir=args.root_dir, csim_dir=args.csim_dir, output_dir=args.output_dir, dse_database=dse_database)
    dse_executor.execute()

    dse_analyzer = DSEAnalyzer()
    dse_analyzer.analyze(dse_database, results_dir=args.results_dir, operator_filter=args.op)


def main():
    parser = ArgumentParser(description="Run DSE for SRAM tiling")
    parser.add_argument("--task_num", type=int, default=10, help="Number of tasks to run")
    parser.add_argument("--root_dir", type=str, default="../", help="Root directory for the DSE")
    parser.add_argument("--csim_dir", type=str, default="Csim", help="Directory for C simulation")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for some temporary files")
    parser.add_argument("--op", type=str, default="conv", help="Operator to filter results (e.g., 'conv', 'gemm')")
    parser.add_argument("--results_dir", type=str, default="../results", help="Directory to store statistics and results")
    # parser.add_argument("--dse_json_file", type=str, required=True, help="Path to the DSE JSON configuration file")
    args = parser.parse_args()

    run_dse(args)


if __name__ == "__main__":
    main()
    # This will run the DSE with the default parameters