import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.Analyzer import Analyzer
from experimental_model.DSEDatabase import DSEDatabase
from tqdm import tqdm
from typing import List
import matplotlib.colors as mcolors
import numpy as np

class DSEAnalyzer:
    def __init__(self,
                 database: DSEDatabase,
                 results_dir: str = "./results",
                 output_dir: str = "./output",
                 dram_bandwidth: int = 256, # bytes per cycle
                 sram_bandwidth: int = 256000,
                 dram_latency_cycles: int = 100,
                 sram_latnecy_cycles: int = 8):
        self.dse_database = database
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.dram_bandwidth = dram_bandwidth
        self.sram_bandwidth = sram_bandwidth
        self.dram_latency_cycles = dram_latency_cycles
        self.sram_latnecy_cycles = sram_latnecy_cycles
        self.sns_palette = sns.color_palette("viridis", as_cmap=True)
        self.hashes = []
        self.df = pd.DataFrame()
        self.operator_filter = None

    def get_all_results(self):
        return self.dse_database.get_all_success_hashes()

    def compute_latency(self, df: pd.DataFrame) -> float:
        dram_access = df['DRAM Read'].sum() + df['DRAM Write'].sum()
        sram_access = df['SRAM Read'].sum() + df['SRAM Write'].sum()
        compute_cycles = df[['ADD', 'MUL', 'DIV', 'Non-Linear Operations']].sum().sum()
        dram_cycles = (dram_access / self.dram_bandwidth) * self.dram_latency_cycles
        sram_cycles = (sram_access/ self.sram_bandwidth) * self.sram_latnecy_cycles
        return dram_cycles + compute_cycles + sram_cycles

    def collect_data(self, hashes: list, operator_filter: str = None) -> pd.DataFrame:
        data = []
        for chash in tqdm(hashes, desc="Collecting data"):
            csv_path = os.path.join(self.results_dir, chash, "statistics.csv")
            if not os.path.exists(csv_path):
                continue

            analyzer = Analyzer()
            analyzer.load(csv_path, verbose=False)
            analyzer.analyze()
            df = analyzer.df

            if operator_filter:
                df = df[df["Operation Name"].str.contains(operator_filter, case=False, na=False)]
                if df.empty:
                    continue

            latency = self.compute_latency(df)
            sram_used = df['SRAM used'].max()

            sram_parameters = self.dse_database.load_from_db(chash)

            data.append({
                "config_hash": chash,
                "latency": latency,
                "sram_used": sram_used,
                **vars(sram_parameters)
            })

        return pd.DataFrame(data)

    def plot_scatter(
            self,
            output_file: str = "sram_vs_latency.png",
            group_by: str = None,
            x: str = "latency",
            y: str = "sram_used",
            x_label: str = "Estimated Latency (cycles)",
            y_label: str = "Max SRAM Used (bytes)",
            title: str = "SRAM Usage vs Latency",
        ):
        plt.figure(figsize=(10, 6))

        df = self.df.copy()

        if group_by and group_by in df.columns:
            # Check if group_by is numeric (for log scaling)
            if np.issubdtype(df[group_by].dtype, np.number):
                unique_vals = np.sort(df[group_by].unique())
                norm = mcolors.LogNorm(vmin=unique_vals.min(), vmax=unique_vals.max())
                cmap = plt.cm.viridis

                # Create log-scaled color palette
                palette = {
                    val: mcolors.to_hex(cmap(norm(val)))
                    for val in unique_vals
                }

                print(f"[INFO] Using log-scaled palette for numeric group_by: {group_by}")
            else:
                # Use default palette for categorical values
                palette = self.sns_palette
                print(f"[INFO] Using categorical palette for group_by: {group_by}")

            sns.scatterplot(data=df, x=x, y=y, hue=group_by, palette=palette)
        else:
            print(f"[WARN] Grouping by '{group_by}' is not available in the DataFrame.")
            sns.scatterplot(data=df, x=x, y=y)

        plt.xlabel(x_label, fontsize=14, weight="bold")
        plt.ylabel(y_label, fontsize=14, weight="bold")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, output_file)
        plt.savefig(output_file)
        print("plot is saved to", output_file)

    def plot_Pareto_optimality(
        self,
        output_file: str = "sram_vs_latency.png",
        parameters: dict = None,
        x: str = "latency",
        y: str = "sram_used",
        x_label: str = "Estimated Latency (cycles)",
        y_label: str = "Max SRAM Used (bytes)",
        title: str = "SRAM Usage vs Latency",
    ):
        plt.figure(figsize=(10, 7))

        line_key = parameters.get("line") if parameters else None
        # at most two parameters are swept, others are filtered
        sweep_keys = parameters.get("sweep", []) if parameters else []
        if len(sweep_keys) > 2:
            raise ValueError("At most two parameters can be swept at a time.")

        df = self.df.copy()

        for key, value in sweep_keys.items():
            if isinstance(value, list):
                # If the value is a list, filter by the first value in the list
                if key in df.columns:
                    df = df[df[key].isin(value)]

        fix_keys = parameters.get("fix", {}) if parameters else {}
        for key, value in fix_keys.items():
            if key in df.columns:
                df = df[df[key] == value]

        if df.empty:
            print("[WARN] No valid results found to plot.")
            return

        print(f"[INFO] Plotting Pareto optimality for parameters: {sweep_keys}")

        unique_vals = np.sort(df[line_key].unique())
        norm = mcolors.LogNorm(vmin=unique_vals.min(), vmax=unique_vals.max())
        cmap = sns.color_palette("rocket", as_cmap=True)

        # Create log-scaled color palette
        palette = {
            val: mcolors.to_hex(cmap(norm(val)))
            for val in unique_vals
        }

        # plot line for Pareto optimality (first sweeped parameter in a line)
        sns.lineplot(
            data=df,
            x=x,
            y=y,
            hue=line_key,
            style=list(sweep_keys.keys())[0] if sweep_keys else None,
            palette=palette,
            marker='o',
            linewidth=2
        )


        # other plot settings
        plt.xlabel(x_label, fontsize=14, weight="bold")
        plt.ylabel(y_label, fontsize=14, weight="bold")
        plt.title(title, fontsize=16, weight="bold")
        plt.grid(True)
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, output_file)
        plt.savefig(output_file)
        print("Pareto optimality plot is saved to", output_file)

    def analyze(self, results_dir: str = None, operator_filter: str = None):
        print("[INFO] Analyzing DSE ...")
        print("[INFO] Results directory:", self.results_dir)
        print("[INFO] Operator filter:", operator_filter)

        self.operator_filter = operator_filter

        if results_dir:
            self.results_dir = results_dir

        self.hashes = self.get_all_results()
        self.df = self.collect_data(self.hashes, operator_filter=operator_filter)

        if self.df.empty:
            print("[WARN] No valid results found to plot.")
            return

    def get_pareto_optimal_solutions(
            self,
            x: str = "latency",
            y: str = "sram_used"
        ) -> pd.DataFrame:
        """        Get Pareto optimal solutions from the DataFrame based on latency and SRAM used.
        Args:
            x (str): Column name for latency.
            y (str): Column name for SRAM used.
        Returns:
            pd.DataFrame: DataFrame containing Pareto optimal solutions.
        """
        if self.df.empty:
            print("[WARN] No data available to compute Pareto optimal solutions.")
            return pd.DataFrame()
        # Sort by latency and SRAM used
        sorted_df = self.df.sort_values(by=[x, y])
        pareto_front = []
        current_best = float('inf')
        for _, row in sorted_df.iterrows():
            if row[y] < current_best:
                pareto_front.append(row)
                current_best = row[y]
        pareto_df = pd.DataFrame(pareto_front)
        print(f"[INFO] Found {len(pareto_df)} Pareto optimal solutions.")
        return pareto_df

    def get_best_pareto_optimal_solution(
            self,
            x: str = "latency",
            y: str = "sram_used",
            ratio: float = 0.5
        ) -> pd.DataFrame:
        """
        Get the best Pareto optimal solution based on a ratio of latency and SRAM used.
        Args:
            x (str): Column name for latency.
            y (str): Column name for SRAM used.
            ratio (float): Ratio to determine the best solution.
        Returns:
            pd.DataFrame: DataFrame containing the best Pareto optimal solution.
        """
        pareto_df = self.get_pareto_optimal_solutions(x, y)
        if pareto_df.empty:
            print("[WARN] No Pareto optimal solutions available to find the best solution.")
            return pd.DataFrame()
        # Calculate the weighted score based on the ratio
        pareto_df['score'] = (pareto_df[x] * ratio) + (pareto_df[y] * (1 - ratio))
        # Find the row with the minimum score
        best_solution = pareto_df.loc[pareto_df['score'].idxmin()]
        print(f"[INFO] Best Pareto optimal solution found with score: {best_solution['score']}")
        return pd.DataFrame([best_solution])

    def plot_pareto_optimal_solutions(
            self,
            output_file: str = "pareto_optimal_solutions.png",
            x: str = "latency",
            y: str = "sram_used",
            x_label: str = "Estimated Latency (cycles)",
            y_label: str = "Max SRAM Used (bytes)",
            title: str = "Pareto Optimal Solutions"
        ):
        pareto_df = self.get_pareto_optimal_solutions(x, y)
        if pareto_df.empty:
            print("[WARN] No Pareto optimal solutions to plot.")
            return

        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=pareto_df, x=x, y=y, color='blue', label='Pareto Optimal Solutions')
        plt.xlabel(x_label, fontsize=14, weight="bold")
        plt.ylabel(y_label, fontsize=14, weight="bold")
        plt.title(title, fontsize=16, weight="bold")
        plt.grid(True)
        plt.tight_layout()
        output_file = os.path.join(self.output_dir, output_file)
        plt.savefig(output_file)
        print("Pareto optimal solutions plot is saved to", output_file)

    def plot_sram_vs_latency(
            self,
            group_bys: List[str] = None
        ):
        for group_by in group_bys or []:
            self.plot_scatter(
                output_file=f"sram_vs_latency_{group_by}.png",
                group_by=group_by,
                x="latency",
                y="sram_used",
                x_label="Estimated Latency (cycles)",
                y_label="Max SRAM Used (bytes)",
                title=f"SRAM Usage vs Latency ({self.operator_filter or 'All Operators'})"
            )
