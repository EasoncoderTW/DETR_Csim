import os
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.Analyzer import Analyzer
from experimental_model.DSEDatabase import DSEDatabase  

class DSEAnalyzer:
    def __init__(self,
                 db_path: str = "./dse_results.db",
                 results_dir: str = "./results",
                 dram_bandwidth: int = 16,
                 dram_latency_cycles: int = 8):
        self.db_path = db_path
        self.results_dir = results_dir
        self.dram_bandwidth = dram_bandwidth
        self.dram_latency_cycles = dram_latency_cycles

    def get_all_results(self, dse_database: DSEDatabase):
        return dse_database.get_all_success_hashes()

    def compute_latency(self, df: pd.DataFrame) -> float:
        dram_access = df['DRAM Read'].sum() + df['DRAM Write'].sum()
        compute_cycles = df[['ADD', 'MUL', 'DIV', 'Non-Linear Operations']].sum().sum()
        dram_cycles = (dram_access / self.dram_bandwidth) * self.dram_latency_cycles
        return dram_cycles + compute_cycles

    def collect_data(self, hashes: list, operator_filter: str = None) -> pd.DataFrame:
        data = []
        for chash in hashes:
            csv_path = os.path.join(self.results_dir, chash, "statistics.csv")
            if not os.path.exists(csv_path):
                continue

            analyzer = Analyzer(csvfile=csv_path)
            analyzer.analyze()
            df = analyzer.df

            if operator_filter:
                df = df[df["Operation Name"].str.contains(operator_filter, case=False, na=False)]
                if df.empty:
                    continue

            latency = self.compute_latency(df)
            sram_used = df['SRAM used'].max()

            data.append({
                "config_hash": chash,
                "latency": latency,
                "sram_used": sram_used
            })

        return pd.DataFrame(data)

    def plot_latency_vs_sram(self, df: pd.DataFrame, output_file: str = "sram_vs_latency.png"):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="latency", y="sram_used")
        plt.xlabel("Estimated Latency (cycles)")
        plt.ylabel("Max SRAM Used (bytes)")
        plt.title("SRAM Usage vs Latency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.show()

    def analyze(self, dse_database: DSEDatabase, results_dir: str = None, operator_filter: str = None):
        if results_dir:
            self.results_dir = results_dir

        hashes = self.get_all_results(dse_database)
        df = self.collect_data(hashes, operator_filter=operator_filter)
        if df.empty:
            print("[WARN] No valid results found to plot.")
        else:
            self.plot_latency_vs_sram(df)
