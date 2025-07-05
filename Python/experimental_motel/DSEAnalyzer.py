import os
import sys
import argparse
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Add Analyzer.py to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from Analyzer import Analyzer

DB_PATH = "./dse_results.db"
RESULTS_DIR = "../../results"

# Configurable DRAM parameters
DRAM_BANDWIDTH = 16  # bytes per cycle
DRAM_LATENCY_CYCLES = 100  # cycles


def get_all_results():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT config_hash FROM dse_results WHERE status = 'ok'")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]


def compute_latency(df: pd.DataFrame) -> float:
    dram_access = df['DRAM Read'].sum() + df['DRAM Write'].sum()
    compute_cycles = df[['ADD', 'MUL', 'DIV', 'Non-Linear Operations']].sum().sum()

    dram_cycles = (dram_access / DRAM_BANDWIDTH) * DRAM_LATENCY_CYCLES
    total_cycles = dram_cycles
    return total_cycles


def collect_data(operator_filter: str = None):
    data = []
    config_hashes = get_all_results()

    for chash in config_hashes:
        csv_path = os.path.join(RESULTS_DIR, chash, "statistics.csv")
        if not os.path.exists(csv_path):
            continue

        analyzer = Analyzer(csvfile=csv_path)
        analyzer.analyze()

        df = analyzer.df
        if operator_filter:
            df = df[df["Operation Name"].str.contains(operator_filter, case=False, na=False)]
            if df.empty:
                continue  

        latency = compute_latency(df)
        sram_used = df['SRAM used'].max()

        data.append({
            "config_hash": chash,
            "latency": latency,
            "sram_used": sram_used
        })

    return pd.DataFrame(data)




def plot_latency_vs_sram(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="latency", y="sram_used")
    plt.xlabel("Estimated Latency (cycles)")
    plt.ylabel("Max SRAM Used (bytes)")
    plt.title("SRAM Usage vs Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("sram_vs_latency.png")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default=None, help="Specify operator name (e.g., Conv2D)")
    args = parser.parse_args()

    df = collect_data(operator_filter=args.op)
    if df.empty:
        print(f"No data found for operator: {args.op}")
    else:
        plot_latency_vs_sram(df)
