from typing import Any, Callable
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import os

COLUMNS = [
    "Operation Name",
    "ADD",
    "MUL",
    "DIV",
    "Non-Linear Operations",
    "Memory Read",
    "Memory Write"
]

class Analyzer(object):
    def __init__(self, csvfile: str, out_dir: str = ""):
        self.out_dir = out_dir
        self.csvfile = ""
        self.df = pd.DataFrame()
        if csvfile:
            self.load(csvfile)

    def load(self, csvfile: str):
        self.csvfile = csvfile
        self.df = pd.read_csv(csvfile)
        print(f"Load CSV file from: {csvfile}")

    @property
    def record_num(self):
        return len(self.df)


    def analyze(self):
        self.df['MACs'] = self.df[['ADD', 'MUL']].max(axis=1)
        self.df['FLOPs'] = self.df[['ADD','MUL','DIV','Non-Linear Operations']].sum(axis=1)
        self.df['Data movement'] = self.df[['Memory Read','Memory Write']].sum(axis=1)
        self.df['Arithmetic Intensity'] = self.df['FLOPs'] / self.df['Data movement']

        self.op_avg = self.df.groupby('Operation Name', as_index=False).mean()
        self.op_sum = self.df.groupby('Operation Name', as_index=False).sum()

    def plot_Arithmetic_Intensity(self):
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(data=self.df, x='Arithmetic Intensity', log_scale=2 ,hue='Operation Name', multiple='stack', kde=False, legend=True)
        plt.title('Arithmetic Intensity')
        plt.xlabel('FLOPs / Bytes')
        plt.ylabel('Ops')
        plt.grid(True)
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir,"Arithmetic_Intensity.png"))

    def plot_bar(self, y="FLOPs", ignore = []):
        df_nonzero = self.op_avg[self.op_avg[y] > 0]
        df_nonzero = df_nonzero[~df_nonzero['Operation Name'].isin(ignore)]
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_nonzero, x='Operation Name', y=y)
        plt.xticks(rotation=90)
        plt.title(f'{y} per Operation')
        plt.ylabel(y)
        plt.tight_layout()
        fname = y.replace(" ","_")
        plt.savefig(os.path.join(self.out_dir,f"bar_{fname}.png"))

    def save_csv(self):
        self.df.to_csv(self.csvfile.replace(".csv", "_out.csv"))
        self.op_avg.to_csv(self.csvfile.replace(".csv", "_avg.csv"))
        self.op_sum.to_csv(self.csvfile.replace(".csv", "_sum.csv"))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze and plot the statistic result.")
    parser.add_argument('-i', '--input_csv', required=True, help="Path to  statistic CSV file.")
    parser.add_argument('-o', '--out_dir', required=True, help="Path to save the analyze result and plot image.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    analyzer = Analyzer(args.input_csv, args.out_dir)

    analyzer.analyze()
    analyzer.plot_Arithmetic_Intensity()
    analyzer.plot_bar("FLOPs",["conv2D.stat",])
    analyzer.plot_bar("MACs",["conv2D.stat",])
    analyzer.plot_bar("Memory Read",["conv2D.stat",])
    analyzer.plot_bar("Memory Write",["conv2D.stat",])
    analyzer.plot_bar("Data movement",["conv2D.stat",])

    analyzer.save_csv()

if __name__ == "__main__":
    main()