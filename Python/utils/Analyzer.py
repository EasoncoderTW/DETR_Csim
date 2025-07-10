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
    "SRAM Read",
    "SRAM Write",
    "DRAM Read",
    "DRAM Write"
    "SRAM size",
    "SRAM used"
]

class Analyzer(object):
    def __init__(self, csvfile: str = "", out_dir: str = ""):
        self.out_dir = out_dir
        self.csvfile = ""
        self.df = pd.DataFrame()
        if csvfile:
            self.load(csvfile)

    def load(self, csvfile: str, verbose: bool = True):
        self.csvfile = csvfile
        self.df = pd.read_csv(csvfile)
        if verbose:
            print(f"Load CSV file from: {csvfile}, total {len(self.df)} records.")

    @property
    def head(self, n: int = 5):
        return self.df.head(n)

    @property
    def record_num(self):
        return len(self.df)

    def analyze(self):
        self.df['MACs'] = self.df[['ADD', 'MUL']].max(axis=1)
        self.df['FLOPs'] = self.df[['ADD','MUL','DIV','Non-Linear Operations']].sum(axis=1)
        self.df['DRAM access'] = self.df[['DRAM Read','DRAM Write']].sum(axis=1)
        self.df['Arithmetic Intensity'] = self.df['FLOPs'] / self.df['DRAM access']
        self.df['SRAM used rate'] = self.df['SRAM used'] / self.df['SRAM size']
        self.df['Arithmetic Intensity'] = self.df['Arithmetic Intensity'].replace([float('inf'), -float('inf')], 0)


        self.op_avg = self.df.groupby('Operation Name', as_index=False).mean()
        self.op_sum = self.df.groupby('Operation Name', as_index=False).sum()

    def plot_Arithmetic_Intensity(self):
        plt.figure(figsize=(12, 5))
        ax = sns.histplot(data=self.df, x='Arithmetic Intensity', log_scale=2 ,hue='Operation Name', multiple='stack', kde=False, legend=True)
        plt.xlabel('FLOPs / Bytes')
        plt.ylabel('Ops')
        plt.grid(True)
        plt.show()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir,"Arithmetic_Intensity.png"))

    def plot_roofline(self, x="DRAM access", y="FLOPs", hue="Operation Name", with_slope=True):
        plt.figure(figsize=(15, 9))
        if with_slope:
            # plot arithmetic intensity slope of each operation
            for _, row in self.op_avg.iterrows():
                slope = row['Arithmetic Intensity']
                plt.plot([1, 1e10], [slope, slope * 1e10], label=f"{row['Operation Name']} (AI={slope:.2f})", linestyle='--')

        # Ensure the order of operations in the plot matches the order in the DataFrame
        operation_order = self.op_avg['Operation Name']
        ax = sns.scatterplot(data=self.df, x=x, y=y, hue=hue, style=hue, s=100, hue_order=operation_order, style_order=operation_order)
        plt.xscale('log')
        plt.yscale('log')
        plt.title('Roofline Model')
        plt.xlabel('DRAM Access (Bytes)')
        plt.ylabel('FLOPs')
        plt.grid(True)
        plt.legend(title=hue)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir,"roofline_plot.png"))

    def plot_sram_used_rate(self):
        # scatter plot of SRAM used rate
        plt.figure(figsize=(12, 5))
        ax = sns.scatterplot(data=self.df, x='SRAM used rate', y='Operation Name', hue='Operation Name', style='Operation Name', s=100)
        # draw the maximum SRAM used rate line
        max_sram_used_rate = self.df['SRAM used rate'].max()
        plt.axvline(max_sram_used_rate, color='red', linestyle='--', label=f'Max SRAM Used Rate: {max_sram_used_rate*100:.2f}%')
        plt.legend(title='Operation Name', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title('SRAM Used Rate by Operation')
        plt.xlabel('SRAM Used Rate')
        plt.ylabel('Operation Name')
        plt.xlim(0, 1)  # Set x-axis limits to [0, 1]
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], [f"{i*100:.0f}%" for i in [0, 0.2, 0.4, 0.6, 0.8, 1]])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir,"sram_used_rate_plot.png"))

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

    # create sub-folder by datetime in the output directory
    sub_dir = os.path.join(args.out_dir, pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(sub_dir, exist_ok=True)
    analyzer.out_dir = sub_dir
    print(f"Output directory: {analyzer.out_dir}")

    analyzer.analyze()
    analyzer.plot_Arithmetic_Intensity()
    analyzer.plot_roofline()
    analyzer.plot_sram_used_rate()

    analyzer.save_csv()

if __name__ == "__main__":
    main()