from concurrent.futures import ThreadPoolExecutor, as_completed
from experimental_model.SramTileParameters import SramTileParameters
import subprocess
import shutil
import os
from typing import List
from tqdm import tqdm
import glob


# -----------------------------
# Csim Task
# -----------------------------
class Compileparams:
    def __init__(self, **kwargs):
        self.CC = kwargs.get('CC', 'gcc')
        self.SRC_DIR = kwargs.get('SRC_DIR', './src')
        self.INCLUDE_DIR = kwargs.get('INCLUDE_DIR', './include')
        self.CFLAGS = kwargs.get('CFLAGS', '-O3 -Wall')
        self.DEFS = kwargs.get('DEFS', '-DSRAM_DEFINE_EXTERN -DANALYZE')
        self.LINK_FLAGS = kwargs.get('LINK_FLAGS', '-lm')
        self.OUTPUT_DIR = kwargs.get('OUTPUT_DIR', './output')
        self.BINARY_NAME = kwargs.get('BINARY_NAME', 'output_binary')

    @property
    def command(self):
        c_files = glob.glob(os.path.join(self.SRC_DIR, "**/*.c"), recursive=True)
        if not c_files:
            raise FileNotFoundError(f"No C source files found in {self.SRC_DIR}")
        c_files_str = " ".join(c_files)
        return f"{self.CC} {self.CFLAGS} {self.DEFS} -I{self.INCLUDE_DIR} {c_files_str} -o {self.OUTPUT_DIR}/{self.BINARY_NAME} {self.LINK_FLAGS}"


class Runparams:
    def __init__(self, **kwargs):
        self.BINARY = kwargs.get('BINARY', './output_binary')
        self.CONFIG_FILE = kwargs.get('CONFIG_FILE', './config.json')
        self.WEIGHT_FILE = kwargs.get('WEIGHT_FILE', './weight.bin')
        self.INPUT_FILE = kwargs.get('INPUT_FILE', './input.bin')
        self.OUTPUT_BOXES = kwargs.get('OUTPUT_BOXES', './output_boxes.bin')
        self.OUTPUT_SCORES = kwargs.get('OUTPUT_SCORES', './output_scores.bin')
        self.LOG_FILE = kwargs.get('LOG_FILE', './log/output.log')

    @property
    def command(self):
        return f"{self.BINARY} {self.CONFIG_FILE} {self.WEIGHT_FILE} {self.INPUT_FILE} {self.OUTPUT_BOXES} {self.OUTPUT_SCORES} 1> {self.LOG_FILE} 2> {self.LOG_FILE.replace('output', 'debug')}"

class CsimTask:
    def __init__(self, task_name: str, params: SramTileParameters, root_dir: str = ".", csim_dir: str = "Csim", output_dir: str = "output"):
        self.task_name = task_name
        self.params = params
        self.root_dir = root_dir
        self.output_dir = os.path.join(self.root_dir, output_dir, task_name)
        self.debug_dir = os.path.join(self.root_dir, "debug", task_name)
        self.stat_csv = os.path.join(self.output_dir, "statistics.csv")

        self.gcc_params = Compileparams(
            SRC_DIR=os.path.join(root_dir, csim_dir),
            INCLUDE_DIR=os.path.join(root_dir, csim_dir, "include"),
            DEFS="-DSRAM_DEFINE_EXTERN -DANALYZE -DSTATISTICS_CSV_FILENAME=\\\"{}\\\" {}".format(self.stat_csv, self.params.CDEFS),
            OUTPUT_DIR=os.path.join(self.output_dir),
            BINARY_NAME="detr"
        )
        self.run_params = Runparams(
            BINARY=os.path.join(self.output_dir, "detr"),
            CONFIG_FILE=os.path.join(root_dir, "model_bundle/config.json"),
            WEIGHT_FILE=os.path.join(root_dir, "model_bundle/detr_weight.bin"),
            INPUT_FILE=os.path.join(root_dir, "model_bundle/model_input.bin"),
            OUTPUT_BOXES=os.path.join(self.output_dir, "Csim/model_output_boxes.bin"),
            OUTPUT_SCORES=os.path.join(self.output_dir, "Csim/model_output_scores.bin"),
            LOG_FILE=os.path.join(self.output_dir, "log/output.log")
        )

    def compile(self):
        # print(f"[INFO] Compiling with flags: {self.params.CDEFS}")
        print("[DEBUG] GCC Command:", self.gcc_params.command)
        os.makedirs(self.gcc_params.OUTPUT_DIR, exist_ok=True)
        result = subprocess.run(self.gcc_params.command, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Compilation failed")

    def run(self):
        os.makedirs(os.path.join(self.output_dir, "log"), exist_ok=True)
        os.makedirs(os.path.join(self.debug_dir, "log"), exist_ok=True)

        result = subprocess.run(self.run_params.command, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Execution failed")

    def archive_results(self):
        result_dir = os.path.join(self.root_dir, "results", self.params.hash)
        os.makedirs(result_dir, exist_ok=True)
        dst_csv = os.path.join(result_dir, "statistics.csv")
        if os.path.exists(self.stat_csv):
            shutil.copyfile(self.stat_csv, dst_csv)
        else:
            print(f"[WARNING] Statistics file does not exist: {self.stat_csv}")

    def cleanup(self):
        shutil.rmtree(os.path.join(self.output_dir), ignore_errors=True)
        print(f"[INFO] Cleaned up output directory: {self.output_dir}")
# -----------------------------
class DSEExecutor:
    def __init__(self, parameters: List[SramTileParameters] ,root_dir: str = ".", csim_dir: str = "Csim", output_dir: str = "output", dse_database=None):
        """
        Initialize the DSE executor with a list of parameters.

        :param parameters: List of SramTileParameters for the tasks.
        :param root_dir: Root directory for the execution.
        :param csim_dir: Directory containing Csim source files.
        :param output_dir: Directory for output files.
        """
        self.parameters = parameters
        self.root_dir = root_dir
        self.csim_dir = csim_dir
        self.output_dir = output_dir
        self.dse_database = dse_database
    @property
    def task_num(self):
        return len(self.parameters)

    def execute(self, n_threads: int = 1):

        def run_task(params: SramTileParameters):
            task_name = f"task_{params.hash}"
            csim_task = CsimTask(
                task_name=task_name,
                params=params,
                root_dir=self.root_dir,
                csim_dir=self.csim_dir,
                output_dir=self.output_dir
            )
            try:
                csim_task.compile()
                csim_task.run()
                csim_task.archive_results()
                if self.dse_database:
                    self.dse_database.save_to_db(params, params.hash, "ok")
            except RuntimeError as e:
                print(f"[ERROR] Task {task_name} failed: {e}")
                if self.dse_database:
                    self.dse_database.save_to_db(params, params.hash, "error")
            finally:
                csim_task.cleanup()

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(run_task, p) for p in self.parameters]

            for _ in tqdm(as_completed(futures), total=len(futures), desc="Executing DSE tasks", unit="task"):
                pass  # You can handle results here if needed (e.g., future.result())
