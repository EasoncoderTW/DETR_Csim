import os
import sqlite3
import hashlib
import subprocess
import pandas as pd
import shutil
from dataclasses import dataclass

DB_PATH = "./dse_results.db"
RESULTS_DIR = "results"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------
# Parameter Definition
# -----------------------------
@dataclass
class SramTileParameters:
    CONV_TILED_OUT_HEIGHT: int
    SRAM_DEFAULT_SIZE: int = 8388608 # 8 MB
    CONV_TILED_OUT_WIDTH: int = 32
    CONV_TILED_OUT_CHANNELS: int = 128
    CONV_TILED_IN_CHANNELS: int = 64
    MAXPOOL_TILED_OUT_HEIGHT: int = 32
    MAXPOOL_TILED_OUT_WIDTH: int = 32
    MAXPOOL_TILED_CHANNELS: int = 128
    LAYERNORM_TILED_N: int = 1024
    BATCHNORM_TILED_CHANNELS: int = 64
    BATCHNORM_TILED_HEIGHT: int = 64
    BATCHNORM_TILED_WIDTH: int = 64
    GEMM_TILED_OUT_DIM: int = 512
    GEMM_TILED_IN_DIM: int = 512
    GEMM_TILED_N: int = 128
    MULTIHEAD_ATTENTION_TILED_Q_LEN: int = 32
    MULTIHEAD_ATTENTION_TILED_KV_LEN: int = 32
    ADD_TILED_SIZE: int = 4096
    SIGMOID_TILED_SIZE: int = 4096

    def to_macro_flags(self):
        return [f"-D{key}={value}" for key, value in self.__dict__.items()]

# -----------------------------
# Database Utilities
# -----------------------------
def init_database():
    db_dir = os.path.dirname(DB_PATH)
    os.makedirs(db_dir, exist_ok=True)  

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dse_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_hash TEXT UNIQUE,
            CONV_TILED_OUT_HEIGHT INTEGER,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()


def get_config_hash(params: SramTileParameters) -> str:
    items = sorted(params.__dict__.items())
    config_str = "_".join(f"{k}={v}" for k, v in items)
    return hashlib.md5(config_str.encode()).hexdigest()

def already_tried(config_hash: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM dse_results WHERE config_hash=?", (config_hash,))
    result = cur.fetchone()
    conn.close()
    return result is not None

def save_to_db(params: SramTileParameters, config_hash: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO dse_results (
            config_hash, CONV_TILED_OUT_HEIGHT, status
        ) VALUES (?, ?, ?)
    """, (config_hash, params.CONV_TILED_OUT_HEIGHT, status))
    conn.commit()
    conn.close()

# -----------------------------
# Csim Task
# -----------------------------
class CsimTask:
    def __init__(self, task_name: str, params: SramTileParameters, config_hash: str):
        self.task_name = task_name
        self.params = params
        self.config_hash = config_hash
        self.stat_csv = os.path.join(ROOT_DIR, "output", "statistics.csv")

    def compile_csim(self):
        macro_flags = " ".join(self.params.to_macro_flags())
        print(f"[INFO] Compiling with flags: {macro_flags}")
        csim_dir = os.path.join(ROOT_DIR, "Csim")
        output_dir = os.path.join(ROOT_DIR, "output", "Csim")
        os.makedirs(output_dir, exist_ok=True)

        src_dir = os.path.join(csim_dir, "src")
        src_files = f"{os.path.join(csim_dir, 'detr.c')} " + " ".join(
            os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".c")
        )
        include = f"-I{os.path.join(csim_dir, 'include')}"
        binary = os.path.join(ROOT_DIR, "detr")

        cmd = (
            f"gcc -O3 -Wall {include} -DSRAM_DEFINE_EXTERN {macro_flags} -DANALYZE "
            f"-DSTATISTICS_CSV_FILENAME=\\\"{self.stat_csv}\\\" "
            f"{src_files} -o {binary} -lm"
        )
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Compilation failed")

    def run(self):
        os.makedirs(os.path.join(ROOT_DIR, "output", "Csim", "debug"), exist_ok=True)
        os.makedirs(os.path.join(ROOT_DIR, "log"), exist_ok=True)

        cmd = (
            f"{os.path.join(ROOT_DIR, 'detr')} "
            f"{os.path.join(ROOT_DIR, 'model_bundle/config.json')} "
            f"{os.path.join(ROOT_DIR, 'model_bundle/detr_weight.bin')} "
            f"{os.path.join(ROOT_DIR, 'model_bundle/model_input.bin')} "
            f"{os.path.join(ROOT_DIR, 'output/Csim/model_output_boxes.bin')} "
            f"{os.path.join(ROOT_DIR, 'output/Csim/model_output_scores.bin')} "
            f"1> {os.path.join(ROOT_DIR, 'log/output.log')} 2> {os.path.join(ROOT_DIR, 'log/debug.log')}"
        )
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            raise RuntimeError("Execution failed")

    def archive_results(self):
        result_dir = os.path.join(CURRENT_DIR, RESULTS_DIR, self.config_hash)
        os.makedirs(result_dir, exist_ok=True)
        dst_csv = os.path.join(result_dir, "statistics.csv")
        if os.path.exists(self.stat_csv):
            shutil.copyfile(self.stat_csv, dst_csv)

    def cleanup(self):
        binary_path = os.path.join(ROOT_DIR, "detr")
        if os.path.exists(binary_path):
            os.remove(binary_path)
        shutil.rmtree(os.path.join(ROOT_DIR, "output", "Csim"), ignore_errors=True)

# -----------------------------
# DSE Producer (vary only one param)
# -----------------------------
def dse_producer():
    param_list = []
    for h in [16, 32, 64, 128]:
        param = SramTileParameters(CONV_TILED_OUT_HEIGHT=h)
        param_list.append(param)
    return param_list

# -----------------------------
# Main Flow
# -----------------------------
def run_dse():
    init_database()
    for param in dse_producer():
        config_hash = get_config_hash(param)
        if already_tried(config_hash):
            print(f"[SKIP] Already tried: {config_hash}")
            continue

        task = CsimTask(f"task_{config_hash}", param, config_hash)
        try:
            task.compile_csim()
            task.run()
            task.archive_results()
            save_to_db(param, config_hash, status="ok")
        except Exception as e:
            print(f"[ERROR] {config_hash}: {e}")
            save_to_db(param, config_hash, status="error")
        finally:
            task.cleanup()

if __name__ == "__main__":
    run_dse()
