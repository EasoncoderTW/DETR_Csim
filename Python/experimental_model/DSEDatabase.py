import sqlite3
from typing import List
from experimental_model.SramTileParameters import SramTileParameters
from dataclasses import asdict, fields

class DSEDatabase:
    """
    A class to manage the DSE (Design Space Experiment) database.
    It provides methods to initialize the database, check if a configuration has already been tried,
    and save results to the database.
    """
    
    def __init__(self, db_path: str = "./dse_results.db"):
        self.db_path = db_path
        self.table = "dse_results"
        self.primary_key = "config_hash"
        self.param_fields = [f.name for f in fields(SramTileParameters)]

    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table} (
                {self.primary_key} TEXT PRIMARY KEY,
                {",".join([f"{key} INTEGER" for key in self.param_fields])},
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def already_tried(self, config_hash: str) -> bool:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {self.table} WHERE {self.primary_key}=?", (config_hash,))
        result = cur.fetchone()
        conn.close()
        return result is not None

    def get_all_success_hashes(self) -> List[str]:
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute(f"SELECT {self.primary_key} FROM {self.table} WHERE status = 'ok'")
        rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows]

    def save_to_db(self, params: SramTileParameters, config_hash: str, status: str):
        param_dict = asdict(params)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        columns = [self.primary_key] + list(param_dict.keys()) + ["status"]
        values = [config_hash] + list(param_dict.values()) + [status]
        placeholders = ",".join(["?"] * len(values))

        cur.execute(f"""
            INSERT OR REPLACE INTO {self.table} (
                {",".join(columns)}
            ) VALUES ({placeholders})
        """, values)

        conn.commit()
        conn.close()