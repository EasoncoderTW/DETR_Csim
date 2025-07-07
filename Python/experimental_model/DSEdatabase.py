import sqlite3
from typing import List
from SramTileParameters import SramTileParameters

class DSEDatabase:
    """
    A class to manage the DSE (Design Space Experiment) database.
    It provides methods to initialize the database, check if a configuration has already been tried,
    and save results to the database.
    """

    DB_PATH = "./dse_results.db"
    DB_TABLE = "dse_results"
    DB_PRIMARY_KEY = "config_hash"

    @staticmethod
    def init_database():
        conn = sqlite3.connect(DSEDatabase.DB_PATH)
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {DSEDatabase.DB_TABLE} (
                {DSEDatabase.DB_PRIMARY_KEY} TEXT PRIMARY KEY,
                {",".join([f"{key} INTEGER" for key in SramTileParameters.__dict__.keys()])},
                status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    @staticmethod
    def already_tried(config_hash: str) -> bool:
        conn = sqlite3.connect(DSEDatabase.DB_PATH)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {DSEDatabase.DB_TABLE} WHERE config_hash=?", (config_hash,))
        result = cur.fetchone()
        conn.close()
        return result is not None

    @staticmethod
    def save_to_db(params: SramTileParameters, config_hash: str, status: str):
        d = SramTileParameters.__dict__
        d_len = len(d)
        conn = sqlite3.connect(DSEDatabase.DB_PATH)
        cur = conn.cursor()
        cur.execute(f"""
            INSERT OR REPLACE INTO {DSEDatabase.DB_TABLE} (
            {DSEDatabase.DB_PRIMARY_KEY}, {",".join([key for key in d.keys()])}, status
            ) VALUES ({",".join(["?"] * (d_len + 1))}, ?, ?)
        """, (*[value for value in d.values()], config_hash, status))
        conn.commit()
        conn.close()