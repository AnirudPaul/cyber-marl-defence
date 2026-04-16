# forensic_logger.py
"""
Simple event logger: writes forensic events to SQLite and (optionally) to CSV.

Usage:
    logger = ForensicLogger('forensics_log.db')
    logger.log_event(features=..., decision='allow', severity=0.12, xai=json_dict, extra={...})

Table schema:
  events(id INTEGER PRIMARY KEY, ts REAL, decision TEXT, severity REAL, features TEXT, xai TEXT, extra TEXT)

The module uses JSON-encoded strings to store nested structures.
"""

import sqlite3
import json
import csv
import os
import time
from typing import Any, Dict, Optional

class ForensicLogger:
    def __init__(self, db_path: str = "forensics_log.db", csv_export_path: Optional[str] = None):
        self.db_path = db_path
        self.csv_export_path = csv_export_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_table()
        if csv_export_path:
            os.makedirs(os.path.dirname(csv_export_path) or ".", exist_ok=True)

    def _ensure_table(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                decision TEXT,
                severity REAL,
                features TEXT,
                xai TEXT,
                extra TEXT
            )
            """
        )
        self.conn.commit()

    def log_event(self, features: Any, decision: str, severity: float = 0.0, xai: Optional[Dict] = None, extra: Optional[Dict] = None):
        ts = time.time()
        f_txt = json.dumps(features)
        xai_txt = json.dumps(xai or {})
        extra_txt = json.dumps(extra or {})
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events (ts, decision, severity, features, xai, extra) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, decision, severity, f_txt, xai_txt, extra_txt),
        )
        self.conn.commit()
        if self.csv_export_path:
            self._append_csv_row(ts, decision, severity, f_txt, xai_txt, extra_txt)

    def _append_csv_row(self, ts, decision, severity, features, xai, extra):
        header = ["ts", "decision", "severity", "features", "xai", "extra"]
        write_header = not os.path.exists(self.csv_export_path)
        with open(self.csv_export_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if write_header:
                w.writerow(header)
            w.writerow([ts, decision, severity, features, xai, extra])

    def export_all_to_csv(self, out_path: str):
        cur = self.conn.cursor()
        cur.execute("SELECT ts, decision, severity, features, xai, extra FROM events ORDER BY ts ASC")
        rows = cur.fetchall()
        header = ["ts", "decision", "severity", "features", "xai", "extra"]
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

    def close(self):
        self.conn.close()
