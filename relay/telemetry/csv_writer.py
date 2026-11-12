"""
Telemetry, JSONL Spans, and results.csv Writer for The Relay.
Enforces exact columns mandated by the assignment brief:
request_id, scenario, source (real or stand-in), start_time, first_token_ms, total_ms,
status (ok, rate_limited, timeout, error), retries, tokens_in, tokens_out,
retrieval_setting, answer_grade (correct, wrong, refused_correctly, refused_wrongly)
"""

import csv
import json
import os
import threading
import time
from typing import Any, Dict, Optional


class ResultsLogger:
    CSV_COLUMNS = [
        "request_id",
        "scenario",
        "source",
        "start_time",
        "first_token_ms",
        "total_ms",
        "status",
        "retries",
        "tokens_in",
        "tokens_out",
        "retrieval_setting",
        "answer_grade"
    ]

    def __init__(self, csv_path: str = "data/results.csv", trace_path: str = "data/traces.jsonl"):
        self.csv_path = csv_path
        self.trace_path = trace_path
        self._lock = threading.Lock()
        self._init_files()

    def _init_files(self):
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_COLUMNS)

    def log_result(
        self,
        request_id: str,
        scenario: str,
        source: str,             # "real" or "stand-in"
        start_time: float,
        first_token_ms: float,
        total_ms: float,
        status: str,             # "ok", "rate_limited", "timeout", "error"
        retries: int,
        tokens_in: int,
        tokens_out: int,
        retrieval_setting: str,  # "R1", "R2", "R3", "R4"
        answer_grade: str        # "correct", "wrong", "refused_correctly", "refused_wrongly"
    ):
        with self._lock:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    request_id,
                    scenario,
                    source,
                    f"{start_time:.3f}",
                    f"{first_token_ms:.1f}",
                    f"{total_ms:.1f}",
                    status,
                    retries,
                    tokens_in,
                    tokens_out,
                    retrieval_setting,
                    answer_grade
                ])

    def log_trace(self, trace_record: Dict[str, Any]):
        with self._lock:
            with open(self.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace_record) + "\n")

