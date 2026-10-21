"""Bounded single-process background jobs with paginated results on disk."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .match_cache import canonical


class JobManager:
    def __init__(self, path=":memory:", *, workers=1, max_pending=4, max_jobs=16):
        if workers < 1 or max_pending < workers or max_jobs < max_pending:
            raise ValueError("invalid job capacity")
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="jev-match")
        self.max_pending, self.max_jobs = max_pending, max_jobs
        self.jobs = OrderedDict()
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        # Handles are process-local; remove orphan results from previous processes.
        self.db.execute("CREATE TABLE IF NOT EXISTS job_rows "
                        "(job TEXT, position INTEGER, payload TEXT, PRIMARY KEY(job,position))")
        self.db.execute("DELETE FROM job_rows")
        self.db.commit()

    def submit(self, fn):
        with self.lock:
            active = sum(j["status"] in {"queued", "running"} for j in self.jobs.values())
            if active >= self.max_pending:
                raise OverflowError("job queue full; retry after an active job completes")
            while len(self.jobs) >= self.max_jobs:
                victim = next((k for k, j in self.jobs.items() if j["status"] in {"completed", "failed"}), None)
                if victim is None:
                    raise OverflowError("job capacity reached")
                self.jobs.pop(victim)
                with self.db:
                    self.db.execute("DELETE FROM job_rows WHERE job=?", (victim,))
            handle = uuid.uuid4().hex
            self.jobs[handle] = {"job_id": handle, "status": "queued", "processed": 0,
                                 "total": None, "created_at": time.time()}
            self.pool.submit(self._run, handle, fn)
            return handle

    def _run(self, handle, fn):
        def progress(done, total):
            with self.lock:
                self.jobs[handle].update(processed=done, total=total)
        with self.lock:
            self.jobs[handle]["status"] = "running"
        try:
            result = fn(progress)
            ranking = result.pop("ranking")
            errors = result.get("errors", [])
            result["error_count"] = len(errors)
            result["errors"] = errors[:100]
            result["errors_truncated"] = len(errors) > 100
            with self.lock, self.db:
                self.db.executemany("INSERT INTO job_rows VALUES (?,?,?)",
                                    ((handle, i, canonical(row)) for i, row in enumerate(ranking)))
                self.jobs[handle].update(status="completed", result_header=result,
                                         result_count=len(ranking), finished_at=time.time())
        except Exception as exc:
            with self.lock:
                self.jobs[handle].update(status="failed", error=str(exc)[:500], finished_at=time.time())

    def status(self, handle):
        with self.lock:
            if handle not in self.jobs:
                raise KeyError("job not found on this worker")
            return {k: v for k, v in self.jobs[handle].items() if k != "result_header"}

    def results(self, handle, offset=0, limit=100):
        if offset < 0 or not 1 <= limit <= 1000:
            raise ValueError("offset must be >=0 and limit must be 1..1000")
        with self.lock:
            status = self.status(handle)
            if status["status"] != "completed":
                raise ValueError("job results are available only after completion")
            rows = [json.loads(row[0]) for row in self.db.execute(
                "SELECT payload FROM job_rows WHERE job=? AND position>=? ORDER BY position LIMIT ?",
                (handle, offset, limit))]
            return {**self.jobs[handle]["result_header"], "ranking": rows,
                    "pagination": {"offset": offset, "limit": limit, "total": status["result_count"],
                                   "next_offset": offset + len(rows) if offset + len(rows) < status["result_count"] else None}}

    def close(self):
        self.pool.shutdown(wait=True)
        with self.lock:
            self.db.close()
