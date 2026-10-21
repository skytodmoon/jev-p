"""Bounded, version-keyed SQLite score cache; no dependency on a model runtime."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def fingerprint(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class ScoreCache:
    def __init__(self, path=":memory:", *, ttl_seconds=3600, max_entries=100000, clock=time.time):
        if ttl_seconds <= 0 or max_entries < 1:
            raise ValueError("cache TTL and capacity must be positive")
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.capacity = max_entries
        self.clock = clock
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS scores "
                        "(key TEXT PRIMARY KEY, value TEXT NOT NULL, expires REAL NOT NULL, touched REAL NOT NULL)")
        self.db.execute("CREATE INDEX IF NOT EXISTS scores_expiry ON scores(expires)")
        self.db.execute("CREATE INDEX IF NOT EXISTS scores_lru ON scores(touched)")
        self.db.commit()

    def get_many(self, keys):
        now = self.clock()
        found = {}
        unique = list(dict.fromkeys(keys))
        with self.lock, self.db:
            for offset in range(0, len(unique), 500):
                chunk = unique[offset:offset + 500]
                marks = ",".join("?" for _ in chunk)
                for key, value in self.db.execute(
                    f"SELECT key,value FROM scores WHERE key IN ({marks}) AND expires>?", chunk + [now]
                ):
                    found[key] = json.loads(value)
            self.db.executemany("UPDATE scores SET touched=? WHERE key=?", ((now, k) for k in found))
        return found

    def put_many(self, entries):
        now = self.clock()
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO scores VALUES (?,?,?,?)",
                                ((key, canonical(value), now + self.ttl, now)
                                 for key, value in entries.items()))
            self.db.execute("DELETE FROM scores WHERE expires<=?", (now,))
            excess = self.db.execute("SELECT COUNT(*) FROM scores").fetchone()[0] - self.capacity
            if excess > 0:
                self.db.execute("DELETE FROM scores WHERE key IN "
                                "(SELECT key FROM scores ORDER BY touched,key LIMIT ?)", (excess,))

    def close(self):
        with self.lock:
            self.db.close()
