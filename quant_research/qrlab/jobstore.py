"""SQLite job queue: the hand-off point between agents and workers (stdlib only)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, name TEXT, domain TEXT, engine TEXT, status TEXT,
  submitted_by TEXT, spec_json TEXT, estimate_json TEXT, result_json TEXT,
  error TEXT, mlflow_run_id TEXT, created_at TEXT, started_at TEXT, finished_at TEXT
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status, created_at);
"""
STATUSES = ("queued", "needs_approval", "running", "done", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    config.ensure_dirs()
    con = sqlite3.connect(config.JOBS_DB, timeout=30, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def insert(spec, estimate: dict) -> str:
    status = "needs_approval" if estimate["requires_approval"] else "queued"
    with connect() as con:
        con.execute("INSERT INTO jobs (id,name,domain,engine,status,submitted_by,spec_json,"
                    "estimate_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (spec.id, spec.name, spec.domain, spec.engine, status, spec.submitted_by,
                     spec.to_json(), json.dumps(estimate), _now()))
    return status


def claim_next():
    con = connect()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1").fetchone()
        if row is None:
            con.execute("COMMIT")
            return None
        con.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (_now(), row["id"]))
        con.execute("COMMIT")
        return get(row["id"])
    finally:
        con.close()


def finish(job_id: str, result: dict | None = None, error: str | None = None) -> None:
    with connect() as con:
        con.execute("UPDATE jobs SET status=?, result_json=?, error=?, mlflow_run_id=?, finished_at=? WHERE id=?",
                    ("failed" if error else "done", json.dumps(result, default=str) if result else None,
                     error, (result or {}).get("mlflow_run_id"), _now(), job_id))


def approve(job_id: str) -> bool:
    with connect() as con:
        row = con.execute("SELECT spec_json FROM jobs WHERE id=? AND status='needs_approval'", (job_id,)).fetchone()
        if not row:
            return False
        spec = json.loads(row["spec_json"])
        spec.setdefault("budget", {})["approved"] = True
        con.execute("UPDATE jobs SET status='queued', spec_json=? WHERE id=?", (json.dumps(spec), job_id))
    return True


def get(job_id: str) -> dict | None:
    with connect() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    for k in ("spec_json", "estimate_json", "result_json"):
        d[k.removesuffix("_json")] = json.loads(d.pop(k)) if d.get(k) else None
    return d


def list_jobs(status: str | None = None, domain: str | None = None,
              submitted_by: str | None = None, limit: int = 50) -> list[dict]:
    q, args = "SELECT id,name,domain,engine,status,submitted_by,created_at,finished_at,mlflow_run_id FROM jobs WHERE 1=1", []
    for col, val in (("status", status), ("domain", domain), ("submitted_by", submitted_by)):
        if val:
            q += f" AND {col}=?"
            args.append(val)
    q += " ORDER BY created_at DESC LIMIT ?"
    with connect() as con:
        return [dict(r) for r in con.execute(q, (*args, limit))]
