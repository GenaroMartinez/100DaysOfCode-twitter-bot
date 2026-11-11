"""Central configuration for the quant research lab.

Everything is overridable through environment variables so the same code can
point at a persistent store (e.g. a mounted volume or a Postgres-backed MLflow
server) without edits.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("QR_DATA_DIR", ROOT / "data")).resolve()
JOBS_DB = Path(os.environ.get("QR_JOBS_DB", DATA_DIR / "jobs.db"))
# Shared cache for downloaded raw data (outside git; safe to share across worktrees)
RAW_DATA_DIR = Path(os.environ.get("QR_RAW_DIR", "/home/user/qrlab_data/raw")).resolve()
ARTIFACT_DIR = Path(os.environ.get("QR_ARTIFACT_DIR", DATA_DIR / "artifacts"))
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{DATA_DIR / 'mlflow.db'}"
)
MLFLOW_ARTIFACT_ROOT = os.environ.get(
    "QR_MLFLOW_ARTIFACT_ROOT", (DATA_DIR / "mlruns").as_uri()
)

CORE_PYTHON = Path(os.environ.get("QR_CORE_PYTHON", ROOT / ".venvs/core/bin/python"))
AUTOGLUON_PYTHON = Path(
    os.environ.get("QR_AUTOGLUON_PYTHON", ROOT / ".venvs/autogluon/bin/python")
)

# Cost / resource guard: a job whose estimate exceeds these limits is held in
# state "needs_approval" instead of running, until a human approves it.
MAX_CPU_MINUTES = float(os.environ.get("QR_MAX_CPU_MINUTES", "30"))
MAX_MEMORY_GB = float(os.environ.get("QR_MAX_MEMORY_GB", "8"))

# This lab is research-only. Nothing in it talks to a broker or exchange order
# endpoint, and this flag is asserted wherever results could be acted on.
LIVE_TRADING_ENABLED = False


def ensure_dirs() -> None:
    for d in (DATA_DIR, ARTIFACT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def assert_research_only() -> None:
    if LIVE_TRADING_ENABLED or os.environ.get("QR_LIVE_TRADING"):
        raise RuntimeError("Live trading is disabled in qrlab by design.")
