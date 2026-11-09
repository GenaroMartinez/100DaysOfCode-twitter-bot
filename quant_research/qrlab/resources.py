"""Compute-capacity report and pre-run cost/resource estimates (stdlib only)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess

from . import config

# rough single-fit seconds at ~500 points, measured on a 4-vCPU container
FIT_SECONDS = {"naive": 0.001, "seasonal_naive": 0.001, "drift": 0.001, "ets": 0.08,
               "theta": 0.03, "arima": 0.25, "ridge_lags": 0.01, "gbr_lags": 0.6}
USD_PER_CPU_HOUR = float(os.environ.get("QR_USD_PER_CPU_HOUR", "0"))


def report() -> dict:
    cpus = os.cpu_count() or 1
    try:
        cpus_avail = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus_avail = cpus
    mem_total = mem_avail = None
    try:
        info = dict(l.split(":", 1) for l in open("/proc/meminfo"))
        mem_total = int(info["MemTotal"].split()[0]) / 1e6
        mem_avail = int(info["MemAvailable"].split()[0]) / 1e6
    except Exception:  # noqa: BLE001
        pass
    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001
            pass
    du = shutil.disk_usage(config.ROOT)
    return {"platform": platform.platform(), "cpus": cpus, "cpus_available": cpus_avail,
            "memory_gb_total": round(mem_total, 2) if mem_total else None,
            "memory_gb_available": round(mem_avail, 2) if mem_avail else None,
            "gpu": gpu or "none", "disk_gb_free": round(du.free / 1e9, 1),
            "guards": {"max_cpu_minutes_without_approval": config.MAX_CPU_MINUTES,
                       "max_memory_gb_without_approval": config.MAX_MEMORY_GB,
                       "live_trading_enabled": config.LIVE_TRADING_ENABLED},
            "usd_per_cpu_hour": USD_PER_CPU_HOUR}


def estimate(spec, n_points: int | None = None) -> dict:
    """Upper-bound-ish estimate; real runs are usually cheaper."""
    if n_points is None:
        d = spec.data
        n_points = (int(d.get("n", 400)) if d.get("type") == "synthetic"
                    else len(d["values"]) if d.get("type") == "inline" else 2000)
    n = n_points
    scale = max(n / 500, 0.2)
    b = spec.budget
    models = spec.params.get("models", list(FIT_SECONDS))
    avg_fit = sum(FIT_SECONDS.get(m, 0.3) for m in models) / max(len(models), 1) * scale
    per_cfg = avg_fit * (spec.cv_windows + 1)
    e = spec.engine
    if e == "baseline":
        cpu_s, wall_s, mem = sum(FIT_SECONDS.get(m, .3) for m in models) * scale * (spec.cv_windows + 1), None, 0.5
    elif e == "optuna":
        cpu_s, wall_s, mem = per_cfg * b.max_trials + 40 * 0.01, None, 0.8
    elif e == "ray_tune":
        cpu_s = per_cfg * b.max_trials + 2 * b.max_cpus
        wall_s, mem = cpu_s / b.max_cpus + 15, 1.5 + 0.4 * b.max_cpus
    elif e == "autogluon":
        # AutoGluon uses most of its time limit; assume it saturates its CPUs.
        preset = spec.params.get("presets", "fast_training")
        light = preset in ("fast_training", "chronos_tiny", "chronos_mini") or spec.params.get("hyperparameters")
        wall_s = min(b.timeout_minutes * 60, 60 + 0.05 * n) if light else b.timeout_minutes * 60
        cpu_s, mem = wall_s * b.max_cpus, 3.0
    else:  # vectorbt
        p = spec.params
        grid = (len(p.get("fast_windows", [0] * 3)) * len(p.get("slow_windows", [0] * 3))
                if p.get("strategy", "ma_cross") == "ma_cross"
                else len(p.get("windows", [0] * 3)) * len(p.get("entry_z", [0] * 3)))
        cpu_s, wall_s, mem = 5 + 0.0005 * n * grid, None, 0.6  # first call includes numba JIT
    wall_s = wall_s if wall_s is not None else cpu_s
    cpu_min = round(cpu_s / 60, 2)
    over = cpu_min > config.MAX_CPU_MINUTES or mem > config.MAX_MEMORY_GB
    return {"engine": e, "est_cpu_minutes": cpu_min, "est_wall_minutes": round(wall_s / 60, 2),
            "est_peak_memory_gb": round(mem, 1), "cpus_requested": b.max_cpus,
            "time_limit_minutes": b.timeout_minutes,
            "est_cost_usd": round(cpu_min / 60 * USD_PER_CPU_HOUR, 4),
            "cost_note": "Local container CPU; no paid APIs or cloud GPUs are used."
                         if USD_PER_CPU_HOUR == 0 else "Based on QR_USD_PER_CPU_HOUR.",
            "requires_approval": bool(over and not b.approved)}
