"""Thin MLflow wrapper so every engine logs runs the same way."""
from __future__ import annotations

import json
import os
from contextlib import contextmanager

from . import config


def _flat(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


@contextmanager
def mlflow_run(spec, run_name: str | None = None, nested: bool = False):
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    config.ensure_dirs()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp_name = f"qrlab/{spec.domain}"
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(exp_name)
    with mlflow.start_run(run_name=run_name or f"{spec.name}:{spec.engine}",
                          nested=nested) as run:
        mlflow.set_tags({"qrlab.job_id": spec.id, "qrlab.engine": spec.engine,
                         "qrlab.domain": spec.domain, "qrlab.submitted_by": spec.submitted_by,
                         **{f"user.{k}": str(v) for k, v in spec.tags.items()}})
        if not nested:
            mlflow.log_params({k: str(v)[:500] for k, v in _flat({
                "horizon": spec.horizon, "cv_windows": spec.cv_windows,
                "data": spec.data if spec.data.get("type") != "inline" else {"type": "inline"},
                "params": spec.params}).items()})
        yield run


def log_result(result: dict) -> None:
    import mlflow

    metrics = {k: float(v) for k, v in _flat(result.get("metrics", {})).items()
               if isinstance(v, (int, float))}
    k = result.get("kalshi")
    if k:
        metrics["kalshi.p_yes"] = k["p_yes"]
        metrics.update({f"kalshi.{m}": v for m, v in k["backtest"].items()})
    if metrics:
        mlflow.log_metrics(metrics)
    mlflow.log_text(json.dumps(result, indent=2, default=str), "result.json")
