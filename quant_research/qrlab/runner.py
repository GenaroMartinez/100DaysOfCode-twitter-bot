"""Dispatch an ExperimentSpec to its engine and track it in MLflow."""
from __future__ import annotations

import importlib
import time

from . import config
from .data import infer_season_length, load_series
from .schemas import ExperimentSpec

ENGINE_MODULES = {"baseline": "baseline", "optuna": "optuna_engine",
                  "ray_tune": "raytune_engine", "autogluon": "autogluon_engine",
                  "vectorbt": "vectorbt_engine"}


def run_spec(spec: ExperimentSpec, track: bool = True) -> dict:
    config.assert_research_only()
    config.ensure_dirs()
    y = load_series(spec.data, spec.domain)
    season = int(spec.params.get("season", infer_season_length(y)))
    engine = importlib.import_module(f"qrlab.engines.{ENGINE_MODULES[spec.engine]}")
    t0 = time.time()
    if not track:
        res = engine.run(spec, y, season)
        res["runtime_seconds"] = round(time.time() - t0, 2)
        return res
    from . import tracking

    with tracking.mlflow_run(spec) as run:
        res = engine.run(spec, y, season)
        res["runtime_seconds"] = round(time.time() - t0, 2)
        res["mlflow_run_id"] = run.info.run_id
        res["job_id"] = spec.id
        tracking.log_result(res)
    return res
