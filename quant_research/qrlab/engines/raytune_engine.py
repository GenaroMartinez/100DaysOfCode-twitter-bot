"""Ray Tune: parallel search across CPUs (OptunaSearch as the search algorithm).

Runs a local Ray instance capped at ``budget.max_cpus``. Point RAY_ADDRESS at
an existing cluster to distribute across machines without code changes.
"""
from __future__ import annotations

import os

from .. import config, models
from .common import evaluate, finalize


def _trainable(cfg, y=None, spec=None, season=1):
    name = cfg["model"]
    prefix = {"ets": "ets_", "theta": "theta", "arima": "arima_",
              "ridge_lags": "ridge_", "gbr_lags": "gbr_"}.get(name)
    keymap = {"ets_trend": "trend", "ets_damped": "damped", "ets_seasonal": "seasonal",
              "theta": "theta", "arima_p": "p", "arima_d": "d", "arima_q": "q",
              "ridge_n_lags": "n_lags", "ridge_alpha": "alpha", "gbr_n_lags": "n_lags",
              "gbr_n_estimators": "n_estimators", "gbr_max_depth": "max_depth",
              "gbr_lr": "learning_rate"}
    params = {keymap[k]: v for k, v in cfg.items() if prefix and k.startswith(prefix)}
    try:
        r = evaluate(y, spec, name, params, season)
        return {**r["scores"], "ok": 1}
    except Exception:  # noqa: BLE001
        return {"mase": 1e9, "rmse": 1e18, "mae": 1e18, "smape": 200.0, "ok": 0}


def _space(tune, candidates):
    none = "none"
    return {
        "model": tune.choice(candidates),
        "ets_trend": tune.choice([none, "add"]), "ets_damped": tune.choice([False, True]),
        "ets_seasonal": tune.choice([none, "add"]), "theta": tune.uniform(0.5, 3.0),
        "arima_p": tune.randint(0, 4), "arima_d": tune.randint(0, 2), "arima_q": tune.randint(0, 3),
        "ridge_n_lags": tune.randint(3, 31), "ridge_alpha": tune.loguniform(1e-3, 100),
        "gbr_n_lags": tune.randint(3, 31), "gbr_n_estimators": tune.randint(50, 301),
        "gbr_max_depth": tune.randint(2, 6), "gbr_lr": tune.loguniform(0.01, 0.3),
    }


def run(spec, y, season: int) -> dict:
    import ray
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch

    metric = spec.params.get("metric", "mase")
    candidates = list(spec.params.get("models", models.MODEL_NAMES))
    os.environ.setdefault("RAY_DEDUP_LOGS", "0")
    started_here = False
    if not ray.is_initialized():
        ray.init(address=os.environ.get("RAY_ADDRESS"),
                 num_cpus=None if os.environ.get("RAY_ADDRESS") else spec.budget.max_cpus,
                 include_dashboard=False, log_to_driver=False, ignore_reinit_error=True,
                 runtime_env={"env_vars": {"PYTHONPATH": str(config.ROOT)}})
        started_here = True
    try:
        tuner = tune.Tuner(
            tune.with_parameters(_trainable, y=y, spec=spec, season=season),
            param_space=_space(tune, candidates),
            tune_config=tune.TuneConfig(
                metric=metric, mode="min", num_samples=spec.budget.max_trials,
                search_alg=OptunaSearch(seed=0), max_concurrent_trials=spec.budget.max_cpus,
                time_budget_s=spec.budget.timeout_minutes * 60 * 0.8),
            run_config=tune.RunConfig(name=f"qrlab-{spec.id}", verbose=0,
                                      storage_path=str(config.ARTIFACT_DIR / "ray")),
        )
        grid = tuner.fit()
        df = grid.get_dataframe()
        best = grid.get_best_result(metric=metric, mode="min").config
    finally:
        if started_here:
            ray.shutdown()

    name = best["model"]
    cfg = {k: (None if v == "none" else v) for k, v in best.items()}
    params = {}
    if name == "ets":
        params = {"trend": cfg["ets_trend"], "damped": cfg["ets_damped"], "seasonal": cfg["ets_seasonal"]}
    elif name == "theta":
        params = {"theta": cfg["theta"]}
    elif name == "arima":
        params = {"p": cfg["arima_p"], "d": cfg["arima_d"], "q": cfg["arima_q"]}
    elif name == "ridge_lags":
        params = {"n_lags": cfg["ridge_n_lags"], "alpha": cfg["ridge_alpha"]}
    elif name == "gbr_lags":
        params = {"n_lags": cfg["gbr_n_lags"], "n_estimators": cfg["gbr_n_estimators"],
                  "max_depth": cfg["gbr_max_depth"], "learning_rate": cfg["gbr_lr"]}
    r = evaluate(y, spec, name, params, season)
    ok = df[df.get("ok", 1) == 1] if "ok" in df else df
    return finalize(y, spec, [{"model": name, "params": params, "weight": 1.0, "cv": r["cv"]}],
                    season, search={"n_trials": int(len(df)), "n_ok": int(len(ok)),
                                    "parallelism": spec.budget.max_cpus})
