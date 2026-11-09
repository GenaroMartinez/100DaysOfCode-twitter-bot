"""Small working test of each installed tool. Run with the matching venv:

  .venvs/core/bin/python scripts/verify_install.py core
  .venvs/autogluon/bin/python scripts/verify_install.py autogluon
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd


def check(name, fn, results):
    t0 = time.time()
    try:
        info = fn()
        results[name] = {"ok": True, "seconds": round(time.time() - t0, 2), **(info or {})}
    except Exception as e:  # noqa: BLE001
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                         "trace": traceback.format_exc()[-800:]}


def t_optuna():
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    st.optimize(lambda t: (t.suggest_float("x", -10, 10) - 2) ** 2, n_trials=30)
    assert abs(st.best_params["x"] - 2) < 1.5
    return {"version": optuna.__version__, "best_x": round(st.best_params["x"], 3)}


def t_ray():
    import ray
    from ray import tune

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        with tempfile.TemporaryDirectory() as d:
            res = tune.Tuner(lambda c: {"loss": (c["x"] - 3) ** 2},
                             param_space={"x": tune.grid_search([1, 2, 3, 4])},
                             tune_config=tune.TuneConfig(metric="loss", mode="min"),
                             run_config=tune.RunConfig(storage_path=d, verbose=0)).fit()
            best = res.get_best_result().config["x"]
    finally:
        ray.shutdown()
    assert best == 3
    return {"version": ray.__version__, "best_x": best}


def t_mlflow():
    import mlflow

    with tempfile.TemporaryDirectory() as d:
        mlflow.set_tracking_uri(f"sqlite:///{d}/m.db")
        mlflow.set_experiment("verify")
        with mlflow.start_run() as r:
            mlflow.log_param("p", 1)
            mlflow.log_metric("m", 0.5)
        got = mlflow.get_run(r.info.run_id).data.metrics["m"]
    assert got == 0.5
    return {"version": mlflow.__version__}


def t_vectorbt():
    import vectorbt as vbt

    rng = np.random.default_rng(0)
    price = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))),
                      index=pd.date_range("2024-01-01", periods=300, freq="D"))
    fast, slow = vbt.MA.run_combs(price, window=[5, 10, 20, 50], r=2, short_names=["fast", "slow"])
    pf = vbt.Portfolio.from_signals(price, fast.ma_crossed_above(slow), fast.ma_crossed_below(slow),
                                    fees=0.001, freq="1D")
    tr = pf.total_return()
    return {"version": vbt.__version__, "n_combos": int(tr.shape[0]),
            "best_total_return": round(float(tr.max()), 4)}


def t_autogluon():
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    import autogluon.timeseries as agts

    t = np.arange(200)
    df = pd.DataFrame({"item_id": "a", "timestamp": pd.date_range("2024-01-01", periods=200, freq="D"),
                       "target": 10 + np.sin(2 * np.pi * t / 7) + 0.01 * t})
    data = TimeSeriesDataFrame.from_data_frame(df)
    with tempfile.TemporaryDirectory() as d:
        p = TimeSeriesPredictor(prediction_length=7, path=d, verbosity=0, freq="D")
        p.fit(data, hyperparameters={"Naive": {}, "SeasonalNaive": {}, "ETS": {}}, time_limit=120)
        fc = p.predict(data)
        best = p.model_best
    assert len(fc) == 7
    return {"version": agts.__version__, "best_model": best}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "core"
    results = {"python": sys.version.split()[0], "env": which}
    tests = {"core": [("optuna", t_optuna), ("ray_tune", t_ray), ("mlflow", t_mlflow),
                      ("vectorbt", t_vectorbt)],
             "autogluon": [("autogluon_timeseries", t_autogluon)]}[which]
    for name, fn in tests:
        check(name, fn, results)
    print(json.dumps(results, indent=2))
    sys.exit(0 if all(v["ok"] for k, v in results.items() if isinstance(v, dict)) else 1)
