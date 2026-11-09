"""Runs inside the AutoGluon venv. Usage: python -m qrlab.engines.ag_worker in.json out.json

Imports only stdlib + numpy/pandas + autogluon so it is independent of the
core venv's packages.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from qrlab import config, evaluation
from qrlab.data import freq_of, infer_season_length, load_series
from qrlab.schemas import ExperimentSpec


def _tsdf(y: pd.Series):
    from autogluon.timeseries import TimeSeriesDataFrame

    df = pd.DataFrame({"item_id": "series", "timestamp": y.index, "target": y.values})
    return TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")


def main(inp: str, out: str) -> None:
    from autogluon.timeseries import TimeSeriesPredictor

    spec = ExperimentSpec.from_json(open(inp).read())
    y = load_series(spec.data, spec.domain)
    season = infer_season_length(y)
    h = spec.horizon
    freq = freq_of(y.index)
    y = y.asfreq(freq).interpolate()
    train = y.iloc[:-h]
    p = spec.params
    path = config.ARTIFACT_DIR / "autogluon" / spec.id
    predictor = TimeSeriesPredictor(
        prediction_length=h, target="target", freq=freq, path=str(path),
        eval_metric=p.get("eval_metric", "MASE"), quantile_levels=list(evaluation.QUANTILES),
        verbosity=0)
    fit_kw = dict(presets=p.get("presets", "fast_training"),
                  time_limit=int(spec.budget.timeout_minutes * 60 * 0.8),
                  num_val_windows=max(1, min(spec.cv_windows, 3)), random_seed=0)
    if p.get("hyperparameters"):
        fit_kw["hyperparameters"] = p["hyperparameters"]
    predictor.fit(_tsdf(train), **fit_kw)

    # holdout evaluation on the last h points (models were not trained on them)
    hold = predictor.predict(_tsdf(train))
    hold_mean = hold["mean"].to_numpy()
    cv = {"preds": hold_mean[None, :], "actuals": y.iloc[-h:].to_numpy()[None, :]}
    metrics = evaluation.scores(cv, y, season)

    # forecast the real future using the full history as context
    fut = predictor.predict(_tsdf(y))
    mean = fut["mean"].to_numpy()
    quantiles = {str(q): fut[str(q)].to_numpy().tolist() for q in evaluation.QUANTILES}
    sigma_h = float((fut["0.9"].iloc[-1] - fut["0.1"].iloc[-1]) / 2.5631)

    lb = predictor.leaderboard(silent=True)
    result = {
        "metrics": metrics,
        "forecast": {"timestamps": [str(t.date()) for t in fut.index.get_level_values("timestamp")],
                     "mean": mean.tolist(), "quantiles": quantiles},
        "last_observation": {"timestamp": str(y.index[-1]), "value": float(y.iloc[-1])},
        "n_observations": int(len(y)),
        "model": {"best": predictor.model_best, "path": str(path)},
        "leaderboard": json.loads(lb.to_json(orient="records")),
    }
    if spec.kalshi:
        k = spec.kalshi.__dict__
        hold_sigma = float((hold["0.9"].iloc[-1] - hold["0.1"].iloc[-1]) / 2.5631)
        result["kalshi"] = {**k,
                            "p_yes": evaluation.settlement_probability(mean[-1], sigma_h, k["strike"], k["direction"]),
                            "backtest": evaluation.kalshi_backtest(cv, np.repeat(hold_sigma, h), k["strike"], k["direction"])}
    with open(out, "w") as f:
        json.dump(result, f, default=str)


if __name__ == "__main__":
    os.environ.setdefault("AUTOGLUON_DISABLE_TELEMETRY", "1")
    main(sys.argv[1], sys.argv[2])
