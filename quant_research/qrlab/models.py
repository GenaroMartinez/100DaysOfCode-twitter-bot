"""Lightweight CPU forecasters with a uniform ``fit(y) -> predict(h)`` API.

These are what Optuna / Ray Tune search over. AutoGluon runs separately in its
own venv (see engines/autogluon_engine.py).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

MODEL_NAMES = ("naive", "seasonal_naive", "drift", "ets", "theta", "arima",
               "ridge_lags", "gbr_lags")


def _lag_matrix(y: np.ndarray, n_lags: int):
    X = np.column_stack([y[i:len(y) - n_lags + i] for i in range(n_lags)])
    return X, y[n_lags:]


def _recursive(model, y: np.ndarray, n_lags: int, h: int) -> np.ndarray:
    hist = list(y[-n_lags:])
    out = []
    for _ in range(h):
        p = float(model.predict(np.array(hist[-n_lags:])[None, :])[0])
        out.append(p)
        hist.append(p)
    return np.array(out)


def forecast(name: str, y: pd.Series, h: int, season: int = 1, **p) -> np.ndarray:
    """Return an h-step point forecast for series ``y``."""
    v = y.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "naive":
            return np.repeat(v[-1], h)
        if name == "seasonal_naive":
            m = max(int(p.get("season", season)), 1)
            if len(v) < m:
                return np.repeat(v[-1], h)
            return np.array([v[-m + (i % m)] for i in range(h)])
        if name == "drift":
            slope = (v[-1] - v[0]) / max(len(v) - 1, 1)
            return v[-1] + slope * np.arange(1, h + 1)
        if name == "ets":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing

            m = int(p.get("season", season))
            seasonal = p.get("seasonal") if m > 1 and len(v) > 2 * m else None
            fit = ExponentialSmoothing(
                v, trend=p.get("trend"), damped_trend=bool(p.get("damped", False)) and p.get("trend") is not None,
                seasonal=seasonal, seasonal_periods=m if seasonal else None,
            ).fit()
            return np.asarray(fit.forecast(h))
        if name == "theta":
            from statsmodels.tsa.forecasting.theta import ThetaModel

            m = int(p.get("season", season))
            fit = ThetaModel(v, period=m if m > 1 and len(v) > 2 * m else 1,
                             deseasonalize=m > 1 and len(v) > 2 * m).fit()
            return np.asarray(fit.forecast(h, theta=float(p.get("theta", 2.0))))
        if name == "arima":
            from statsmodels.tsa.arima.model import ARIMA

            order = (int(p.get("p", 1)), int(p.get("d", 1)), int(p.get("q", 0)))
            fit = ARIMA(v, order=order).fit()
            return np.asarray(fit.forecast(h))
        if name in ("ridge_lags", "gbr_lags"):
            n_lags = int(p.get("n_lags", 14))
            n_lags = max(2, min(n_lags, len(v) // 3))
            # model log-differences for scale-free learning when positive
            pos = bool(np.all(v > 0))
            base = np.log(v) if pos else v
            d = np.diff(base)
            X, t = _lag_matrix(d, n_lags)
            if name == "ridge_lags":
                from sklearn.linear_model import Ridge

                mdl = Ridge(alpha=float(p.get("alpha", 1.0)))
            else:
                from sklearn.ensemble import GradientBoostingRegressor

                mdl = GradientBoostingRegressor(
                    n_estimators=int(p.get("n_estimators", 100)),
                    max_depth=int(p.get("max_depth", 3)),
                    learning_rate=float(p.get("learning_rate", 0.05)), random_state=0)
            mdl.fit(X, t)
            dh = _recursive(mdl, d, n_lags, h)
            path = base[-1] + np.cumsum(dh)
            return np.exp(path) if pos else path
    raise ValueError(f"unknown model {name!r}")


def suggest_params(trial, name: str) -> dict:
    """Optuna search space per model (used by optuna and ray_tune engines)."""
    if name == "ets":
        return {"trend": trial.suggest_categorical("ets_trend", [None, "add"]),
                "damped": trial.suggest_categorical("ets_damped", [False, True]),
                "seasonal": trial.suggest_categorical("ets_seasonal", [None, "add"])}
    if name == "theta":
        return {"theta": trial.suggest_float("theta", 0.5, 3.0)}
    if name == "arima":
        return {"p": trial.suggest_int("arima_p", 0, 3), "d": trial.suggest_int("arima_d", 0, 1),
                "q": trial.suggest_int("arima_q", 0, 2)}
    if name == "ridge_lags":
        return {"n_lags": trial.suggest_int("ridge_n_lags", 3, 30),
                "alpha": trial.suggest_float("ridge_alpha", 1e-3, 100, log=True)}
    if name == "gbr_lags":
        return {"n_lags": trial.suggest_int("gbr_n_lags", 3, 30),
                "n_estimators": trial.suggest_int("gbr_n_estimators", 50, 300),
                "max_depth": trial.suggest_int("gbr_max_depth", 2, 5),
                "learning_rate": trial.suggest_float("gbr_lr", 0.01, 0.3, log=True)}
    return {}
