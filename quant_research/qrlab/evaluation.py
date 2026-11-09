"""Rolling-origin backtesting, probabilistic outputs and Kalshi settlement odds."""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .data import freq_of

QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(q: float) -> float:
    # Acklam-free: bisection is plenty for a handful of quantiles
    lo, hi = -10.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if _norm_cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def rolling_cv(y: pd.Series, h: int, windows: int,
               fcst: Callable[[pd.Series, int], np.ndarray]) -> dict:
    """Return per-window forecasts and actuals for the last ``windows`` origins."""
    windows = max(1, min(windows, (len(y) - 20) // h))
    preds, actuals = [], []
    for w in range(windows, 0, -1):
        cut = len(y) - w * h
        train, test = y.iloc[:cut], y.iloc[cut:cut + h]
        p = np.asarray(fcst(train, h), dtype=float)
        if not np.all(np.isfinite(p)):
            raise FloatingPointError("non-finite forecast")
        preds.append(p)
        actuals.append(test.to_numpy())
    return {"preds": np.array(preds), "actuals": np.array(actuals), "train_len": len(y)}


def scores(cv: dict, y: pd.Series, season: int = 1) -> dict:
    e = cv["actuals"] - cv["preds"]
    v = y.to_numpy()
    m = season if len(v) > season + 1 else 1
    scale = np.mean(np.abs(v[m:] - v[:-m])) or 1.0
    denom = np.abs(cv["actuals"]) + np.abs(cv["preds"])
    return {
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mae": float(np.mean(np.abs(e))),
        "mase": float(np.mean(np.abs(e)) / scale),
        "smape": float(np.mean(2 * np.abs(e) / np.where(denom == 0, 1, denom)) * 100),
    }


def step_sigma(cv: dict, y: pd.Series) -> np.ndarray:
    """Per-horizon-step error std from CV, floored by a random-walk scale."""
    e = cv["actuals"] - cv["preds"]
    h = e.shape[1]
    base = np.std(np.diff(y.to_numpy())) or 1e-9
    rw = base * np.sqrt(np.arange(1, h + 1))
    if e.shape[0] >= 2:
        s = np.sqrt(np.mean(e ** 2, axis=0))
        return np.maximum(s, 0.5 * rw)
    return np.maximum(np.abs(e[0]), rw)


def quantile_forecast(mean: np.ndarray, sigma: np.ndarray) -> dict:
    return {str(q): (mean + _norm_ppf(q) * sigma).tolist() for q in QUANTILES}


def settlement_probability(mean_h: float, sigma_h: float, strike: float,
                           direction: str = "above") -> float:
    """P(YES) for a Kalshi-style binary contract settling at horizon h."""
    z = (strike - mean_h) / max(sigma_h, 1e-12)
    p_above = 1 - _norm_cdf(z)
    return float(p_above if direction == "above" else 1 - p_above)


def kalshi_backtest(cv: dict, sigma: np.ndarray, strike: float, direction: str) -> dict:
    """Brier score / log loss of settlement probabilities on CV windows."""
    probs, outcomes = [], []
    for p, a in zip(cv["preds"], cv["actuals"]):
        prob = settlement_probability(p[-1], sigma[-1], strike, direction)
        hit = a[-1] > strike if direction == "above" else a[-1] < strike
        probs.append(prob)
        outcomes.append(float(hit))
    pr, oc = np.clip(probs, 1e-6, 1 - 1e-6), np.array(outcomes)
    return {"brier": float(np.mean((pr - oc) ** 2)),
            "log_loss": float(-np.mean(oc * np.log(pr) + (1 - oc) * np.log(1 - pr))),
            "n": len(pr)}


def package_result(y: pd.Series, mean: np.ndarray, cv: dict, metrics: dict,
                   kalshi: Optional[dict] = None, **extra) -> dict:
    sigma = step_sigma(cv, y)
    freq = freq_of(y.index)
    future_idx = pd.date_range(y.index[-1], periods=len(mean) + 1, freq=freq)[1:]
    out = {
        "metrics": metrics,
        "forecast": {"timestamps": [str(t.date()) for t in future_idx],
                     "mean": [float(x) for x in mean],
                     "quantiles": quantile_forecast(mean, sigma)},
        "last_observation": {"timestamp": str(y.index[-1]), "value": float(y.iloc[-1])},
        "n_observations": int(len(y)),
    }
    if kalshi:
        out["kalshi"] = {
            **kalshi,
            "p_yes": settlement_probability(mean[-1], sigma[-1], kalshi["strike"],
                                            kalshi.get("direction", "above")),
            "backtest": kalshi_backtest(cv, sigma, kalshi["strike"],
                                        kalshi.get("direction", "above")),
        }
    out.update(extra)
    return out
