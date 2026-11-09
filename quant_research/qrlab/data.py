"""Data loading for every supported domain.

Only depends on numpy/pandas/requests so it can be imported from both the core
venv and the AutoGluon venv.

Supported ``data.type`` values:
  csv / parquet  - local file; ``timestamp_col`` and ``value_col``
  inline         - ``values`` list (+ optional ``timestamps``) sent by an agent
  synthetic      - domain-flavoured generator for tests and dry runs
  coingecko      - crypto daily prices (``coin``, ``days``)
  open_meteo     - historical daily weather (``latitude``, ``longitude``,
                   ``start``, ``end``, ``variable``)
  fred           - macro / commodity series via FRED CSV (``series_id``),
                   e.g. CPIAUCSL, UNRATE, DCOILWTICO, GOLDAMGBD228NLBM
All network sources are read-only GETs.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

HTTP_TIMEOUT = 30
UA = {"User-Agent": "qrlab-research/0.1"}


def load_series(data: dict, domain: str | None = None) -> pd.Series:
    kind = data["type"]
    loader = _LOADERS.get(kind)
    if loader is None:
        raise ValueError(f"unknown data.type {kind!r}; options: {sorted(_LOADERS)}")
    s = loader(data, domain)
    s = pd.to_numeric(s, errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 20:
        raise ValueError(f"series too short ({len(s)} points)")
    s.name = data.get("value_col", s.name or "y")
    return s.astype(float)


# ---------------------------------------------------------------- local ----
def _file(data: dict, _domain) -> pd.Series:
    path = Path(data["path"]).expanduser()
    df = pd.read_parquet(path) if data["type"] == "parquet" else pd.read_csv(path)
    ts, val = data.get("timestamp_col", "timestamp"), data.get("value_col", "value")
    return pd.Series(df[val].values, index=pd.to_datetime(df[ts]), name=val)


def _inline(data: dict, _domain) -> pd.Series:
    values = data["values"]
    if "timestamps" in data:
        idx = pd.to_datetime(data["timestamps"])
    else:
        idx = pd.date_range(data.get("start", "2020-01-01"), periods=len(values),
                            freq=data.get("freq", "D"))
    return pd.Series(values, index=idx)


def _synthetic(data: dict, domain) -> pd.Series:
    n = int(data.get("n", 400))
    rng = np.random.default_rng(int(data.get("seed", 0)))
    freq = data.get("freq", "MS" if domain == "macro" else "D")
    idx = pd.date_range(data.get("start", "2020-01-01"), periods=n, freq=freq)
    t = np.arange(n)
    if domain == "crypto":  # geometric random walk with fat-ish tails
        r = rng.standard_t(4, n) * 0.025 + 0.0005
        y = 30000 * np.exp(np.cumsum(r))
    elif domain == "commodities":  # mean-reverting (OU) around 80
        y = np.empty(n); y[0] = 80
        for i in range(1, n):
            y[i] = y[i - 1] + 0.05 * (80 - y[i - 1]) + rng.normal(0, 1.2)
    elif domain == "weather":  # annual seasonality + noise (deg C)
        y = 15 + 10 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 2.5, n)
    elif domain == "macro":  # trending index with AR(1) noise
        e = np.zeros(n)
        for i in range(1, n):
            e[i] = 0.6 * e[i - 1] + rng.normal(0, 0.15)
        y = 250 + 0.3 * t + e
    else:  # kalshi: implied probability path in (0, 1)
        z = np.cumsum(rng.normal(0, 0.08, n))
        y = 1 / (1 + np.exp(-z))
    return pd.Series(y, index=idx)


# -------------------------------------------------------------- network ----
def _get(url: str, **params):
    import requests

    r = requests.get(url, params=params, headers=UA, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r


def _coingecko(data: dict, _domain) -> pd.Series:
    coin = data.get("coin", "bitcoin")
    j = _get(f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart",
             vs_currency=data.get("vs", "usd"), days=data.get("days", 365),
             interval="daily").json()
    arr = np.array(j["prices"])
    return pd.Series(arr[:, 1], index=pd.to_datetime(arr[:, 0], unit="ms"))


def _open_meteo(data: dict, _domain) -> pd.Series:
    var = data.get("variable", "temperature_2m_max")
    j = _get("https://archive-api.open-meteo.com/v1/archive",
             latitude=data["latitude"], longitude=data["longitude"],
             start_date=data["start"], end_date=data["end"], daily=var,
             timezone="UTC").json()
    return pd.Series(j["daily"][var], index=pd.to_datetime(j["daily"]["time"]))


def _fred(data: dict, _domain) -> pd.Series:
    sid = data["series_id"]
    r = _get("https://fred.stlouisfed.org/graph/fredgraph.csv", id=sid)
    df = pd.read_csv(io.StringIO(r.text))
    df = df.replace(".", np.nan)
    return pd.Series(df.iloc[:, 1].values, index=pd.to_datetime(df.iloc[:, 0]))


_LOADERS = {
    "csv": _file, "parquet": _file, "inline": _inline, "synthetic": _synthetic,
    "coingecko": _coingecko, "open_meteo": _open_meteo, "fred": _fred,
}


def freq_of(index, default: str = "D") -> str:
    """Frequency string of a DatetimeIndex (robust to pandas' offset-return change)."""
    f = pd.infer_freq(index[-30:]) if len(index) >= 3 else None
    if f is None:
        return default
    return getattr(f, "freqstr", f)


def infer_season_length(s: pd.Series) -> int:
    f = freq_of(s.index, "").upper()
    if f.startswith("D") or f.startswith("B"):
        return 7
    if f.startswith("M"):
        return 12
    if f.startswith("Q"):
        return 4
    if f.startswith("H"):
        return 24
    if f.startswith("W"):
        return 52
    return 1
