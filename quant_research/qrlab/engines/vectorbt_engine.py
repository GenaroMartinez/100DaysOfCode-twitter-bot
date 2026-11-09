"""VectorBT strategy backtests on historical data only (paper, no orders).

Every parameter combination is backtested in a single vectorised pass on the
in-sample segment; the best combination is then re-scored on the untouched
out-of-sample segment so reported performance is not purely in-sample.

params:
  strategy: "ma_cross" | "zscore_reversion"
  fast_windows / slow_windows     (ma_cross)
  windows / entry_z               (zscore_reversion)
  fees (default 0.001), slippage (default 0.0005), oos_fraction (default 0.3)
  select_by: "sharpe_ratio" | "total_return" | "sortino_ratio"
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .. import config
from ..data import freq_of


def _signals(price: pd.Series, p: dict):
    if p.get("strategy", "ma_cross") == "ma_cross":
        fasts = p.get("fast_windows", [5, 10, 20])
        slows = p.get("slow_windows", [30, 50, 100])
        combos = [(f, s) for f, s in itertools.product(fasts, slows) if f < s]
        cols = pd.MultiIndex.from_tuples(combos, names=["fast", "slow"])
        ent, ex = {}, {}
        for f, s in combos:
            mf, ms = price.rolling(f).mean(), price.rolling(s).mean()
            above = mf > ms
            ent[(f, s)] = above & ~above.shift(1, fill_value=False)
            ex[(f, s)] = ~above & above.shift(1, fill_value=False)
    else:
        windows = p.get("windows", [10, 20, 40])
        zs = p.get("entry_z", [1.0, 1.5, 2.0])
        combos = list(itertools.product(windows, zs))
        cols = pd.MultiIndex.from_tuples(combos, names=["window", "entry_z"])
        ent, ex = {}, {}
        for w, z in combos:
            score = (price - price.rolling(w).mean()) / price.rolling(w).std()
            ent[(w, z)] = score < -z
            ex[(w, z)] = score > 0
    E = pd.DataFrame(ent).reindex(columns=cols) if combos else None
    X = pd.DataFrame(ex).reindex(columns=cols) if combos else None
    if E is None:
        raise ValueError("empty parameter grid")
    return E.fillna(False), X.fillna(False)


def _stats(pf) -> pd.DataFrame:
    return pd.DataFrame({"total_return": pf.total_return(), "sharpe_ratio": pf.sharpe_ratio(),
                         "sortino_ratio": pf.sortino_ratio(), "max_drawdown": pf.max_drawdown(),
                         "n_trades": pf.trades.count()})


def run(spec, y: pd.Series, season: int) -> dict:
    import vectorbt as vbt

    config.assert_research_only()
    p = spec.params
    freq = freq_of(y.index)
    split = int(len(y) * (1 - float(p.get("oos_fraction", 0.3))))
    entries, exits = _signals(y, p)
    kw = dict(fees=float(p.get("fees", 0.001)), slippage=float(p.get("slippage", 0.0005)),
              freq=freq, init_cash=float(p.get("init_cash", 10_000)))
    price_df = pd.DataFrame({c: y for c in entries.columns}, index=y.index)
    price_df.columns = entries.columns

    ins = _stats(vbt.Portfolio.from_signals(price_df.iloc[:split], entries.iloc[:split],
                                            exits.iloc[:split], **kw))
    sel = p.get("select_by", "sharpe_ratio")
    ranked = ins.replace([np.inf, -np.inf], np.nan).dropna(subset=[sel]).sort_values(sel, ascending=False)
    if ranked.empty:
        raise RuntimeError("no valid in-sample results (too few trades?)")
    best = ranked.index[0]
    oos = _stats(vbt.Portfolio.from_signals(price_df.iloc[split:], entries.iloc[split:],
                                            exits.iloc[split:], **kw)).loc[[best]]
    bh = vbt.Portfolio.from_holding(y.iloc[split:], **kw)

    names = list(entries.columns.names)
    def rec(idx, row):
        return {**dict(zip(names, idx if isinstance(idx, tuple) else (idx,))),
                **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}}

    oos_row = rec(best, oos.iloc[0])
    return {
        "metrics": {"oos_sharpe": oos_row["sharpe_ratio"] or 0.0,
                    "oos_total_return": oos_row["total_return"] or 0.0,
                    "oos_max_drawdown": oos_row["max_drawdown"] or 0.0,
                    "oos_buy_hold_return": float(bh.total_return()),
                    "in_sample_best_" + sel: float(ranked.iloc[0][sel]),
                    "n_combinations": int(len(entries.columns))},
        "best_params": {k: oos_row[k] for k in names},
        "out_of_sample": oos_row,
        "in_sample_top": [rec(i, r) for i, r in ranked.head(10).iterrows()],
        "split": {"in_sample_end": str(y.index[split - 1]), "oos_start": str(y.index[split])},
        "note": "Historical simulation only; no orders are placed.",
    }
