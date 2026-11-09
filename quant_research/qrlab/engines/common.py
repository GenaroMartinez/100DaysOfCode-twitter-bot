from __future__ import annotations

import numpy as np
import pandas as pd

from .. import evaluation, models


def evaluate(y: pd.Series, spec, model: str, params: dict, season: int) -> dict:
    """CV-score one (model, params) configuration."""
    fn = lambda train, h: models.forecast(model, train, h, season=season, **params)  # noqa: E731
    cv = evaluation.rolling_cv(y, spec.horizon, spec.cv_windows, fn)
    return {"cv": cv, "scores": evaluation.scores(cv, y, season)}


def finalize(y: pd.Series, spec, components: list[dict], season: int, **extra) -> dict:
    """Refit each component on all data and blend with its weight.

    components: [{"model", "params", "weight", "cv"}]
    """
    w = np.array([c["weight"] for c in components], dtype=float)
    w = w / w.sum()
    mean = sum(wi * models.forecast(c["model"], y, spec.horizon, season=season, **c["params"])
               for wi, c in zip(w, components))
    cv = {"preds": sum(wi * c["cv"]["preds"] for wi, c in zip(w, components)),
          "actuals": components[0]["cv"]["actuals"]}
    metrics = evaluation.scores(cv, y, season)
    kalshi = spec.kalshi.__dict__ if spec.kalshi else None
    return evaluation.package_result(
        y, np.asarray(mean), cv, metrics, kalshi,
        model={"components": [{"model": c["model"], "params": c["params"], "weight": float(wi)}
                              for wi, c in zip(w, components)]},
        **extra)
