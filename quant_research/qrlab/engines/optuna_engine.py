"""Optuna: model selection + hyperparameter search, then a weight search over
the top-k configurations ("mathematical combination testing").

Studies persist in data/optuna.db so they can be resumed or inspected with
``optuna-dashboard sqlite:///data/optuna.db``.
"""
from __future__ import annotations

import numpy as np

from .. import config, evaluation, models
from .common import evaluate, finalize


def run(spec, y, season: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    metric = spec.params.get("metric", "mase")
    candidates = list(spec.params.get("models", models.MODEL_NAMES))
    storage = f"sqlite:///{config.DATA_DIR / 'optuna.db'}"
    cache: dict[int, dict] = {}

    def objective(trial):
        name = trial.suggest_categorical("model", candidates)
        params = models.suggest_params(trial, name)
        r = evaluate(y, spec, name, params, season)
        cache[trial.number] = {"model": name, "params": params, "cv": r["cv"]}
        for k, v in r["scores"].items():
            trial.set_user_attr(k, v)
        return r["scores"][metric]

    study = optuna.create_study(
        study_name=f"{spec.id}-search", storage=storage, load_if_exists=True,
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=spec.budget.max_trials,
                   timeout=spec.budget.timeout_minutes * 60 * 0.8,
                   catch=(Exception,), gc_after_trial=True)

    done = [t for t in study.trials if t.value is not None and t.number in cache]
    if not done:
        raise RuntimeError("no successful Optuna trials")
    done.sort(key=lambda t: t.value)

    # distinct top-k configs (dedupe by model+params)
    top, seen = [], set()
    for t in done:
        key = (cache[t.number]["model"], str(sorted(cache[t.number]["params"].items())))
        if key not in seen:
            seen.add(key)
            top.append(cache[t.number])
        if len(top) >= int(spec.params.get("ensemble_top_k", 3)):
            break

    components = [dict(top[0], weight=1.0)]
    ensemble_info = {"used": False}
    if len(top) > 1:
        actuals = top[0]["cv"]["actuals"]

        def w_objective(trial):
            w = np.array([trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(len(top))])
            if w.sum() == 0:
                return float("inf")
            w = w / w.sum()
            preds = sum(wi * c["cv"]["preds"] for wi, c in zip(w, top))
            return evaluation.scores({"preds": preds, "actuals": actuals}, y, season)[metric]

        ws = optuna.create_study(study_name=f"{spec.id}-weights", storage=storage,
                                 load_if_exists=True, direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=1))
        ws.enqueue_trial({f"w{i}": 1.0 if i == 0 else 0.0 for i in range(len(top))})
        ws.optimize(w_objective, n_trials=int(spec.params.get("weight_trials", 40)))
        if ws.best_value < done[0].value * 0.999:
            w = np.array([ws.best_params[f"w{i}"] for i in range(len(top))])
            components = [dict(c, weight=float(wi)) for c, wi in zip(top, w / w.sum()) if wi > 0]
            ensemble_info = {"used": True, "cv_" + metric: ws.best_value}
        ensemble_info["single_best_cv_" + metric] = done[0].value

    trials = [{"number": t.number, "value": t.value, **t.params} for t in done[:20]]
    return finalize(y, spec, components, season, ensemble=ensemble_info,
                    search={"n_trials": len(study.trials), "n_ok": len(done),
                            "top_trials": trials, "storage": storage})
