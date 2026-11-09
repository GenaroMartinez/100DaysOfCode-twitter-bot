import json
import os

import pytest

from qrlab.client import ResearchClient
from qrlab.runner import run_spec
from qrlab.schemas import ExperimentSpec


def spec(**kw):
    base = {"name": "t", "domain": "crypto", "engine": "baseline",
            "data": {"type": "synthetic", "n": 250, "seed": 1}, "horizon": 5, "cv_windows": 3,
            "budget": {"max_trials": 6, "max_cpus": 2, "timeout_minutes": 2}}
    base.update(kw)
    return ExperimentSpec.from_dict(base)


@pytest.mark.parametrize("domain", ["crypto", "commodities", "weather", "macro", "kalshi"])
def test_baseline_all_domains(domain):
    r = run_spec(spec(domain=domain), track=False)
    assert len(r["forecast"]["mean"]) == 5 and r["metrics"]["mase"] > 0


def test_optuna_with_kalshi_and_mlflow():
    s = spec(domain="kalshi", engine="optuna", kalshi={"strike": 0.5, "direction": "above"},
             params={"models": ["naive", "drift", "ets", "ridge_lags"]})
    r = run_spec(s)
    assert 0 <= r["kalshi"]["p_yes"] <= 1 and r["mlflow_run_id"]
    assert "brier" in r["kalshi"]["backtest"]


def test_raytune():
    r = run_spec(spec(engine="ray_tune", params={"models": ["naive", "drift", "theta", "ridge_lags"]}),
                 track=False)
    assert r["search"]["n_trials"] >= 1


def test_vectorbt():
    r = run_spec(spec(engine="vectorbt", data={"type": "synthetic", "n": 400, "seed": 3},
                      params={"fast_windows": [5, 10], "slow_windows": [20, 40]}), track=False)
    assert r["metrics"]["n_combinations"] == 4 and "oos_sharpe" in r["metrics"]


def test_inline_data_and_queue_roundtrip():
    rc = ResearchClient(agent="grokbot-test")
    vals = [20 + (i % 7) for i in range(120)]
    job = rc.submit({"name": "q", "domain": "weather", "engine": "baseline",
                     "data": {"type": "inline", "values": vals}, "horizon": 3}, run_now=True)
    res = rc.result(job["id"])
    assert res["status"] == "done", res.get("error")
    assert rc.history(domain="weather")[0]["id"] == job["id"]


def test_cost_guard_holds_big_jobs():
    rc = ResearchClient(agent="grokbot-test")
    job = rc.submit({"name": "big", "domain": "macro", "engine": "autogluon",
                     "data": {"type": "synthetic", "n": 300}, "params": {"presets": "best_quality"},
                     "budget": {"max_cpus": 4, "timeout_minutes": 120}})
    assert job["status"] == "needs_approval"
    assert rc.approve(job["id"]) and rc.status(job["id"]) == "queued"
    # leave it un-run: mark failed so a worker doesn't pick it up later
    from qrlab import jobstore
    jobstore.finish(job["id"], error="test cleanup")


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), "..", ".venvs/autogluon/bin/python")),
                    reason="autogluon venv missing")
def test_autogluon_subprocess():
    s = spec(domain="weather", engine="autogluon", data={"type": "synthetic", "n": 200, "seed": 2},
             kalshi={"strike": 15, "direction": "above"},
             params={"hyperparameters": {"Naive": {}, "SeasonalNaive": {}, "ETS": {}}},
             budget={"max_trials": 1, "max_cpus": 2, "timeout_minutes": 2})
    r = run_spec(s)
    assert len(r["forecast"]["mean"]) == 5 and "p_yes" in r["kalshi"]
