# qrlab: quantitative experimentation lab

Research-only forecasting and backtesting service that agents (GrokBot or any
other bot) can submit experiments to and read results from. It lives entirely
under `quant_research/` and does not touch the Node bot in `src/`.

**No live trading.** No module places orders or calls a broker/exchange order
endpoint. `config.LIVE_TRADING_ENABLED` is hard-coded `False` and asserted
before every run. VectorBT is used only on historical prices.

## Layout

```
quant_research/
  qrlab/
    schemas.py        ExperimentSpec: the contract between agents and engines
    data.py           loaders: csv/parquet, inline, synthetic, coingecko, open_meteo, fred
    models.py         CPU forecasters (naive, seasonal_naive, drift, ets, theta, arima, ridge_lags, gbr_lags)
    evaluation.py     rolling-origin CV, MASE/RMSE/sMAPE, quantiles, Kalshi P(YES) + Brier/log-loss
    engines/
      baseline.py         all models at defaults (cheap benchmark)
      optuna_engine.py    model selection + HPO, then ensemble-weight search over top-k
      raytune_engine.py   parallel search (OptunaSearch) on local Ray or RAY_ADDRESS cluster
      autogluon_engine.py runs ag_worker.py inside the isolated AutoGluon venv
      vectorbt_engine.py  vectorised parameter-grid backtests with out-of-sample check
    tracking.py       MLflow logging (sqlite store under data/)
    jobstore.py       SQLite job queue (queued / needs_approval / running / done / failed)
    resources.py      capacity report + per-job CPU/memory/cost estimate
    client.py         ResearchClient: the API agents call
    worker.py, cli.py JSON CLI + queue worker
  integrations/node/qrlab.js   bridge for the existing Node bots (no new npm deps)
  examples/*.json              one spec per domain/engine
  scripts/setup_envs.sh        recreate both venvs + run verification
  scripts/verify_install.py    small working test per tool
  tests/                       end-to-end smoke tests
  data/  (git-ignored)         jobs.db, mlflow.db, optuna.db, artifacts = persistent research memory
```

## Environments (Python 3.11)

| venv | contents | why separate |
|---|---|---|
| `.venvs/core` | Optuna 5.0, Ray 2.60 (Tune), MLflow 3.18, VectorBT 1.2, statsmodels, scikit-learn, pandas 3.1 | lightweight; runs the queue, tracking, and most engines |
| `.venvs/autogluon` | AutoGluon TimeSeries 1.6.3, torch 2.13, pandas 2.3 | AutoGluon pins pandas < 3 and pulls in torch; isolating it avoids conflicts |

Only the core venv writes to MLflow, so a single MLflow version owns the tracking DB.

## Domains

`crypto`, `commodities`, `weather`, `macro`, `kalshi`. Any spec can include a
`kalshi` block (`strike`, `direction`) to turn the forecast distribution into a
settlement probability for a binary contract, backtested with Brier score and log loss
on the CV windows. Example: forecast an NYC daily high, then ask P(high > 15°C).

## Using it from an agent

```python
import sys; sys.path.insert(0, "/home/user/100DaysOfCode-twitter-bot/quant_research")
from qrlab.client import ResearchClient

rc = ResearchClient(agent="grokbot-weather")
print(rc.estimate(spec))            # check cost before submitting
job = rc.submit(spec)               # -> {"id", "status", "estimate"}
res = rc.result(job["id"], wait=True)
rc.history(domain="weather")        # compact memory of past results
```

Submit, status, and result use only the standard library. A worker must be
running to execute queued jobs:

```bash
.venvs/core/bin/python -m qrlab worker          # long-running
.venvs/core/bin/python -m qrlab worker --once   # drain the queue and exit
.venvs/core/bin/python -m qrlab --help
.venvs/core/bin/mlflow ui --backend-store-uri sqlite:///data/mlflow.db   # browse runs
```

From Node: `require('./quant_research/integrations/node/qrlab').submit(spec, 'grokbot')`.

## Cost and resource guard

Every submission is estimated first (`est_cpu_minutes`, `est_peak_memory_gb`,
`est_cost_usd`). A job whose estimate exceeds `QR_MAX_CPU_MINUTES` (default 30)
or `QR_MAX_MEMORY_GB` (default 8) goes to `needs_approval` and does not run until
`python -m qrlab approve <id>` (or `budget.approved: true`) is set by a human.
Set `QR_USD_PER_CPU_HOUR` if you run on billed hardware to get dollar estimates.

## Network note

In the current cloud environment, outbound access to api.coingecko.com,
archive-api.open-meteo.com, fred.stlouisfed.org and api.elections.kalshi.com is
blocked by the network policy, so live-data examples fail there. Synthetic, inline,
and local CSV/parquet data work everywhere.

## Tests

```bash
.venvs/core/bin/python -m pytest -q tests          # ~40 s, uses an isolated temp data dir
.venvs/core/bin/python scripts/verify_install.py core
.venvs/autogluon/bin/python scripts/verify_install.py autogluon
```
