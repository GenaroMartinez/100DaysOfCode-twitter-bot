"""Agent-facing API. This is what a GrokBot (or any other) agent imports.

    import sys; sys.path.insert(0, "/path/to/quant_research")
    from qrlab.client import ResearchClient

    rc = ResearchClient(agent="grokbot-macro")
    job = rc.submit({"name": "cpi-3m", "domain": "macro", "engine": "optuna",
                     "data": {"type": "fred", "series_id": "CPIAUCSL"}, "horizon": 3})
    rc.result(job["id"], wait=True)

Submitting, polling and fetching results only need the standard library, so an
agent can use this from its own interpreter. Running experiments requires the
core venv (``python -m qrlab worker``), or ``run_now=True`` from inside it.
"""
from __future__ import annotations

import time
from typing import Union

from . import jobstore, resources
from .schemas import ExperimentSpec

SpecLike = Union[dict, ExperimentSpec]


class ResearchClient:
    def __init__(self, agent: str = "unknown"):
        self.agent = agent

    def _spec(self, spec: SpecLike) -> ExperimentSpec:
        if isinstance(spec, dict):
            spec = {"submitted_by": self.agent, **spec}
            return ExperimentSpec.from_dict(spec)
        return spec.validate()

    # ---- planning --------------------------------------------------------
    def resources(self) -> dict:
        return resources.report()

    def estimate(self, spec: SpecLike) -> dict:
        return resources.estimate(self._spec(spec))

    # ---- lifecycle -------------------------------------------------------
    def submit(self, spec: SpecLike, run_now: bool = False) -> dict:
        s = self._spec(spec)
        est = resources.estimate(s)
        status = jobstore.insert(s, est)
        out = {"id": s.id, "status": status, "estimate": est}
        if run_now and status == "queued":
            from .worker import process_one

            process_one(job_id=s.id)
            out["status"] = jobstore.get(s.id)["status"]
        return out

    def status(self, job_id: str) -> str | None:
        j = jobstore.get(job_id)
        return j["status"] if j else None

    def result(self, job_id: str, wait: bool = False, timeout: float = 3600,
               poll: float = 5.0) -> dict | None:
        deadline = time.time() + timeout
        while True:
            j = jobstore.get(job_id)
            if j is None:
                raise KeyError(job_id)
            if j["status"] in ("done", "failed") or not wait or time.time() > deadline:
                return j
            time.sleep(poll)

    def approve(self, job_id: str) -> bool:
        """Human-in-the-loop approval for jobs over the cost guard."""
        return jobstore.approve(job_id)

    def list(self, **filters) -> list[dict]:
        return jobstore.list_jobs(**filters)

    def history(self, domain: str | None = None, limit: int = 20) -> list[dict]:
        """Compact memory of past results for an agent's context window."""
        out = []
        for row in jobstore.list_jobs(status="done", domain=domain, limit=limit):
            r = jobstore.get(row["id"])["result"] or {}
            out.append({"id": row["id"], "name": row["name"], "engine": row["engine"],
                        "finished_at": row["finished_at"], "metrics": r.get("metrics"),
                        "kalshi_p_yes": (r.get("kalshi") or {}).get("p_yes"),
                        "forecast_mean": (r.get("forecast") or {}).get("mean")})
        return out
