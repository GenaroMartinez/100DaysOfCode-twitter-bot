"""Queue worker: run inside the core venv (``python -m qrlab worker``)."""
from __future__ import annotations

import logging
import time
import traceback

from . import jobstore
from .runner import run_spec
from .schemas import ExperimentSpec

log = logging.getLogger("qrlab.worker")


def _execute(job: dict) -> None:
    spec = ExperimentSpec.from_dict(job["spec"])
    log.info("running %s (%s/%s)", spec.id, spec.domain, spec.engine)
    try:
        jobstore.finish(spec.id, result=run_spec(spec))
    except Exception as e:  # noqa: BLE001
        jobstore.finish(spec.id, error=f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2000:]}")


def process_one(job_id: str | None = None) -> bool:
    if job_id:
        with jobstore.connect() as con:
            n = con.execute("UPDATE jobs SET status='running', started_at=? WHERE id=? AND status='queued'",
                            (jobstore._now(), job_id)).rowcount
        job = jobstore.get(job_id) if n else None
    else:
        job = jobstore.claim_next()
    if not job:
        return False
    _execute(job)
    return True


def loop(poll: float = 5.0, once: bool = False) -> None:
    while True:
        did = process_one()
        if once and not did:
            return
        if not did:
            time.sleep(poll)
