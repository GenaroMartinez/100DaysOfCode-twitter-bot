"""JSON-in/JSON-out CLI so agents in any language (e.g. the Node bots) can use the lab.

  python -m qrlab resources
  python -m qrlab estimate examples/crypto_optuna.json
  python -m qrlab submit examples/crypto_optuna.json [--run] [--agent NAME]
  python -m qrlab worker [--once] [--poll 5]
  python -m qrlab status|result|approve JOB_ID
  python -m qrlab list [--status done] [--domain crypto]
  python -m qrlab history [--domain kalshi]
  python -m qrlab run examples/weather_baseline.json      # direct, still tracked
A spec argument of "-" reads JSON from stdin.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys


def _load(path: str) -> dict:
    return json.load(sys.stdin if path == "-" else open(path))


def _out(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv=None) -> int:
    from .client import ResearchClient

    ap = argparse.ArgumentParser(prog="qrlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resources")
    for c in ("estimate", "run"):
        sub.add_parser(c).add_argument("spec")
    s = sub.add_parser("submit")
    s.add_argument("spec"); s.add_argument("--run", action="store_true")
    s.add_argument("--agent", default=None)
    w = sub.add_parser("worker")
    w.add_argument("--once", action="store_true"); w.add_argument("--poll", type=float, default=5)
    for c in ("status", "result", "approve"):
        p = sub.add_parser(c); p.add_argument("job_id")
        if c == "result":
            p.add_argument("--wait", action="store_true"); p.add_argument("--timeout", type=float, default=3600)
    ls = sub.add_parser("list")
    ls.add_argument("--status"); ls.add_argument("--domain"); ls.add_argument("--agent")
    ls.add_argument("--limit", type=int, default=50)
    h = sub.add_parser("history"); h.add_argument("--domain"); h.add_argument("--limit", type=int, default=20)
    a = ap.parse_args(argv)
    rc = ResearchClient(agent=getattr(a, "agent", None) or "cli")

    if a.cmd == "resources":
        _out(rc.resources())
    elif a.cmd == "estimate":
        _out(rc.estimate(_load(a.spec)))
    elif a.cmd == "submit":
        _out(rc.submit(_load(a.spec), run_now=a.run))
    elif a.cmd == "run":
        from .runner import run_spec
        from .schemas import ExperimentSpec

        spec = ExperimentSpec.from_dict({"submitted_by": "cli", **_load(a.spec)})
        est = rc.estimate(spec)
        if est["requires_approval"]:
            _out({"error": "estimate exceeds cost guard; set budget.approved=true", "estimate": est})
            return 2
        _out(run_spec(spec))
    elif a.cmd == "worker":
        from .worker import loop

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
        loop(poll=a.poll, once=a.once)
    elif a.cmd == "status":
        _out({"id": a.job_id, "status": rc.status(a.job_id)})
    elif a.cmd == "result":
        _out(rc.result(a.job_id, wait=a.wait, timeout=a.timeout))
    elif a.cmd == "approve":
        _out({"id": a.job_id, "approved": rc.approve(a.job_id)})
    elif a.cmd == "list":
        _out(rc.list(status=a.status, domain=a.domain, submitted_by=a.agent, limit=a.limit))
    elif a.cmd == "history":
        _out(rc.history(domain=a.domain, limit=a.limit))
    return 0


if __name__ == "__main__":
    sys.exit(main())
