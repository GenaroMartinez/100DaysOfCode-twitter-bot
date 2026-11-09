"""Evaluate every model at default settings; cheap sanity benchmark."""
from __future__ import annotations

from .. import models
from .common import evaluate, finalize


def run(spec, y, season: int) -> dict:
    metric = spec.params.get("metric", "mase")
    board = []
    for name in spec.params.get("models", models.MODEL_NAMES):
        try:
            r = evaluate(y, spec, name, {}, season)
            board.append({"model": name, "params": {}, "cv": r["cv"], **r["scores"]})
        except Exception as e:  # noqa: BLE001 - a failed model is just skipped
            board.append({"model": name, "error": str(e)})
    ok = sorted([b for b in board if "error" not in b], key=lambda b: b[metric])
    if not ok:
        raise RuntimeError("all baseline models failed")
    best = dict(ok[0], weight=1.0)
    leaderboard = [{k: v for k, v in b.items() if k != "cv"} for b in board]
    return finalize(y, spec, [best], season, leaderboard=leaderboard)
