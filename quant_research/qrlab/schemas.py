"""Experiment specification shared by every engine, agent and worker."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

DOMAINS = ("crypto", "commodities", "weather", "macro", "kalshi")
ENGINES = ("baseline", "optuna", "ray_tune", "autogluon", "vectorbt")


@dataclass
class Budget:
    max_trials: int = 20
    max_cpus: int = 2
    timeout_minutes: float = 10.0
    approved: bool = False  # set by a human to run above the cost guard


@dataclass
class KalshiContract:
    """Binary settlement rule: YES if value at horizon is above/below strike."""

    strike: float
    direction: str = "above"  # "above" | "below"
    ticker: Optional[str] = None  # informational; read-only lookups only


@dataclass
class ExperimentSpec:
    name: str
    domain: str
    engine: str
    data: dict
    horizon: int = 7
    params: dict = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    kalshi: Optional[KalshiContract] = None
    cv_windows: int = 3
    submitted_by: str = "unknown"
    tags: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def validate(self) -> "ExperimentSpec":
        if self.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}, got {self.domain!r}")
        if self.engine not in ENGINES:
            raise ValueError(f"engine must be one of {ENGINES}, got {self.engine!r}")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if "type" not in self.data:
            raise ValueError("data.type is required")
        if self.kalshi and self.kalshi.direction not in ("above", "below"):
            raise ValueError("kalshi.direction must be 'above' or 'below'")
        if self.budget.max_cpus < 1 or self.budget.max_trials < 1:
            raise ValueError("budget values must be positive")
        return self

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExperimentSpec":
        d = dict(d)
        d["budget"] = Budget(**d.get("budget", {}) or {})
        k = d.get("kalshi")
        d["kalshi"] = KalshiContract(**k) if k else None
        return cls(**d).validate()

    @classmethod
    def from_json(cls, s: str) -> "ExperimentSpec":
        return cls.from_dict(json.loads(s))
