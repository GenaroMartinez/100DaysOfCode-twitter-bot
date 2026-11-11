"""Shared interfaces for the combination discovery engine.

Every subpackage (datasets, features, zoo, validation, combine, memory, ...)
codes against these contracts so they can be developed independently.

Core rule for anti-leakage: a ``Dataset.frame`` index is the UTC time at which
each row's information became AVAILABLE (publication / release lag applied),
not the period it describes. Joining datasets on that index is point-in-time.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ data ----


@dataclass
class Dataset:
    frame: pd.DataFrame  # UTC tz-aware DatetimeIndex = availability time
    meta: dict = field(default_factory=dict)
    # expected meta keys: source, symbol, fields, freq, fetched_at,
    # version_hash, availability ("realtime" | "release_lag:<rule>"), url

    def validate(self) -> "Dataset":
        idx = self.frame.index
        if not isinstance(idx, pd.DatetimeIndex) or idx.tz is None:
            raise ValueError("Dataset.frame needs a tz-aware DatetimeIndex (UTC)")
        if not idx.is_monotonic_increasing:
            raise ValueError("Dataset.frame index must be sorted ascending")
        if idx.has_duplicates:
            raise ValueError("Dataset.frame index has duplicates")
        for k in ("source", "symbol", "freq", "version_hash"):
            if k not in self.meta:
                raise ValueError(f"Dataset.meta missing {k!r}")
        return self


def frame_hash(frame: pd.DataFrame) -> str:
    """Stable content hash of a DataFrame (for dataset versioning)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------- features ----

FeatureFn = Callable[[pd.DataFrame], pd.DataFrame]
FEATURES: dict[str, dict] = {}


def register_feature(name: str, family: str, description: str = "") -> Callable[[FeatureFn], FeatureFn]:
    """Register a CAUSAL feature function: output row t may use input rows <= t only."""

    def deco(fn: FeatureFn) -> FeatureFn:
        if name in FEATURES:
            raise ValueError(f"feature {name!r} already registered")
        FEATURES[name] = {"fn": fn, "family": family, "description": description}
        return fn

    return deco


# --------------------------------------------------------------- models ----


@runtime_checkable
class Forecaster(Protocol):
    name: str
    kind: str  # "regressor" | "classifier"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Forecaster": ...

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def has_proba(model: Any) -> bool:
    return callable(getattr(model, "predict_proba", None))


# ----------------------------------------------------------- validation ----


@runtime_checkable
class Splitter(Protocol):
    def split(self, index: pd.Index) -> Iterator[tuple[np.ndarray, np.ndarray]]: ...


# ------------------------------------------------------------- registry ----

METHOD_KINDS = ("forecaster", "feature", "indicator", "test", "calibrator",
                "simulator", "decision", "validation")
METHOD_STATUS = ("implemented", "library-available", "unsupported")


@dataclass
class MethodSpec:
    name: str
    family: str  # one of the 23 family names
    kind: str  # METHOD_KINDS
    purpose: str
    inputs: str
    target: str
    assumptions: str
    horizons: str
    compute: str
    combines_with: list[str]
    limitations: str
    python_impl: list[str]
    verification: str
    status: str = "library-available"  # METHOD_STATUS
    qrlab_entrypoint: Optional[str] = None  # dotted path when implemented in qrlab

    def validate(self) -> "MethodSpec":
        if self.kind not in METHOD_KINDS:
            raise ValueError(f"{self.name}: kind {self.kind!r} not in {METHOD_KINDS}")
        if self.status not in METHOD_STATUS:
            raise ValueError(f"{self.name}: status {self.status!r} not in {METHOD_STATUS}")
        if self.status == "implemented" and not self.qrlab_entrypoint:
            raise ValueError(f"{self.name}: implemented methods need qrlab_entrypoint")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------- hashing ----


def content_hash(obj: Any) -> str:
    """sha256 of canonical JSON; identical experiment configs hash identically."""
    s = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:20]


# ---------------------------------------------------- validation labels ----

VALIDATION_STATUS = ("exploratory", "backtested", "independently_validated",
                     "paper_tested", "production_eligible", "rejected")
