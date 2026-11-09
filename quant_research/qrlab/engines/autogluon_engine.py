"""AutoGluon TimeSeries, executed in its own venv to isolate torch/pandas pins.

The core process serialises the spec to JSON, runs ``ag_worker`` with the
AutoGluon interpreter, and logs the returned result to MLflow itself, so only
one MLflow version ever touches the tracking database.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from .. import config


def run(spec, y, season: int) -> dict:
    py = config.AUTOGLUON_PYTHON
    if not py.exists():
        raise RuntimeError(f"AutoGluon venv not found at {py}; see scripts/setup_envs.sh")
    with tempfile.TemporaryDirectory() as td:
        inp, out = Path(td) / "in.json", Path(td) / "out.json"
        payload = spec.to_dict()
        # ship the already-loaded series so both venvs see identical data
        payload["data"] = {"type": "inline", "timestamps": [str(t) for t in y.index],
                           "values": [float(v) for v in y.values]}
        inp.write_text(json.dumps(payload))
        n = str(spec.budget.max_cpus)
        env = {**os.environ, "PYTHONPATH": str(config.ROOT), "OMP_NUM_THREADS": n,
               "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "QR_AG_NUM_CPUS": n}
        timeout = spec.budget.timeout_minutes * 60 * 1.5 + 120
        proc = subprocess.run([str(py), "-m", "qrlab.engines.ag_worker", str(inp), str(out)],
                              cwd=config.ROOT, env=env, capture_output=True, text=True,
                              timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"AutoGluon worker failed:\n{proc.stderr[-3000:]}")
        return json.loads(out.read_text())
