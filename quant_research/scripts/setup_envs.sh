#!/usr/bin/env bash
# Recreate the two isolated Python 3.11 environments. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON311:-$(command -v python3.11)}
command -v uv >/dev/null || pip install --user uv
[ -d .venvs/core ] || uv venv -p "$PY" .venvs/core
[ -d .venvs/autogluon ] || uv venv -p "$PY" .venvs/autogluon
VIRTUAL_ENV=.venvs/core uv pip install -r requirements-core.txt
VIRTUAL_ENV=.venvs/autogluon uv pip install -r requirements-autogluon.txt
.venvs/core/bin/python scripts/verify_install.py core
.venvs/autogluon/bin/python scripts/verify_install.py autogluon
