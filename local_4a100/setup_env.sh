#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname -- "$SCRIPT_DIR")"
cd "$REPO_DIR"
PYTHON_BIN="${RISKPO_PYTHON:-python3.10}"
REQUIREMENTS_FILE="${RISKPO_REQUIREMENTS:-local_4a100/requirements.txt}"
"$PYTHON_BIN" -c 'import sys, platform; assert sys.version_info[:2] == (3,10); assert platform.system() == "Linux"; assert platform.machine() == "x86_64"'
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install -r "$REQUIREMENTS_FILE"
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
echo "Environment installed. Activate it with: source .venv/bin/activate"
echo "Check the allocated GPUs with: python local_4a100/check_environment.py"
