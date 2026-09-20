#!/usr/bin/env bash
# ==============================================================================
# Script: data/download.sh
# Purpose: Pulls CoIR-Retrieval/apps (train/valid/test) into data/raw/
# Idempotent: skips if already present.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Detect python executable (favor local virtual environment)
if [[ -f "${REPO_ROOT}/../.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${REPO_ROOT}/../.venv/Scripts/python.exe"
elif [[ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
elif [[ -f "${REPO_ROOT}/../.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/../.venv/bin/python"
elif [[ -f "${REPO_ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
else
    PYTHON_BIN="python"
fi

PY_TARGET="${SCRIPT_DIR}/download.py"
if command -v wslpath >/dev/null 2>&1; then
    if [[ "${PYTHON_BIN}" == *".exe"* ]]; then
        PY_TARGET="$(wslpath -w "${PY_TARGET}")"
    fi
fi

echo "[ariadne] Running dataset download and preparation using ${PYTHON_BIN}..."
"${PYTHON_BIN}" "${PY_TARGET}"
