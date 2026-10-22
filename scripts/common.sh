#!/usr/bin/env bash

# Resolve every launcher relative to the repository instead of a local
# workstation path. Set PYBIN explicitly to select a virtual environment.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYBIN=${PYBIN:-python}
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
