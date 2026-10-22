#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=${CONDA_BIN:-conda}
TARGET_PYTHON=${TARGET_PYTHON:-$(command -v python)}
GRAPH_TOOL_SPEC=${GRAPH_TOOL_SPEC:-graph-tool-base}
PYEMD_VERSION=${PYEMD_VERSION:-1.0.0}
PYEMD_SOURCE=${PYEMD_SOURCE:-https://github.com/wmayner/pyemd/archive/refs/tags/1.0.0.tar.gz}

if [[ -z "${TARGET_PYTHON}" || ! -x "${TARGET_PYTHON}" ]]; then
  echo "Activate the target environment or set TARGET_PYTHON to its Python executable." >&2
  exit 2
fi
if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "Conda is required; activate it or set CONDA_BIN to the conda executable." >&2
  exit 2
fi

if command -v micromamba >/dev/null 2>&1; then
  MAMBA_BIN=$(command -v micromamba)
  CLEANUP_DIR=""
else
  CLEANUP_DIR=$(mktemp -d)
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "${CLEANUP_DIR}" bin/micromamba
  MAMBA_BIN="${CLEANUP_DIR}/bin/micromamba"
fi
CONDA_BASE=$("${CONDA_BIN}" info --base)
MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-${CONDA_BASE}/micromamba-root} \
  "${MAMBA_BIN}" install -y -p "$(dirname "${TARGET_PYTHON}")/.." \
  -c conda-forge "${GRAPH_TOOL_SPEC}"
if [[ -n "${CLEANUP_DIR}" ]]; then
  rm -rf -- "${CLEANUP_DIR}"
fi

# Build pyemd against the active NumPy instead of allowing the solver to
# downgrade or replace it.
if ! "${TARGET_PYTHON}" - <<'PY' >/dev/null 2>&1
import pyemd  # noqa: F401
PY
then
  "${TARGET_PYTHON}" -m pip install "Cython>=3.0" setuptools_scm
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYEMD="${PYEMD_VERSION}" \
    "${TARGET_PYTHON}" -m pip install \
      --no-cache-dir \
      --no-build-isolation \
      --force-reinstall \
      --no-deps \
      "${PYEMD_SOURCE}"
fi

"${TARGET_PYTHON}" - <<'PY'
import warnings
warnings.filterwarnings("ignore", message="Error importing cairo.*")
warnings.filterwarnings("ignore", message="Error importing draw module.*")
import graph_tool.all as gt
import numpy
import pyemd
import scipy
import torch

graph = gt.Graph(directed=False)
graph.add_vertex(4)
graph.add_edge_list([(0, 1), (1, 2), (2, 3)])
gt.minimize_blockmodel_dl(graph)
print("Exact environment validated")
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("scipy", scipy.__version__)
print("pyemd", pyemd.__version__)
PY
