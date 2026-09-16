#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if command -v micromamba >/dev/null 2>&1; then
  micromamba env create -f environment.yml
elif command -v mamba >/dev/null 2>&1; then
  mamba env create -f environment.yml
elif command -v conda >/dev/null 2>&1; then
  conda env create -f environment.yml
else
  printf 'No conda-compatible environment manager was found.\n' >&2
  printf 'Install with: python -m venv .venv && .venv/bin/pip install -r requirements.txt\n' >&2
  exit 1
fi

printf 'Environment created. Activate it with: conda activate voila3d\n'
