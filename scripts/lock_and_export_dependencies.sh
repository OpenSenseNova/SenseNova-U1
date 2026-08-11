#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ "${1:-}" == "--check" ]]; then
    uv --quiet --project . lock --check
    uv --quiet --directory training lock --check
    uv run --quiet --isolated --no-project --python '>=3.11' python scripts/render_requirements.py --check
else
    uv --quiet --project . lock
    uv --quiet --directory training lock
    uv run --quiet --isolated --no-project --python '>=3.11' python scripts/render_requirements.py
fi
