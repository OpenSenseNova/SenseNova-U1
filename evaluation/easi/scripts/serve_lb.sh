#!/usr/bin/env bash
# Deprecated: merged into serve.sh. serve.sh now handles both DP=1 (direct)
# and DP>1 (multi-replica + LB) based on the DP env var.
#
# This shim forwards to serve.sh with a warning. Update your commands:
#
#   Old:  DP=4 TP=2 bash evaluation/easi/scripts/serve_lb.sh
#   New:  DP=4 TP=2 bash evaluation/easi/scripts/serve.sh
set -euo pipefail

echo "[serve_lb] DEPRECATED: use serve.sh instead — it now supports DP>1 natively." >&2
echo "[serve_lb] forwarding to serve.sh with the same env..." >&2
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/serve.sh" "$@"
