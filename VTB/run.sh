#!/usr/bin/env bash
# Thin wrapper around run.py (optional). Prefer: python run.py --mode continuous|discrete
# Usage: MODE=discrete ./run.sh --stages test
set -euo pipefail
cd "$(dirname "$0")"
exec python run.py --mode "${MODE:-continuous}" "$@"
