#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_NAME="${EXP_NAME:-pi05_libero_hard_router}"
export HARD_ENABLED="${HARD_ENABLED:-true}"
export SOFT_ENABLED="${SOFT_ENABLED:-false}"
export HARD_TOKENS="${HARD_TOKENS:-128}"
export SOFT_TOKENS="${SOFT_TOKENS:-0}"
exec bash "$SCRIPT_DIR/train_pi05_libero_vision_router_pytorch.sh" "$@"
