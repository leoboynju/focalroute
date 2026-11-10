#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

normalize_bool() {
  local val="${1:-}"
  val="$(printf '%s' "$val" | tr '[:upper:]' '[:lower:]')"
  case "$val" in
    1|true|yes|on) printf 'true' ;;
    0|false|no|off) printf 'false' ;;
    *)
      echo "Error: invalid boolean value '$1'" >&2
      exit 2
      ;;
  esac
}

CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
EXP_NAME="${EXP_NAME:-pi05_libero_vision_router}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-42}"
RUN_MODE="${RUN_MODE:-overwrite}"

PYTORCH_WEIGHT_PATH="${PYTORCH_WEIGHT_PATH:-$REPO_ROOT/models/pi05_libero_pytorch}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$REPO_ROOT/checkpoints}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$REPO_ROOT/assets}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"

HARD_ENABLED="$(normalize_bool "${HARD_ENABLED:-true}")"
SOFT_ENABLED="$(normalize_bool "${SOFT_ENABLED:-true}")"
WANDB_ENABLED_NORM="$(normalize_bool "$WANDB_ENABLED")"
HARD_TOKENS="${HARD_TOKENS:-128}"
SOFT_TOKENS="${SOFT_TOKENS:-128}"
SELECTION_TEMPERATURE="${SELECTION_TEMPERATURE:-1.0}"

VISION_ROUTER_BUDGET_PATH="${VISION_ROUTER_BUDGET_PATH:-}"
VISION_ROUTER_CALIBRATION_SOURCE="${VISION_ROUTER_CALIBRATION_SOURCE:-vit}"
VISION_ROUTER_ATTENTION_COVERAGE="${VISION_ROUTER_ATTENTION_COVERAGE:-0.8}"
VISION_ROUTER_BUDGET_QUANTILE="${VISION_ROUTER_BUDGET_QUANTILE:-0.95}"
VISION_ROUTER_GLOBAL_COVERAGE_UPPER="${VISION_ROUTER_GLOBAL_COVERAGE_UPPER:-0.9}"
VISION_ROUTER_GLOBAL_MIN_TOKENS="${VISION_ROUTER_GLOBAL_MIN_TOKENS:-64}"
VISION_ROUTER_TOKEN_ALIGNMENT="${VISION_ROUTER_TOKEN_ALIGNMENT:-32}"

if [[ "$HARD_ENABLED" == "false" && "$SOFT_ENABLED" == "false" ]]; then
  echo "Error: train_pi05_libero_vision_router_pytorch.sh requires at least one VisionRouter branch enabled." >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: PYTHON_BIN is not executable: $PYTHON_BIN" >&2
  exit 2
fi

ARGS=(
  "$CONFIG_NAME"
  "--exp-name" "$EXP_NAME"
  "--checkpoint-base-dir" "$CHECKPOINT_BASE_DIR"
  "--assets-base-dir" "$ASSETS_BASE_DIR"
  "--batch-size" "$BATCH_SIZE"
  "--num-train-steps" "$NUM_TRAIN_STEPS"
  "--save-interval" "$SAVE_INTERVAL"
  "--log-interval" "$LOG_INTERVAL"
  "--num-workers" "$NUM_WORKERS"
  "--seed" "$SEED"
  "--model.vision-router-selection-temperature" "$SELECTION_TEMPERATURE"
)

if [[ "$HARD_ENABLED" == "true" ]]; then
  ARGS+=("--model.vision-router-hard-enabled" "--model.vision-router-hard-tokens" "$HARD_TOKENS")
else
  ARGS+=("--model.no-vision-router-hard-enabled")
fi

if [[ "$SOFT_ENABLED" == "true" ]]; then
  ARGS+=("--model.vision-router-soft-enabled" "--model.vision-router-soft-tokens" "$SOFT_TOKENS")
else
  ARGS+=("--model.no-vision-router-soft-enabled")
fi

if [[ "$WANDB_ENABLED_NORM" == "true" ]]; then
  ARGS+=("--wandb-enabled")
else
  ARGS+=("--no-wandb-enabled")
fi

case "$RUN_MODE" in
  resume)
    if [[ -n "$VISION_ROUTER_BUDGET_PATH" ]]; then
      echo "Error: VISION_ROUTER_BUDGET_PATH cannot be used when RUN_MODE=resume." >&2
      exit 2
    fi
    ARGS+=("--resume" "--no-overwrite")
    ;;
  overwrite)
    if [[ -n "$PYTORCH_WEIGHT_PATH" ]]; then
      if [[ ! -e "$PYTORCH_WEIGHT_PATH" ]]; then
        echo "Error: PYTORCH_WEIGHT_PATH does not exist: $PYTORCH_WEIGHT_PATH" >&2
        exit 2
      fi
      ARGS+=("--pytorch-weight-path" "$PYTORCH_WEIGHT_PATH")
    fi
    ARGS+=("--overwrite" "--no-resume")
    ;;
  *)
    echo "Error: unsupported RUN_MODE '$RUN_MODE' (expected 'overwrite' or 'resume')." >&2
    exit 2
    ;;
esac

if [[ -n "$VISION_ROUTER_BUDGET_PATH" ]]; then
  if [[ "$HARD_ENABLED" != "true" || "$SOFT_ENABLED" != "true" ]]; then
    echo "Error: VISION_ROUTER_BUDGET_PATH requires both HARD_ENABLED=true and SOFT_ENABLED=true." >&2
    exit 2
  fi
  if [[ ! -e "$VISION_ROUTER_BUDGET_PATH" ]]; then
    echo "Error: VISION_ROUTER_BUDGET_PATH does not exist: $VISION_ROUTER_BUDGET_PATH" >&2
    exit 2
  fi
  ARGS+=(
    "--vision-router-budget-path" "$VISION_ROUTER_BUDGET_PATH"
    "--vision-router-calibration-source" "$VISION_ROUTER_CALIBRATION_SOURCE"
    "--vision-router-attention-coverage" "$VISION_ROUTER_ATTENTION_COVERAGE"
    "--vision-router-budget-quantile" "$VISION_ROUTER_BUDGET_QUANTILE"
    "--vision-router-global-coverage-upper" "$VISION_ROUTER_GLOBAL_COVERAGE_UPPER"
    "--vision-router-global-min-tokens" "$VISION_ROUTER_GLOBAL_MIN_TOKENS"
    "--vision-router-token-alignment" "$VISION_ROUTER_TOKEN_ALIGNMENT"
  )
fi

if [[ "$#" -gt 0 ]]; then
  ARGS+=("$@")
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_MODE="$WANDB_MODE"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC_PER_NODE" \
    "$SCRIPT_DIR/train_pytorch.py" \
    "${ARGS[@]}"
else
  exec "$PYTHON_BIN" "$SCRIPT_DIR/train_pytorch.py" "${ARGS[@]}"
fi
