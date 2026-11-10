from collections.abc import Sequence
from contextlib import ExitStack
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import shutil
from typing import Any, Literal

import numpy as np
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing

AttentionSource = Literal["vit", "vlm", "action"]

TOKEN_BUDGET_FORMAT_VERSION = 1
GLOBAL_TOKEN_METHOD = "after_aligned_hard"
DEFAULT_CANDIDATE_BUDGETS: tuple[int, ...] = (32, 64, 96, 128, 160, 192, 256, 384, 512)
DEFAULT_VIT_LAYERS: tuple[int, ...] = (23, 24, 25, 26)
DEFAULT_VLM_LAYERS: tuple[int, ...] = (14, 15, 16, 17)
DEFAULT_ACTION_LAYERS: tuple[int, ...] = (14, 15, 16, 17)
DEFAULT_ACTION_NUM_STEPS: int = 10
DEFAULT_ACTION_STEPS: tuple[int, ...] = (0, 5, 9)

METHOD_BY_SOURCE: dict[str, str] = {
    "vit": "full_vit_patch_attention",
    "vlm": "full_prefix_prompt_to_visual_attention",
    "action": "full_action_to_visual_attention",
}


def default_budget_filenames(source: str = "vit") -> tuple[str, str]:
    if source not in METHOD_BY_SOURCE:
        raise ValueError(f"Unsupported attention_source '{source}'. Expected one of {sorted(METHOD_BY_SOURCE)}")
    return f"token_budget_{source}.json", f"token_budget_{source}.npz"


def align_token_count(count: int | float, alignment: int = 32) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be > 0, got {alignment}")
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    return int(math.ceil(float(count) / float(alignment)) * int(alignment))


def validate_And_normalize_relevance(
    relevance: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate per-sample relevance scores and normalize over valid visual tokens.

    Returns:
        normalized_probs: float64 array [F, N] summing to 1 over valid positions per row
        valid_mask_bool: bool array [F, N]
        raw_mass: float64 array [F] of valid score sums prior to normalization
    """
    scores = np.asarray(relevance, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError(f"relevance must be a non-empty 2D array [F, N], got shape {scores.shape}")

    if valid_mask is None:
        mask_bool = np.ones_like(scores, dtype=bool)
    else:
        mask_bool = np.asarray(valid_mask, dtype=bool)
        if mask_bool.shape != scores.shape:
            raise ValueError(f"valid_mask shape {mask_bool.shape} does not match relevance shape {scores.shape}")

    if not np.all(np.isfinite(scores)):
        raise ValueError("relevance contains NaN or infinite values")

    valid_scores = np.where(mask_bool, scores, 0.0)
    if np.any(valid_scores < 0.0) or np.any(scores[mask_bool] < 0.0):
        raise ValueError("relevance contains negative values")

    valid_counts = mask_bool.sum(axis=1)
    if np.any(valid_counts <= 0):
        raise ValueError("Each sample must have at least one valid visual token")

    raw_mass = valid_scores.sum(axis=1)
    if not np.all(np.isfinite(raw_mass)) or np.any(raw_mass <= 0.0):
        raise ValueError("Each sample must have positive valid attention mass")

    normalized = valid_scores / raw_mass[:, None]
    return normalized, mask_bool, raw_mass


def compute_tokens_for_coverage(
    relevance: np.ndarray,
    valid_mask: np.ndarray | None,
    coverage_target: float,
) -> np.ndarray:
    """Compute per-sample minimal token count K_f(c) reaching cumulative coverage c."""
    if not (0.0 < float(coverage_target) <= 1.0):
        raise ValueError(f"coverage_target must be in (0, 1], got {coverage_target}")

    normalized, mask_bool, _ = validate_And_normalize_relevance(relevance, valid_mask)
    num_samples, num_tokens = normalized.shape
    counts = np.zeros(num_samples, dtype=np.int64)
    indices = np.arange(num_tokens, dtype=np.int64)
    target = float(coverage_target)

    for i in range(num_samples):
        row_mask = mask_bool[i]
        row_probs = normalized[i]
        valid_count = int(row_mask.sum())
        # Stable descending sort among valid tokens (tie-breaking by original patch index ascending).
        order = np.lexsort((indices, -row_probs, ~row_mask))[:valid_count]
        sorted_probs = row_probs[order]
        csum = np.cumsum(sorted_probs, dtype=np.float64)
        csum = csum / csum[-1]
        idx = int(np.searchsorted(csum >= (target - 1e-12), True, side="left"))
        counts[i] = min(idx + 1, valid_count)

    return counts


def _quantile_dict(values: np.ndarray, sample_quantile: float) -> dict[str, int]:
    q_map = {
        "min": int(np.min(values)),
        "p05": int(np.quantile(values, 0.05, method="higher")),
        "p25": int(np.quantile(values, 0.25, method="higher")),
        "p50": int(np.quantile(values, 0.50, method="higher")),
        "p75": int(np.quantile(values, 0.75, method="higher")),
        "p90": int(np.quantile(values, 0.90, method="higher")),
        "p95": int(np.quantile(values, 0.95, method="higher")),
        "p99": int(np.quantile(values, 0.99, method="higher")),
        "max": int(np.max(values)),
        f"q_{float(sample_quantile):.4f}": int(np.quantile(values, float(sample_quantile), method="higher")),
    }
    return q_map


def compute_token_budget_summary(
    relevance: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    coverage_target: float = 0.80,
    sample_quantile: float = 0.95,
    alignment: int = 32,
    global_coverage_upper: float = 0.90,
    global_min_tokens: int = 64,
    candidate_budgets: Sequence[int] = DEFAULT_CANDIDATE_BUDGETS,
    visual_attention_mass: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute token budget summary statistics and per-sample hard token counts K_f(c)."""
    if not (0.0 < float(coverage_target) <= 1.0):
        raise ValueError(f"coverage_target must be in (0, 1], got {coverage_target}")
    if not (0.0 < float(sample_quantile) <= 1.0):
        raise ValueError(f"sample_quantile must be in (0, 1], got {sample_quantile}")
    if not (0.0 < float(global_coverage_upper) <= 1.0):
        raise ValueError(f"global_coverage_upper must be in (0, 1], got {global_coverage_upper}")
    if alignment <= 0:
        raise ValueError(f"alignment must be > 0, got {alignment}")
    if global_min_tokens <= 0:
        raise ValueError(f"global_min_tokens must be > 0, got {global_min_tokens}")

    normalized, mask_bool, inferred_mass = validate_And_normalize_relevance(relevance, valid_mask)
    num_samples, total_visual_tokens = normalized.shape
    valid_counts = mask_bool.sum(axis=1)
    min_valid_tokens = int(valid_counts.min())
    max_valid_tokens = int(valid_counts.max())

    hard_tokens_per_sample = compute_tokens_for_coverage(normalized, mask_bool, coverage_target)
    upper_tokens_per_sample = compute_tokens_for_coverage(normalized, mask_bool, global_coverage_upper)

    raw_hard_quantile = int(np.quantile(hard_tokens_per_sample, float(sample_quantile), method="higher"))
    recommended_hard_tokens = align_token_count(raw_hard_quantile, alignment)
    if recommended_hard_tokens > max_valid_tokens:
        raise ValueError(
            f"Aligned hard token budget ({recommended_hard_tokens}) exceeds maximum valid visual tokens ({max_valid_tokens})"
        )

    residual_per_sample = np.maximum(upper_tokens_per_sample - recommended_hard_tokens, 0)
    raw_upper_quantile = int(np.quantile(upper_tokens_per_sample, float(sample_quantile), method="higher"))
    raw_global_quantile = int(np.quantile(residual_per_sample, float(sample_quantile), method="higher"))
    floored_global_tokens = max(int(global_min_tokens), raw_global_quantile)
    recommended_global_tokens = align_token_count(floored_global_tokens, alignment)
    recommended_total_tokens = recommended_hard_tokens + recommended_global_tokens

    # Precompute sorted cumulative sums per sample for coverage curves.
    indices = np.arange(total_visual_tokens, dtype=np.int64)
    sorted_csums: list[np.ndarray] = []
    for i in range(num_samples):
        v_count = int(valid_counts[i])
        order = np.lexsort((indices, -normalized[i], ~mask_bool[i]))[:v_count]
        csum = np.cumsum(normalized[i][order], dtype=np.float64)
        csum = csum / csum[-1]
        sorted_csums.append(csum)

    curve_ks = sorted({int(k) for k in (*candidate_budgets, recommended_hard_tokens) if 1 <= int(k) <= max_valid_tokens})
    coverage_curves: dict[str, dict[str, float | int]] = {}
    for k in curve_ks:
        masses = np.array([float(csum[min(k, len(csum)) - 1]) for csum in sorted_csums], dtype=np.float64)
        coverage_curves[str(k)] = {
            "budget": int(k),
            "mean_coverage": float(np.mean(masses)),
            "p05_coverage": float(np.quantile(masses, 0.05)),
            "fraction_meeting_target": float(np.mean(masses >= (float(coverage_target) - 1e-12))),
        }

    mass_array = np.asarray(inferred_mass if visual_attention_mass is None else visual_attention_mass, dtype=np.float64)
    summary: dict[str, Any] = {
        "coverage_target": float(coverage_target),
        "sample_quantile": float(sample_quantile),
        "alignment": int(alignment),
        "num_samples": int(num_samples),
        "total_visual_tokens": int(total_visual_tokens),
        "min_valid_visual_tokens": min_valid_tokens,
        "max_valid_visual_tokens": max_valid_tokens,
        "raw_hard_quantile_tokens": raw_hard_quantile,
        "recommended_hard_tokens": recommended_hard_tokens,
        "recommended_global_tokens": recommended_global_tokens,
        "recommended_total_tokens": recommended_total_tokens,
        "global_token_method": GLOBAL_TOKEN_METHOD,
        "global_subtracted_hard_tokens": recommended_hard_tokens,
        "global_coverage_upper": float(global_coverage_upper),
        "global_min_tokens": int(global_min_tokens),
        "raw_upper_quantile_tokens": raw_upper_quantile,
        "raw_global_quantile_tokens": raw_global_quantile,
        "floored_global_tokens": floored_global_tokens,
        "target_global_tokens": recommended_global_tokens,
        "hard_token_quantiles": _quantile_dict(hard_tokens_per_sample, sample_quantile),
        "upper_token_quantiles": _quantile_dict(upper_tokens_per_sample, sample_quantile),
        "global_residual_quantiles": _quantile_dict(residual_per_sample, sample_quantile),
        "visual_attention_mass_mean": float(np.mean(mass_array)),
        "coverage_curves": coverage_curves,
    }
    return summary, hard_tokens_per_sample


def build_budget_report(
    *,
    config_name: str,
    repo_id: str | None,
    checkpoint: str,
    attention_source: str,
    model_config_dict: dict[str, Any],
    summary: dict[str, Any],
    layers: Sequence[int],
    camera_names: Sequence[str] = _preprocessing.IMAGE_KEYS,
    seed: int = 42,
    sample_indices_source: str = "uniform_without_replacement",
    num_steps: int | None = None,
    action_steps: Sequence[int] | None = None,
    action_times: Sequence[float] | None = None,
) -> dict[str, Any]:
    if attention_source not in METHOD_BY_SOURCE:
        raise ValueError(f"Unsupported attention_source '{attention_source}'")
    report: dict[str, Any] = {
        "format_version": TOKEN_BUDGET_FORMAT_VERSION,
        "config_name": str(config_name),
        "repo_id": repo_id,
        "checkpoint": str(checkpoint),
        "attention_source": str(attention_source),
        "method": METHOD_BY_SOURCE[attention_source],
        "model_config": model_config_dict,
        "seed": int(seed),
        "sample_indices_source": str(sample_indices_source),
        "layers": [int(x) for x in layers],
        "camera_names": list(camera_names),
        "summary": summary,
    }
    if attention_source == "action":
        steps = list(DEFAULT_ACTION_STEPS if action_steps is None else action_steps)
        total_steps = int(DEFAULT_ACTION_NUM_STEPS if num_steps is None else num_steps)
        times = (
            [round(1.0 - float(s) / float(total_steps), 6) for s in steps]
            if action_times is None
            else [float(t) for t in action_times]
        )
        report["num_steps"] = total_steps
        report["action_steps"] = [int(s) for s in steps]
        report["action_times"] = times
    return report


def save_budget_artifacts(
    output_json_path: str | os.PathLike,
    report: dict[str, Any],
    *,
    sample_indices: np.ndarray,
    relevance: np.ndarray,
    valid_mask: np.ndarray,
    hard_tokens_per_sample: np.ndarray,
    visual_attention_mass: np.ndarray | None = None,
    output_npz_path: str | os.PathLike | None = None,
) -> tuple[pathlib.Path, pathlib.Path]:
    """Save token_budget_<source>.json and .npz without overwriting existing files."""
    json_path = pathlib.Path(output_json_path)
    npz_path = json_path.with_suffix(".npz") if output_npz_path is None else pathlib.Path(output_npz_path)

    if json_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing budget report: {json_path}")
    if npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing budget cache: {npz_path}")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    npz_payload: dict[str, np.ndarray] = {
        "sample_indices": np.asarray(sample_indices, dtype=np.int64),
        "relevance": np.asarray(relevance, dtype=np.float32),
        "valid_mask": np.asarray(valid_mask, dtype=bool),
        "hard_tokens_per_sample": np.asarray(hard_tokens_per_sample, dtype=np.int64),
    }
    if visual_attention_mass is not None:
        npz_payload["visual_attention_mass"] = np.asarray(visual_attention_mass, dtype=np.float32)

    np.savez(npz_path, **npz_payload)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return json_path, npz_path


def resolve_budget_json_path(path: str | os.PathLike) -> pathlib.Path:
    target = pathlib.Path(path)
    if target.is_dir():
        return target / "token_budget_vit.json"
    return target


def load_budget(
    path: str | os.PathLike,
    *,
    expected_source: str | None = None,
    expected_config_name: str | None = None,
    expected_repo_id: str | None = None,
) -> dict[str, Any]:
    """Load and validate a token_budget_<source>.json report."""
    json_path = resolve_budget_json_path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"Token budget report not found at {json_path}")

    report = json.loads(json_path.read_text())
    if not isinstance(report, dict) or report.get("format_version") != TOKEN_BUDGET_FORMAT_VERSION:
        raise ValueError(f"Unsupported token budget format_version in {json_path}")

    source = report.get("attention_source")
    if source not in METHOD_BY_SOURCE:
        raise ValueError(f"Invalid attention_source '{source}' in {json_path}")
    if report.get("method") != METHOD_BY_SOURCE[source]:
        raise ValueError(
            f"Mismatched method '{report.get('method')}' for attention_source '{source}' in {json_path}"
        )

    if expected_source is not None and source != expected_source:
        raise ValueError(f"Budget attention_source '{source}' does not match expected '{expected_source}'")
    if expected_config_name is not None and report.get("config_name") != expected_config_name:
        raise ValueError(
            f"Budget config_name '{report.get('config_name')}' does not match expected '{expected_config_name}'"
        )
    if expected_repo_id is not None and report.get("repo_id") != expected_repo_id:
        raise ValueError(f"Budget repo_id '{report.get('repo_id')}' does not match expected '{expected_repo_id}'")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError(f"Missing summary section in {json_path}")

    for int_key in (
        "num_samples",
        "total_visual_tokens",
        "min_valid_visual_tokens",
        "max_valid_visual_tokens",
        "recommended_hard_tokens",
        "recommended_global_tokens",
        "recommended_total_tokens",
        "alignment",
    ):
        val = summary.get(int_key)
        if not isinstance(val, int) or val <= 0:
            raise ValueError(f"Summary field '{int_key}' must be a positive integer in {json_path}, got {val}")

    if summary["recommended_hard_tokens"] > summary["max_valid_visual_tokens"]:
        raise ValueError("recommended_hard_tokens exceeds max_valid_visual_tokens in summary")
    if summary["recommended_total_tokens"] != summary["recommended_hard_tokens"] + summary["recommended_global_tokens"]:
        raise ValueError("recommended_total_tokens does not equal H + G in summary")

    return report


@dataclasses.dataclass(frozen=True)
class ResolvedTrainingBudget:
    source_json_path: pathlib.Path
    source_npz_path: pathlib.Path | None
    report: dict[str, Any]
    summary: dict[str, Any]
    selection: dict[str, Any]
    hard_tokens: int
    soft_tokens: int


def _lookup_quantile_from_summary(quantiles_dict: dict[str, Any], q: float, default_q: float, default_val: int) -> int:
    if math.isclose(float(q), float(default_q), rel_tol=0.0, abs_tol=1e-9):
        return int(default_val)
    q_key = f"q_{float(q):.4f}"
    if q_key in quantiles_dict:
        return int(quantiles_dict[q_key])
    pct_map = {0.05: "p05", 0.25: "p25", 0.50: "p50", 0.75: "p75", 0.90: "p90", 0.95: "p95", 0.99: "p99"}
    for k_float, k_name in pct_map.items():
        if math.isclose(float(q), k_float, rel_tol=0.0, abs_tol=1e-9) and k_name in quantiles_dict:
            return int(quantiles_dict[k_name])
    raise ValueError(f"Quantile {q} is not available in cached JSON summary without NPZ cache")


def resolve_training_budget(
    budget_path: str | os.PathLike,
    *,
    expected_source: str | None = "vit",
    expected_config_name: str | None = None,
    expected_repo_id: str | None = None,
    coverage_target: float | None = None,
    sample_quantile: float | None = None,
    alignment: int | None = None,
    global_coverage_upper: float | None = None,
    global_min_tokens: int | None = None,
) -> ResolvedTrainingBudget:
    """Resolve H/G from a budget JSON + optional sibling NPZ cache."""
    json_path = resolve_budget_json_path(budget_path)
    report = load_budget(
        json_path,
        expected_source=expected_source,
        expected_config_name=expected_config_name,
        expected_repo_id=expected_repo_id,
    )
    orig_summary = report["summary"]

    c = float(orig_summary["coverage_target"] if coverage_target is None else coverage_target)
    q = float(orig_summary["sample_quantile"] if sample_quantile is None else sample_quantile)
    align = int(orig_summary["alignment"] if alignment is None else alignment)
    u = float(orig_summary.get("global_coverage_upper", 0.90) if global_coverage_upper is None else global_coverage_upper)
    g_min = int(orig_summary.get("global_min_tokens", 64) if global_min_tokens is None else global_min_tokens)

    npz_path = json_path.with_suffix(".npz")
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as data:
            relevance = data["relevance"]
            valid_mask = data["valid_mask"]
            vis_mass = data["visual_attention_mass"] if "visual_attention_mass" in data.files else None
        resolved_summary, _ = compute_token_budget_summary(
            relevance,
            valid_mask,
            coverage_target=c,
            sample_quantile=q,
            alignment=align,
            global_coverage_upper=u,
            global_min_tokens=g_min,
            visual_attention_mass=vis_mass,
        )
        used_npz: pathlib.Path | None = npz_path
    else:
        used_npz = None
        if orig_summary.get("global_token_method") != GLOBAL_TOKEN_METHOD:
            raise ValueError(
                f"Cannot migrate legacy global_token_method '{orig_summary.get('global_token_method')}' without NPZ cache"
            )
        if not math.isclose(c, float(orig_summary["coverage_target"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Cannot change coverage_target without sibling NPZ cache")
        if not math.isclose(u, float(orig_summary["global_coverage_upper"]), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Cannot change global_coverage_upper without sibling NPZ cache")
        if align <= 0 or g_min <= 0 or not (0.0 < q <= 1.0):
            raise ValueError("Invalid budget parameters")

        raw_hard_q = _lookup_quantile_from_summary(
            orig_summary.get("hard_token_quantiles", {}),
            q,
            float(orig_summary["sample_quantile"]),
            int(orig_summary["raw_hard_quantile_tokens"]),
        )
        raw_upper_q = _lookup_quantile_from_summary(
            orig_summary.get("upper_token_quantiles", {}),
            q,
            float(orig_summary["sample_quantile"]),
            int(orig_summary["raw_upper_quantile_tokens"]),
        )
        h_tokens = align_token_count(raw_hard_q, align)
        if h_tokens > int(orig_summary["max_valid_visual_tokens"]):
            raise ValueError(
                f"Aligned hard token budget ({h_tokens}) exceeds max valid visual tokens ({orig_summary['max_valid_visual_tokens']})"
            )
        raw_global_q = max(raw_upper_q - h_tokens, 0)
        floored_global = max(g_min, raw_global_q)
        g_tokens = align_token_count(floored_global, align)
        resolved_summary = {
            **orig_summary,
            "coverage_target": c,
            "sample_quantile": q,
            "alignment": align,
            "global_coverage_upper": u,
            "global_min_tokens": g_min,
            "raw_hard_quantile_tokens": raw_hard_q,
            "recommended_hard_tokens": h_tokens,
            "recommended_global_tokens": g_tokens,
            "recommended_total_tokens": h_tokens + g_tokens,
            "global_subtracted_hard_tokens": h_tokens,
            "raw_upper_quantile_tokens": raw_upper_q,
            "raw_global_quantile_tokens": raw_global_q,
            "floored_global_tokens": floored_global,
            "target_global_tokens": g_tokens,
        }

    h_tokens = int(resolved_summary["recommended_hard_tokens"])
    g_tokens = int(resolved_summary["recommended_global_tokens"])
    selection = {
        "attention_source": str(report["attention_source"]),
        "coverage_target": c,
        "sample_quantile": q,
        "global_coverage_upper": u,
        "global_min_tokens": g_min,
        "alignment": align,
        "hard_tokens": h_tokens,
        "soft_tokens": g_tokens,
        "total_tokens": h_tokens + g_tokens,
    }
    return ResolvedTrainingBudget(
        source_json_path=json_path.resolve(),
        source_npz_path=used_npz.resolve() if used_npz is not None else None,
        report=report,
        summary=resolved_summary,
        selection=selection,
        hard_tokens=h_tokens,
        soft_tokens=g_tokens,
    )


def _sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_or_verify_identical(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if _sha256_file(src) != _sha256_file(dst):
            raise ValueError(f"Refusing to overwrite existing calibration file with different content: {dst}")
        return
    shutil.copy2(src, dst)


def archive_training_budget(
    resolved: ResolvedTrainingBudget,
    checkpoint_dir: str | os.PathLike,
    *,
    run_output_dir: str | os.PathLike | None = None,
) -> pathlib.Path:
    """Archive original JSON, optional NPZ, and training_budget.json into checkpoint_dir/calibration (and RUN_OUTPUT_DIR/calibration)."""
    ckpt_cal_dir = pathlib.Path(checkpoint_dir) / "calibration"
    target_dirs = [ckpt_cal_dir]
    if run_output_dir is not None and str(run_output_dir).strip():
        out_cal_dir = pathlib.Path(run_output_dir) / "calibration"
        if out_cal_dir.resolve() != ckpt_cal_dir.resolve():
            target_dirs.append(out_cal_dir)

    archived_ckpt_json: pathlib.Path | None = None
    for cal_dir in target_dirs:
        cal_dir.mkdir(parents=True, exist_ok=True)
        dst_json = cal_dir / resolved.source_json_path.name
        _copy_or_verify_identical(resolved.source_json_path, dst_json)

        dst_npz: pathlib.Path | None = None
        if resolved.source_npz_path is not None and resolved.source_npz_path.exists():
            dst_npz = cal_dir / resolved.source_npz_path.name
            _copy_or_verify_identical(resolved.source_npz_path, dst_npz)

        training_budget_payload = {
            "format_version": TOKEN_BUDGET_FORMAT_VERSION,
            "source_budget_path": str(resolved.source_json_path),
            "archived_budget_path": str(dst_json),
            "archived_npz_path": str(dst_npz) if dst_npz is not None else None,
            "config_name": resolved.report.get("config_name"),
            "repo_id": resolved.report.get("repo_id"),
            "checkpoint": resolved.report.get("checkpoint"),
            "selection": resolved.selection,
            "summary": resolved.summary,
        }
        training_budget_path = cal_dir / "training_budget.json"
        serialized = json.dumps(training_budget_payload, indent=2, sort_keys=True) + "\n"
        if training_budget_path.exists():
            existing = json.loads(training_budget_path.read_text())
            if existing.get("selection") != training_budget_payload["selection"]:
                raise ValueError(
                    f"Refusing to overwrite existing training_budget.json with different selection at {training_budget_path}"
                )
        else:
            training_budget_path.write_text(serialized)

        if cal_dir == ckpt_cal_dir:
            archived_ckpt_json = dst_json

    assert archived_ckpt_json is not None
    return archived_ckpt_json


@dataclasses.dataclass
class CollectedAttentionBatch:
    raw_scores_by_source: dict[str, np.ndarray]
    normalized_by_source: dict[str, np.ndarray]
    visual_mass_by_source: dict[str, np.ndarray]
    valid_mask: np.ndarray


def collect_teacher_attention_batch(
    model: pi0_pytorch.PI0Pytorch,
    observation,
    *,
    sources: Sequence[str] = ("vit",),
    vit_layers: Sequence[int] | None = None,
    vlm_layers: Sequence[int] | None = None,
    action_layers: Sequence[int] | None = None,
    num_steps: int = DEFAULT_ACTION_NUM_STEPS,
    action_steps: Sequence[int] = DEFAULT_ACTION_STEPS,
    noise: torch.Tensor | None = None,
) -> CollectedAttentionBatch:
    """Collect teacher patch relevance scores [B, N] on a router-free PI0Pytorch model."""
    if getattr(model, "vision_router", None) is not None:
        raise ValueError("Teacher attention collection requires a router-free PI0Pytorch model (vision_router=None)")

    requested_sources = tuple(dict.fromkeys(sources))
    if not requested_sources:
        raise ValueError("At least one attention source must be requested")
    for src in requested_sources:
        if src not in METHOD_BY_SOURCE:
            raise ValueError(f"Unsupported attention source '{src}'")

    need_vit = "vit" in requested_sources
    need_vlm = "vlm" in requested_sources
    need_action = "action" in requested_sources

    vision_tower = model.paligemma_with_expert.paligemma.vision_tower
    vision_layers_list = vision_tower.vision_model.encoder.layers
    vlm_model = model.paligemma_with_expert.paligemma.language_model
    vlm_layers_list = vlm_model.layers
    expert_model = model.paligemma_with_expert.gemma_expert.model
    expert_layers_list = expert_model.layers

    resolved_vit_layers = (
        tuple(range(max(0, len(vision_layers_list) - 4), len(vision_layers_list)))
        if vit_layers is None
        else tuple(int(i) for i in vit_layers)
    )
    resolved_vlm_layers = (
        tuple(range(max(0, len(vlm_layers_list) - 4), len(vlm_layers_list)))
        if vlm_layers is None
        else tuple(int(i) for i in vlm_layers)
    )
    resolved_action_layers = (
        tuple(range(max(0, len(expert_layers_list) - 4), len(expert_layers_list)))
        if action_layers is None
        else tuple(int(i) for i in action_layers)
    )
    resolved_action_steps = tuple(int(s) for s in action_steps)
    if need_action:
        if num_steps <= 0:
            raise ValueError(f"num_steps must be > 0, got {num_steps}")
        if not resolved_action_steps or any(s < 0 or s >= num_steps for s in resolved_action_steps):
            raise ValueError(f"action_steps {resolved_action_steps} out of range for num_steps={num_steps}")

    was_training = model.training
    model.eval()

    vit_Configs = [vision_tower.config, vision_tower.vision_model.config]
    prev_vit_impls = [ getattr(c, "_attn_implementation", None) for c in vit_Configs ]
    prev_vit_out_attns = [ getattr(c, "output_attentions", False) for c in vit_Configs ]
    prev_vlm_impl = getattr(vlm_model.config, "_attn_implementation", None)
    prev_expert_impl = getattr(expert_model.config, "_attn_implementation", None)

    with ExitStack() as stack:
        def _restore():
            for c, impl, out_a in zip(vit_Configs, prev_vit_impls, prev_vit_out_attns, strict=True):
                c._attn_implementation = impl  # noqa: SLF001
                c.output_attentions = out_a
            vlm_model.config._attn_implementation = prev_vlm_impl  # noqa: SLF001
            expert_model.config._attn_implementation = prev_expert_impl  # noqa: SLF001
            if was_training:
                model.train()

        stack.callback(_restore)

        with torch.inference_mode():
            images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)  # noqa: SLF001
            images = [
                img.permute(0, 3, 1, 2) if (img.ndim == 4 and img.shape[1] != 3 and img.shape[-1] == 3) else img
                for img in images
            ]
            bsize = state.shape[0]
            device = state.device
            num_cameras = len(images)

            vit_records: dict[int, list[torch.Tensor]] = {l: [] for l in resolved_vit_layers}
            vlm_records: dict[int, torch.Tensor] = {}
            action_records: dict[tuple[int, int], torch.Tensor] = {}

            if need_vit:
                for c in vit_Configs:
                    c._attn_implementation = "eager"  # noqa: SLF001
                    c.output_attentions = True

                def _vit_pre_hook(_mod, args, kwargs):
                    new_kwargs = dict(kwargs)
                    new_kwargs["output_attentions"] = True
                    return args, new_kwargs

                for layer_idx in resolved_vit_layers:
                    attn_mod = vision_layers_list[layer_idx].self_attn
                    stack.callback(attn_mod.register_forward_pre_hook(_vit_pre_hook, with_kwargs=True).remove)

                    def _make_vit_hook(l_idx: int):
                        def _hook(_mod, _inp, output):
                            attn_weights = output[1]
                            if attn_weights is None:
                                raise RuntimeError(f"ViT layer {l_idx} did not return attention weights")
                            reduced = attn_weights.detach().float().mean(dim=(1, 2)).cpu()
                            vit_records[l_idx].append(reduced)
                            return (output[0], None)

                        return _hook

                    stack.callback(attn_mod.register_forward_hook(_make_vit_hook(layer_idx)).remove)

            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks
            )
            num_lang_tokens = int(lang_masks.shape[1])
            total_visual_tokens = int(prefix_embs.shape[1] - num_lang_tokens)
            patches_per_camera = total_visual_tokens // num_cameras

            valid_visual_mask_t = torch.cat(
                [m[:, None].expand(bsize, patches_per_camera) for m in img_masks],
                dim=1,
            )
            valid_visual_mask_cpu = valid_visual_mask_t.detach().cpu().numpy().astype(bool)
            lang_masks_cpu = lang_masks.detach().cpu()

            if need_vlm or need_action:
                vlm_model.config._attn_implementation = "eager"  # noqa: SLF001
                if need_vlm:
                    for layer_idx in resolved_vlm_layers:
                        attn_mod = vlm_layers_list[layer_idx].self_attn

                        def _make_vlm_hook(l_idx: int):
                            def _hook(_mod, _inp, output):
                                attn_weights = output[1]
                                if attn_weights is None:
                                    raise RuntimeError(f"VLM layer {l_idx} did not return attention weights")
                                prompt_to_vis = (
                                    attn_weights[:, :, total_visual_tokens : total_visual_tokens + num_lang_tokens, :total_visual_tokens]
                                    .detach()
                                    .float()
                                    .mean(dim=1)
                                    .cpu()
                                )
                                q_mask = lang_masks_cpu.to(dtype=torch.float32).unsqueeze(-1)
                                q_denom = q_mask.sum(dim=1)
                                if torch.any(q_denom <= 0):
                                    raise ValueError("Sample has zero valid prompt queries in VLM attention")
                                vlm_records[l_idx] = (prompt_to_vis * q_mask).sum(dim=1) / q_denom

                            return _hook

                        stack.callback(attn_mod.register_forward_hook(_make_vlm_hook(layer_idx)).remove)

                prefix_att_2d_masks = pi0_pytorch.make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
                prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
                prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)  # noqa: SLF001

                _, past_key_values = model.paligemma_with_expert.forward(
                    attention_mask=prefix_att_2d_masks_4d,
                    position_ids=prefix_position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, None],
                    use_cache=True,
                )

                if need_action:
                    expert_model.config._attn_implementation = "eager"  # noqa: SLF001
                    current_step_holder = [-1]
                    action_horizon = int(model.config.action_horizon)
                    selected_step_set = set(resolved_action_steps)

                    for layer_idx in resolved_action_layers:
                        attn_mod = expert_layers_list[layer_idx].self_attn

                        def _make_action_hook(l_idx: int):
                            def _hook(_mod, _inp, output):
                                step_idx = current_step_holder[0]
                                if step_idx not in selected_step_set:
                                    return
                                attn_weights = output[1]
                                if attn_weights is None:
                                    raise RuntimeError(f"Action expert layer {l_idx} did not return attention weights")
                                act_to_vis = (
                                    attn_weights[:, :, -action_horizon:, :total_visual_tokens]
                                    .detach()
                                    .float()
                                    .mean(dim=(1, 2))
                                    .cpu()
                                )
                                action_records[(step_idx, l_idx)] = act_to_vis

                            return _hook

                        stack.callback(attn_mod.register_forward_hook(_make_action_hook(layer_idx)).remove)

                    if noise is None:
                        actions_shape = (bsize, model.config.action_horizon, model.config.action_dim)
                        x_t = model.sample_noise(actions_shape, device)
                    else:
                        x_t = noise.to(device=device, dtype=torch.float32)

                    dt = torch.tensor(-1.0 / float(num_steps), dtype=torch.float32, device=device)
                    time_val = torch.tensor(1.0, dtype=torch.float32, device=device)
                    for step_idx in range(num_steps):
                        current_step_holder[0] = step_idx
                        expanded_time = time_val.expand(bsize)
                        v_t = model.denoise_step(
                            state,
                            prefix_pad_masks,
                            past_key_values,
                            x_t,
                            expanded_time,
                        )
                        x_t = x_t + dt * v_t
                        time_val = time_val + dt

    raw_scores_by_source: dict[str, np.ndarray] = {}
    normalized_by_source: dict[str, np.ndarray] = {}
    visual_mass_by_source: dict[str, np.ndarray] = {}

    if need_vit:
        per_layer_vit = []
        for layer_idx in resolved_vit_layers:
            cam_tensors = vit_records[layer_idx]
            if len(cam_tensors) != num_cameras:
                raise RuntimeError(
                    f"Expected {num_cameras} camera attention tensors for ViT layer {layer_idx}, got {len(cam_tensors)}"
                )
            per_layer_vit.append(torch.cat(cam_tensors, dim=1))
        raw_vit = torch.stack(per_layer_vit, dim=0).mean(dim=0).numpy()
        raw_vit = np.where(valid_visual_mask_cpu, raw_vit, 0.0).astype(np.float32)
        norm_vit, _, mass_vit = validate_And_normalize_relevance(raw_vit, valid_visual_mask_cpu)
        raw_scores_by_source["vit"] = raw_vit
        normalized_by_source["vit"] = norm_vit.astype(np.float32)
        visual_mass_by_source["vit"] = mass_vit.astype(np.float32)

    if need_vlm:
        per_layer_vlm = [vlm_records[l_idx] for l_idx in resolved_vlm_layers]
        raw_vlm = torch.stack(per_layer_vlm, dim=0).mean(dim=0).numpy()
        raw_vlm = np.where(valid_visual_mask_cpu, raw_vlm, 0.0).astype(np.float32)
        norm_vlm, _, mass_vlm = validate_And_normalize_relevance(raw_vlm, valid_visual_mask_cpu)
        raw_scores_by_source["vlm"] = raw_vlm
        normalized_by_source["vlm"] = norm_vlm.astype(np.float32)
        visual_mass_by_source["vlm"] = mass_vlm.astype(np.float32)

    if need_action:
        per_step_layer_action = [
            action_records[(step_idx, l_idx)]
            for step_idx in resolved_action_steps
            for l_idx in resolved_action_layers
        ]
        raw_action = torch.stack(per_step_layer_action, dim=0).mean(dim=0).numpy()
        raw_action = np.where(valid_visual_mask_cpu, raw_action, 0.0).astype(np.float32)
        norm_action, _, mass_action = validate_And_normalize_relevance(raw_action, valid_visual_mask_cpu)
        raw_scores_by_source["action"] = raw_action
        normalized_by_source["action"] = norm_action.astype(np.float32)
        visual_mass_by_source["action"] = mass_action.astype(np.float32)

    return CollectedAttentionBatch(
        raw_scores_by_source=raw_scores_by_source,
        normalized_by_source=normalized_by_source,
        visual_mass_by_source=visual_mass_by_source,
        valid_mask=valid_visual_mask_cpu,
    )
