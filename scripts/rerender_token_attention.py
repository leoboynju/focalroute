#!/usr/bin/env python3
"""Re-render teacher attention videos and recompute token budgets at a new coverage threshold from cached NPZ files."""

import argparse
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
from typing import Any

import jax
import numpy as np
import torch
import tqdm

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.models_pytorch import preprocessing_pytorch as _preprocessing
from openpi.training import attention_visualization as _vis
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training import token_budget

ALL_SOURCES: tuple[str, ...] = ("vit", "vlm", "action")


def _collate_samples(items: Sequence[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def build_transformed_dataset(train_config: _config.TrainConfig):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    transformed = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
    return transformed, data_config


def rerender_from_cache(
    *,
    source_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    coverage: float,
    expected_checkpoint_dir: str | pathlib.Path | None = None,
    sample_quantile: float | None = None,
    alignment: int | None = None,
    global_coverage_upper: float | None = None,
    global_min_tokens: int | None = None,
    fps: int | None = None,
    color_max_multiplier: float | None = None,
    assets_base_dir: str | None = None,
    dataset=None,
) -> dict[str, Any]:
    if not (0.0 < float(coverage) <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")

    src_dir = pathlib.Path(source_dir)
    out_dir = pathlib.Path(output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")

    manifest_path = src_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest.json not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("status") != "completed":
        raise ValueError(f"Source manifest is not marked completed in {manifest_path}")
    if manifest.get("router_free") is not True:
        raise ValueError(f"Source manifest is not router_free in {manifest_path}")

    manifest_ckpt = pathlib.Path(manifest["checkpoint_dir"]).resolve()
    ckpt_to_verify = manifest_ckpt if expected_checkpoint_dir is None else pathlib.Path(expected_checkpoint_dir).resolve()
    if expected_checkpoint_dir is not None and ckpt_to_verify != manifest_ckpt:
        raise ValueError(
            f"expected_checkpoint_dir ({ckpt_to_verify}) does not match manifest checkpoint_dir ({manifest_ckpt})"
        )

    if pi0_pytorch.checkpoint_has_vision_router(ckpt_to_verify):
        raise ValueError(f"Checkpoint {ckpt_to_verify} contains a VisionRouter; expected router-free teacher")

    config_name = str(manifest["config_name"])
    train_config = _config.get_config(config_name)
    if assets_base_dir is not None:
        train_config = dataclasses.replace(train_config, assets_base_dir=str(assets_base_dir))
    if not isinstance(train_config.model, _pi0_config.Pi0Config):
        raise ValueError(f"Config '{config_name}' model is not a Pi0Config")

    teacher_model_cfg = dataclasses.replace(
        train_config.model,
        pytorch_compile_mode=None,
        vision_router_hard_enabled=False,
        vision_router_soft_enabled=False,
    )
    expected_model_config_dict = pi0_pytorch.build_model_config_dict(teacher_model_cfg)
    if manifest.get("model_config") != expected_model_config_dict:
        raise ValueError("Source manifest model_config dictionary does not match current teacher model configuration")

    actual_fingerprint = _vis.compute_action_out_proj_fingerprint(ckpt_to_verify)
    if manifest.get("action_out_proj_fingerprint") != actual_fingerprint:
        raise ValueError(
            f"Checkpoint action_out_proj fingerprint mismatch: manifest={manifest.get('action_out_proj_fingerprint')} vs actual={actual_fingerprint}"
        )

    if dataset is None:
        dataset, _ = build_transformed_dataset(train_config)

    resolved_quantile = float(manifest["sample_quantile"] if sample_quantile is None else sample_quantile)
    resolved_alignment = int(manifest["alignment"] if alignment is None else alignment)
    resolved_upper = float(manifest["global_coverage_upper"] if global_coverage_upper is None else global_coverage_upper)
    resolved_min_tokens = int(manifest["global_min_tokens"] if global_min_tokens is None else global_min_tokens)
    resolved_fps = int(manifest.get("fps", 10) if fps is None else fps)
    resolved_color_mult = float(
        manifest.get("color_max_multiplier", 8.0) if color_max_multiplier is None else color_max_multiplier
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    episodes_dir = out_dir / "episodes"
    videos_dir = out_dir / "videos"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    merged_indices: list[np.ndarray] = []
    merged_valid_masks: list[np.ndarray] = []
    merged_raw_by_source: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
    merged_mass_by_source: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
    new_episode_manifests: list[dict[str, Any]] = []

    for ep_info in tqdm.tqdm(manifest["episodes"], desc="Re-rendering episodes"):
        ep_id = int(ep_info["episode_id"])
        src_ep_npz = src_dir / ep_info["cache_npz"]
        if not src_ep_npz.exists():
            raise FileNotFoundError(f"Cached episode NPZ not found at {src_ep_npz}")

        with np.load(src_ep_npz, allow_pickle=False) as ep_data:
            ds_indices = np.asarray(ep_data["dataset_indices"], dtype=np.int64)
            valid_mask = np.asarray(ep_data["valid_mask"], dtype=bool)
            raw_by_s = {s: np.asarray(ep_data[f"{s}_raw_scores"], dtype=np.float32) for s in ALL_SOURCES}
            rel_by_s = {s: np.asarray(ep_data[f"{s}_relevance"], dtype=np.float32) for s in ALL_SOURCES}
            mass_by_s = {s: np.asarray(ep_data[f"{s}_visual_mass"], dtype=np.float32) for s in ALL_SOURCES}

        items = [dataset[int(idx)] for idx in ds_indices]
        collated = _collate_samples(items)
        tensor_batch = jax.tree.map(lambda x: torch.as_tensor(x, device="cpu"), collated)
        observation = _model.Observation.from_dict(tensor_batch)
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=False)
        pre_imgs_np = [img.detach().cpu().numpy() for img in processed.images.values()]
        pre_masks_np = [m.detach().cpu().numpy().astype(bool) for m in processed.image_masks.values()]

        total_vis = valid_mask.shape[1]
        patches_per_cam = total_vis // len(pre_imgs_np)
        grid_side = int(round(patches_per_cam**0.5))
        layout = _vis.VisualTokenLayout(
            num_cameras=len(pre_imgs_np),
            grid_height=grid_side,
            grid_width=grid_side,
        )

        ep_sel_mask: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
        ep_sel_counts: dict[str, list[int]] = {s: [] for s in ALL_SOURCES}
        ep_frames_rgb: list[np.ndarray] = []

        for f_idx in range(len(ds_indices)):
            cam_imgs_f = [cam_arr[f_idx] for cam_arr in pre_imgs_np]
            cam_masks_f = [bool(mask_arr[f_idx]) for mask_arr in pre_masks_np]
            sels_f: dict[str, _vis.MinimalCoverageSelection] = {}
            for s in ALL_SOURCES:
                sel = _vis.select_minimal_coverage_set(raw_by_s[s][f_idx], valid_mask[f_idx], coverage)
                sels_f[s] = sel
                ep_sel_mask[s].append(sel.selected_mask)
                ep_sel_counts[s].append(sel.num_selected)

            frame_rgb = _vis.render_attention_frame(
                cam_imgs_f,
                cam_masks_f,
                sels_f,
                episode_id=ep_id,
                frame_idx=f_idx,
                coverage=coverage,
                color_max_multiplier=resolved_color_mult,
                camera_names=_preprocessing.IMAGE_KEYS,
                layout=layout,
            )
            ep_frames_rgb.append(frame_rgb)

        video_path = videos_dir / f"episode_{ep_id:04d}.mp4"
        encoding_info = _vis.encode_h264_mp4(ep_frames_rgb, video_path, fps=resolved_fps)

        dst_ep_npz = episodes_dir / f"episode_{ep_id:04d}.npz"
        npz_dict: dict[str, Any] = {
            "episode_id": np.asarray(ep_id, dtype=np.int64),
            "dataset_indices": ds_indices,
            "sample_indices": ds_indices,
            "frame_indices": np.arange(len(ds_indices), dtype=np.int64),
            "valid_mask": valid_mask,
        }
        per_source_stats: dict[str, Any] = {}
        for s in ALL_SOURCES:
            sel_mask_arr = np.stack(ep_sel_mask[s], axis=0).astype(bool)
            sel_counts_arr = np.asarray(ep_sel_counts[s], dtype=np.int64)
            npz_dict[f"{s}_raw_scores"] = raw_by_s[s]
            npz_dict[f"{s}_relevance"] = rel_by_s[s]
            npz_dict[f"{s}_visual_mass"] = mass_by_s[s]
            npz_dict[f"{s}_selected_mask"] = sel_mask_arr
            npz_dict[f"{s}_selected_counts"] = sel_counts_arr

            merged_raw_by_source[s].append(raw_by_s[s])
            merged_mass_by_source[s].append(mass_by_s[s])
            per_source_stats[s] = {
                "mean_selected_tokens": float(np.mean(sel_counts_arr)),
                "p95_selected_tokens": int(np.quantile(sel_counts_arr, 0.95, method="higher")),
            }

        np.savez(dst_ep_npz, **npz_dict)
        merged_indices.append(ds_indices)
        merged_valid_masks.append(valid_mask)
        new_episode_manifests.append(
            {
                "episode_id": ep_id,
                "num_frames": int(len(ds_indices)),
                "dataset_indices": [int(x) for x in ds_indices],
                "cache_npz": str(dst_ep_npz.relative_to(out_dir)),
                "video_path": str(video_path.relative_to(out_dir)),
                "encoding": encoding_info,
                "per_source_stats": per_source_stats,
            }
        )

    all_sample_indices = np.concatenate(merged_indices, axis=0)
    all_valid_mask = np.concatenate(merged_valid_masks, axis=0)
    layers_by_source = manifest["layers"]
    repo_id = manifest.get("repo_id")

    budgets_summary: dict[str, Any] = {}
    budget_files: dict[str, dict[str, str]] = {}
    for s in ALL_SOURCES:
        raw_all = np.concatenate(merged_raw_by_source[s], axis=0)
        mass_all = np.concatenate(merged_mass_by_source[s], axis=0)
        summary_s, hard_per_sample_s = token_budget.compute_token_budget_summary(
            raw_all,
            all_valid_mask,
            coverage_target=coverage,
            sample_quantile=resolved_quantile,
            alignment=resolved_alignment,
            global_coverage_upper=resolved_upper,
            global_min_tokens=resolved_min_tokens,
            visual_attention_mass=mass_all,
        )
        report_s = token_budget.build_budget_report(
            config_name=config_name,
            repo_id=repo_id,
            checkpoint=str(ckpt_to_verify),
            attention_source=s,
            model_config_dict=expected_model_config_dict,
            summary=summary_s,
            layers=layers_by_source[s],
            seed=int(manifest.get("seed", 42)),
            sample_indices_source="episode_trajectories",
            num_steps=int(manifest.get("num_steps", 10)) if s == "action" else None,
            action_steps=manifest.get("action_steps") if s == "action" else None,
            action_times=manifest.get("action_times") if s == "action" else None,
        )
        json_name, _ = token_budget.default_budget_filenames(s)
        saved_json, saved_npz = token_budget.save_budget_artifacts(
            out_dir / json_name,
            report_s,
            sample_indices=all_sample_indices,
            relevance=raw_all,
            valid_mask=all_valid_mask,
            hard_tokens_per_sample=hard_per_sample_s,
            visual_attention_mass=mass_all,
        )
        budgets_summary[s] = summary_s
        budget_files[s] = {
            "json": str(saved_json.relative_to(out_dir)),
            "npz": str(saved_npz.relative_to(out_dir)),
        }

    new_manifest = {
        **manifest,
        "source_dir": str(src_dir.resolve()),
        "coverage": float(coverage),
        "sample_quantile": resolved_quantile,
        "alignment": resolved_alignment,
        "global_coverage_upper": resolved_upper,
        "global_min_tokens": resolved_min_tokens,
        "fps": resolved_fps,
        "color_max_multiplier": resolved_color_mult,
        "episodes": new_episode_manifests,
        "budgets": budgets_summary,
        "budget_files": budget_files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(new_manifest, indent=2, sort_keys=True) + "\n")
    logging.info("Re-rendered %d episodes at coverage=%.2f -> %s", len(new_episode_manifests), coverage, out_dir)
    return new_manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-render cached teacher attention at a new coverage threshold.")
    parser.add_argument("--source-dir", "--source_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--coverage", "--coverage-target", "--coverage_target", type=float, required=True)
    parser.add_argument("--expected-checkpoint-dir", "--expected_checkpoint_dir", default=None)
    parser.add_argument("--sample-quantile", "--sample_quantile", type=float, default=None)
    parser.add_argument("--alignment", type=int, default=None)
    parser.add_argument("--global-coverage-upper", "--global_coverage_upper", type=float, default=None)
    parser.add_argument("--global-min-tokens", "--global_min_tokens", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--color-max-multiplier", "--color_max_multiplier", type=float, default=None)
    parser.add_argument("--assets-base-dir", "--assets_base_dir", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    rerender_from_cache(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        coverage=args.coverage,
        expected_checkpoint_dir=args.expected_checkpoint_dir,
        sample_quantile=args.sample_quantile,
        alignment=args.alignment,
        global_coverage_upper=args.global_coverage_upper,
        global_min_tokens=args.global_min_tokens,
        fps=args.fps,
        color_max_multiplier=args.color_max_multiplier,
        assets_base_dir=args.assets_base_dir,
    )


if __name__ == "__main__":
    main()
