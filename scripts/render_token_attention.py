#!/usr/bin/env python3
"""Render teacher attention heatmaps and minimal-coverage patch sets for ViT, VLM, and Action Expert."""

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


def _unwrap_base_dataset(dataset):
    curr = dataset
    visited = set()
    while hasattr(curr, "_dataset") and id(curr) not in visited:
        visited.add(id(curr))
        curr = curr._dataset  # noqa: SLF001
    return curr


def resolve_episode_indices(
    dataset,
    *,
    episode_ids: Sequence[int] | None = None,
    num_episodes: int = 10,
    seed: int = 42,
    frame_stride: int = 1,
) -> list[tuple[int, np.ndarray]]:
    """Resolve (episode_id, dataset_indices) pairs from a LeRobot or test dataset."""
    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be > 0, got {frame_stride}")

    base = _unwrap_base_dataset(dataset)
    ep_map: dict[int, np.ndarray] = {}

    if hasattr(base, "episode_data_index") and isinstance(base.episode_data_index, dict):
        from_arr = np.asarray(base.episode_data_index["from"], dtype=np.int64)
        to_arr = np.asarray(base.episode_data_index["to"], dtype=np.int64)
        for ep_id in range(len(from_arr)):
            ep_map[int(ep_id)] = np.arange(int(from_arr[ep_id]), int(to_arr[ep_id]), frame_stride, dtype=np.int64)
    elif hasattr(base, "episode_to_indices") and isinstance(base.episode_to_indices, dict):
        for ep_id, idxs in base.episode_to_indices.items():
            ep_map[int(ep_id)] = np.asarray(idxs, dtype=np.int64)[::frame_stride]
    else:
        total_len = len(dataset)
        if episode_ids:
            chunk = max(1, total_len // len(episode_ids))
            for i, ep_id in enumerate(episode_ids):
                start = min(i * chunk, total_len - 1)
                end = total_len if i == len(episode_ids) - 1 else min((i + 1) * chunk, total_len)
                ep_map[int(ep_id)] = np.arange(start, max(start + 1, end), frame_stride, dtype=np.int64)
        else:
            count = max(1, min(int(num_episodes), total_len))
            splits = np.array_split(np.arange(total_len, dtype=np.int64), count)
            for ep_id, split in enumerate(splits):
                if len(split) > 0:
                    ep_map[int(ep_id)] = split[::frame_stride]

    if episode_ids is not None and len(episode_ids) > 0:
        chosen_ids = [int(e) for e in episode_ids]
        for ep_id in chosen_ids:
            if ep_id not in ep_map or len(ep_map[ep_id]) == 0:
                raise ValueError(f"Episode {ep_id} not found or empty in dataset")
    else:
        available = sorted(ep_id for ep_id, idxs in ep_map.items() if len(idxs) > 0)
        if not available:
            raise ValueError("No non-empty episodes found in dataset")
        rng = np.random.default_rng(seed)
        take = min(int(num_episodes), len(available))
        chosen_ids = [int(x) for x in rng.choice(available, size=take, replace=False)]

    return [(ep_id, ep_map[ep_id]) for ep_id in chosen_ids]


def build_transformed_dataset(train_config: _config.TrainConfig):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    transformed = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
    return transformed, data_config


def make_episode_noise(
    *,
    seed: int,
    episode_id: int,
    action_horizon: int,
    action_dim: int,
) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) * 100_003 + int(episode_id))
    return torch.randn((1, int(action_horizon), int(action_dim)), generator=gen, dtype=torch.float32)


def render_episodes(
    *,
    checkpoint_dir: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    config_name: str = "pi05_libero",
    num_episodes: int = 10,
    episode_ids: Sequence[int] | None = None,
    batch_size: int = 2,
    frame_stride: int = 1,
    fps: int = 10,
    seed: int = 42,
    coverage: float = 0.80,
    sample_quantile: float = 0.95,
    alignment: int = 32,
    global_coverage_upper: float = 0.90,
    global_min_tokens: int = 64,
    color_max_multiplier: float = 8.0,
    vit_layers: Sequence[int] | None = None,
    vlm_layers: Sequence[int] | None = None,
    action_layers: Sequence[int] | None = None,
    num_steps: int = token_budget.DEFAULT_ACTION_NUM_STEPS,
    action_steps: Sequence[int] = token_budget.DEFAULT_ACTION_STEPS,
    assets_base_dir: str | None = None,
    device: str | None = None,
    dataset=None,
    repo_id_override: str | None = None,
) -> dict[str, Any]:
    ckpt_path = pathlib.Path(checkpoint_dir).resolve()
    out_dir = pathlib.Path(output_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {out_dir}")
    if pi0_pytorch.checkpoint_has_vision_router(ckpt_path):
        raise ValueError(f"render_token_attention requires a router-free teacher checkpoint, got {ckpt_path}")

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
    train_config = dataclasses.replace(train_config, model=teacher_model_cfg, seed=int(seed))

    if dataset is None:
        dataset, data_config = build_transformed_dataset(train_config)
        repo_id = data_config.repo_id if repo_id_override is None else repo_id_override
    else:
        repo_id = getattr(train_config.data, "repo_id", None) if repo_id_override is None else repo_id_override

    resolved_device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = teacher_model_cfg.load_pytorch(train_config, str(ckpt_path)).to(resolved_device)
    if getattr(model, "vision_router", None) is not None:
        raise ValueError("Loaded teacher model unexpectedly constructed a VisionRouter")
    model.eval()

    vit_total = len(model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers)
    vlm_total = len(model.paligemma_with_expert.paligemma.language_model.layers)
    act_total = len(model.paligemma_with_expert.gemma_expert.model.layers)
    resolved_vit_layers = (
        tuple(range(max(0, vit_total - 4), vit_total)) if vit_layers is None else tuple(int(x) for x in vit_layers)
    )
    resolved_vlm_layers = (
        tuple(range(max(0, vlm_total - 4), vlm_total)) if vlm_layers is None else tuple(int(x) for x in vlm_layers)
    )
    resolved_action_layers = (
        tuple(range(max(0, act_total - 4), act_total))
        if action_layers is None
        else tuple(int(x) for x in action_layers)
    )
    resolved_action_steps = tuple(int(s) for s in action_steps)
    action_times = [round(1.0 - float(s) / float(num_steps), 6) for s in resolved_action_steps]

    episodes_plan = resolve_episode_indices(
        dataset,
        episode_ids=episode_ids,
        num_episodes=num_episodes,
        seed=seed,
        frame_stride=frame_stride,
    )

    out_dir.mkdir(parents=True, exist_ok=False)
    episodes_dir = out_dir / "episodes"
    videos_dir = out_dir / "videos"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    merged_indices: list[np.ndarray] = []
    merged_valid_masks: list[np.ndarray] = []
    merged_raw_by_source: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
    merged_rel_by_source: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
    merged_mass_by_source: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
    episode_manifests: list[dict[str, Any]] = []

    for ep_id, ds_indices in episodes_plan:
        ep_noise_single = make_episode_noise(
            seed=seed,
            episode_id=ep_id,
            action_horizon=teacher_model_cfg.action_horizon,
            action_dim=teacher_model_cfg.action_dim,
        )
        ep_raw: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
        ep_rel: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
        ep_mass: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
        ep_sel_mask: dict[str, list[np.ndarray]] = {s: [] for s in ALL_SOURCES}
        ep_sel_counts: dict[str, list[int]] = {s: [] for s in ALL_SOURCES}
        ep_valid_masks: list[np.ndarray] = []
        ep_frames_rgb: list[np.ndarray] = []

        for start in tqdm.trange(0, len(ds_indices), batch_size, desc=f"Episode {ep_id}"):
            batch_ds_idx = ds_indices[start : start + batch_size]
            items = [dataset[int(idx)] for idx in batch_ds_idx]
            collated = _collate_samples(items)
            tensor_batch = jax.tree.map(lambda x: torch.as_tensor(x, device=resolved_device), collated)
            observation = _model.Observation.from_dict(tensor_batch)

            bsize = int(observation.state.shape[0])
            batch_noise = ep_noise_single.expand(bsize, -1, -1).to(resolved_device)

            collected = token_budget.collect_teacher_attention_batch(
                model,
                observation,
                sources=ALL_SOURCES,
                vit_layers=resolved_vit_layers,
                vlm_layers=resolved_vlm_layers,
                action_layers=resolved_action_layers,
                num_steps=num_steps,
                action_steps=resolved_action_steps,
                noise=batch_noise,
            )

            with torch.inference_mode():
                pre_imgs, pre_masks, _, _, _ = model._preprocess_observation(observation, train=False)  # noqa: SLF001
                pre_imgs_np = [img.detach().cpu().numpy() for img in pre_imgs]
                pre_masks_np = [m.detach().cpu().numpy().astype(bool) for m in pre_masks]

            total_vis = collected.valid_mask.shape[1]
            patches_per_cam = total_vis // len(pre_imgs_np)
            grid_side = int(round(patches_per_cam**0.5))
            layout = _vis.VisualTokenLayout(
                num_cameras=len(pre_imgs_np),
                grid_height=grid_side,
                grid_width=grid_side,
            )

            ep_valid_masks.append(collected.valid_mask)
            for s in ALL_SOURCES:
                ep_raw[s].append(collected.raw_scores_by_source[s])
                ep_rel[s].append(collected.normalized_by_source[s])
                ep_mass[s].append(collected.visual_mass_by_source[s])

            for b_idx in range(bsize):
                frame_offset = start + b_idx
                cam_imgs_b = [cam_arr[b_idx] for cam_arr in pre_imgs_np]
                cam_masks_b = [bool(mask_arr[b_idx]) for mask_arr in pre_masks_np]
                sels_b: dict[str, _vis.MinimalCoverageSelection] = {}
                for s in ALL_SOURCES:
                    sel = _vis.select_minimal_coverage_set(
                        collected.raw_scores_by_source[s][b_idx],
                        collected.valid_mask[b_idx],
                        coverage,
                    )
                    sels_b[s] = sel
                    ep_sel_mask[s].append(sel.selected_mask)
                    ep_sel_counts[s].append(sel.num_selected)

                frame_rgb = _vis.render_attention_frame(
                    cam_imgs_b,
                    cam_masks_b,
                    sels_b,
                    episode_id=ep_id,
                    frame_idx=frame_offset,
                    coverage=coverage,
                    color_max_multiplier=color_max_multiplier,
                    camera_names=_preprocessing.IMAGE_KEYS,
                    layout=layout,
                )
                ep_frames_rgb.append(frame_rgb)

        video_path = videos_dir / f"episode_{ep_id:04d}.mp4"
        encoding_info = _vis.encode_h264_mp4(ep_frames_rgb, video_path, fps=fps)

        ep_valid_arr = np.concatenate(ep_valid_masks, axis=0)
        ep_npz_path = episodes_dir / f"episode_{ep_id:04d}.npz"
        npz_dict: dict[str, Any] = {
            "episode_id": np.asarray(ep_id, dtype=np.int64),
            "dataset_indices": np.asarray(ds_indices, dtype=np.int64),
            "sample_indices": np.asarray(ds_indices, dtype=np.int64),
            "frame_indices": np.arange(len(ds_indices), dtype=np.int64),
            "valid_mask": ep_valid_arr.astype(bool),
        }
        per_source_stats: dict[str, Any] = {}
        for s in ALL_SOURCES:
            raw_arr = np.concatenate(ep_raw[s], axis=0).astype(np.float32)
            rel_arr = np.concatenate(ep_rel[s], axis=0).astype(np.float32)
            mass_arr = np.concatenate(ep_mass[s], axis=0).astype(np.float32)
            sel_mask_arr = np.stack(ep_sel_mask[s], axis=0).astype(bool)
            sel_counts_arr = np.asarray(ep_sel_counts[s], dtype=np.int64)

            npz_dict[f"{s}_raw_scores"] = raw_arr
            npz_dict[f"{s}_relevance"] = rel_arr
            npz_dict[f"{s}_visual_mass"] = mass_arr
            npz_dict[f"{s}_selected_mask"] = sel_mask_arr
            npz_dict[f"{s}_selected_counts"] = sel_counts_arr

            merged_raw_by_source[s].append(raw_arr)
            merged_rel_by_source[s].append(rel_arr)
            merged_mass_by_source[s].append(mass_arr)
            per_source_stats[s] = {
                "mean_selected_tokens": float(np.mean(sel_counts_arr)),
                "p95_selected_tokens": int(np.quantile(sel_counts_arr, 0.95, method="higher")),
            }

        np.savez(ep_npz_path, **npz_dict)
        merged_indices.append(np.asarray(ds_indices, dtype=np.int64))
        merged_valid_masks.append(ep_valid_arr)

        episode_manifests.append(
            {
                "episode_id": int(ep_id),
                "num_frames": int(len(ds_indices)),
                "dataset_indices": [int(x) for x in ds_indices],
                "cache_npz": str(ep_npz_path.relative_to(out_dir)),
                "video_path": str(video_path.relative_to(out_dir)),
                "encoding": encoding_info,
                "per_source_stats": per_source_stats,
            }
        )

    all_sample_indices = np.concatenate(merged_indices, axis=0)
    all_valid_mask = np.concatenate(merged_valid_masks, axis=0)
    model_config_dict = pi0_pytorch.build_model_config_dict(teacher_model_cfg)
    layers_by_source = {
        "vit": list(resolved_vit_layers),
        "vlm": list(resolved_vlm_layers),
        "action": list(resolved_action_layers),
    }

    budgets_summary: dict[str, Any] = {}
    budget_files: dict[str, dict[str, str]] = {}
    for s in ALL_SOURCES:
        raw_all = np.concatenate(merged_raw_by_source[s], axis=0)
        mass_all = np.concatenate(merged_mass_by_source[s], axis=0)
        summary_s, hard_per_sample_s = token_budget.compute_token_budget_summary(
            raw_all,
            all_valid_mask,
            coverage_target=coverage,
            sample_quantile=sample_quantile,
            alignment=alignment,
            global_coverage_upper=global_coverage_upper,
            global_min_tokens=global_min_tokens,
            visual_attention_mass=mass_all,
        )
        report_s = token_budget.build_budget_report(
            config_name=config_name,
            repo_id=repo_id,
            checkpoint=str(ckpt_path),
            attention_source=s,
            model_config_dict=model_config_dict,
            summary=summary_s,
            layers=layers_by_source[s],
            seed=seed,
            sample_indices_source="episode_trajectories",
            num_steps=num_steps if s == "action" else None,
            action_steps=resolved_action_steps if s == "action" else None,
            action_times=action_times if s == "action" else None,
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

    manifest = {
        "format_version": 1,
        "status": "completed",
        "router_free": True,
        "config_name": str(config_name),
        "repo_id": repo_id,
        "checkpoint_dir": str(ckpt_path),
        "model_config": model_config_dict,
        "action_out_proj_fingerprint": _vis.compute_action_out_proj_fingerprint(ckpt_path),
        "seed": int(seed),
        "fps": int(fps),
        "frame_stride": int(frame_stride),
        "color_max_multiplier": float(color_max_multiplier),
        "coverage": float(coverage),
        "sample_quantile": float(sample_quantile),
        "alignment": int(alignment),
        "global_coverage_upper": float(global_coverage_upper),
        "global_min_tokens": int(global_min_tokens),
        "layers": layers_by_source,
        "num_steps": int(num_steps),
        "action_steps": list(resolved_action_steps),
        "action_times": action_times,
        "encoding": {
            "codec": "h264",
            "codec_tag": "avc1",
            "pix_fmt": "yuv420p",
            "faststart": True,
            "fps": int(fps),
        },
        "episode_ids": [int(ep_id) for ep_id, _ in episodes_plan],
        "num_episodes": len(episodes_plan),
        "total_frames": int(len(all_sample_indices)),
        "episodes": episode_manifests,
        "budgets": budgets_summary,
        "budget_files": budget_files,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logging.info("Rendered %d episodes (%d frames) -> %s", len(episodes_plan), len(all_sample_indices), out_dir)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ViT/VLM/Action teacher attention videos and budget caches.")
    parser.add_argument("--config-name", "--config_name", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        default="models/pi05_libero_pytorch",
    )
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--num-episodes", "--num_episodes", type=int, default=10)
    parser.add_argument("--episode-ids", "--episode_ids", type=int, nargs="+", default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2)
    parser.add_argument("--frame-stride", "--frame_stride", type=int, default=1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage", "--coverage-target", "--coverage_target", type=float, default=0.80)
    parser.add_argument("--sample-quantile", "--sample_quantile", type=float, default=0.95)
    parser.add_argument("--alignment", type=int, default=32)
    parser.add_argument("--global-coverage-upper", "--global_coverage_upper", type=float, default=0.90)
    parser.add_argument("--global-min-tokens", "--global_min_tokens", type=int, default=64)
    parser.add_argument("--color-max-multiplier", "--color_max_multiplier", type=float, default=8.0)
    parser.add_argument("--vit-layers", "--vit_layers", type=int, nargs="+", default=None)
    parser.add_argument("--vlm-layers", "--vlm_layers", type=int, nargs="+", default=None)
    parser.add_argument("--action-layers", "--action_layers", type=int, nargs="+", default=None)
    parser.add_argument("--num-steps", "--num_steps", type=int, default=token_budget.DEFAULT_ACTION_NUM_STEPS)
    parser.add_argument(
        "--action-steps",
        "--action_steps",
        type=int,
        nargs="+",
        default=list(token_budget.DEFAULT_ACTION_STEPS),
    )
    parser.add_argument("--assets-base-dir", "--assets_base_dir", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    render_episodes(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        config_name=args.config_name,
        num_episodes=args.num_episodes,
        episode_ids=args.episode_ids,
        batch_size=args.batch_size,
        frame_stride=args.frame_stride,
        fps=args.fps,
        seed=args.seed,
        coverage=args.coverage,
        sample_quantile=args.sample_quantile,
        alignment=args.alignment,
        global_coverage_upper=args.global_coverage_upper,
        global_min_tokens=args.global_min_tokens,
        color_max_multiplier=args.color_max_multiplier,
        vit_layers=args.vit_layers,
        vlm_layers=args.vlm_layers,
        action_layers=args.action_layers,
        num_steps=args.num_steps,
        action_steps=args.action_steps,
        assets_base_dir=args.assets_base_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
