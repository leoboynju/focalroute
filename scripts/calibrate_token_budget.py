#!/usr/bin/env python3
"""Offline teacher attention token-budget calibration for PI0.5 VisionRouter.

Note: Historical docstring mentioned 'action' as the default source; the actual CLI/config
default is 'vit' (SigLIP last 4 layers).
"""

import argparse
from collections.abc import Sequence
import dataclasses
import logging
import pathlib

import jax
import numpy as np
import torch
import tqdm

import openpi.models.model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.models_pytorch import pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training import token_budget


def _collate_samples(items: Sequence[dict]) -> dict:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def sample_dataset_indices(dataset_length: int, num_samples: int, seed: int = 42) -> np.ndarray:
    if dataset_length <= 0:
        raise ValueError(f"dataset_length must be > 0, got {dataset_length}")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be > 0, got {num_samples}")
    rng = np.random.default_rng(seed)
    count = min(int(num_samples), int(dataset_length))
    return np.asarray(rng.choice(int(dataset_length), size=count, replace=False), dtype=np.int64)


def build_transformed_dataset(train_config: _config.TrainConfig):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    raw_dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    transformed = _data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=False)
    return transformed, data_config


def run_calibration(
    *,
    checkpoint_dir: str | pathlib.Path,
    output_path: str | pathlib.Path,
    config_name: str = "pi05_libero",
    attention_source: str = "vit",
    num_samples: int = 1024,
    batch_size: int = 2,
    num_workers: int = 0,
    seed: int = 42,
    coverage_target: float = 0.80,
    sample_quantile: float = 0.95,
    alignment: int = 32,
    global_coverage_upper: float = 0.90,
    global_min_tokens: int = 64,
    layers: Sequence[int] | None = None,
    num_steps: int = token_budget.DEFAULT_ACTION_NUM_STEPS,
    action_steps: Sequence[int] = token_budget.DEFAULT_ACTION_STEPS,
    assets_base_dir: str | None = None,
    device: str | None = None,
    dataset=None,
    repo_id_override: str | None = None,
) -> tuple[pathlib.Path, pathlib.Path, dict]:
    del num_workers  # deterministic main-process indexing over sampled frames
    ckpt_path = pathlib.Path(checkpoint_dir)
    if pi0_pytorch.checkpoint_has_vision_router(ckpt_path):
        raise ValueError(f"Calibration requires a router-free teacher checkpoint, but router found in {ckpt_path}")

    out_target = pathlib.Path(output_path)
    json_name, npz_name = token_budget.default_budget_filenames(attention_source)
    if out_target.suffix == ".json":
        json_path = out_target
        npz_path = out_target.with_suffix(".npz")
    else:
        json_path = out_target / json_name
        npz_path = out_target / npz_name

    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing calibration outputs: {json_path} / {npz_path}")

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

    if layers is None:
        if attention_source == "vit":
            vit_total = len(model.paligemma_with_expert.paligemma.vision_tower.vision_model.encoder.layers)
            resolved_layers = tuple(range(max(0, vit_total - 4), vit_total))
        elif attention_source == "vlm":
            vlm_total = len(model.paligemma_with_expert.paligemma.language_model.layers)
            resolved_layers = tuple(range(max(0, vlm_total - 4), vlm_total))
        elif attention_source == "action":
            act_total = len(model.paligemma_with_expert.gemma_expert.model.layers)
            resolved_layers = tuple(range(max(0, act_total - 4), act_total))
        else:
            raise ValueError(f"Unsupported attention_source '{attention_source}'")
    else:
        resolved_layers = tuple(int(x) for x in layers)

    sample_indices = sample_dataset_indices(len(dataset), num_samples, seed=seed)

    all_relevance: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_masses: list[np.ndarray] = []

    for start in tqdm.trange(0, len(sample_indices), batch_size, desc=f"Calibrating ({attention_source})"):
        batch_indices = sample_indices[start : start + batch_size]
        items = [dataset[int(idx)] for idx in batch_indices]
        collated = _collate_samples(items)
        tensor_batch = jax.tree.map(lambda x: torch.as_tensor(x, device=resolved_device), collated)
        observation = _model.Observation.from_dict(tensor_batch)

        collected = token_budget.collect_teacher_attention_batch(
            model,
            observation,
            sources=(attention_source,),
            vit_layers=resolved_layers if attention_source == "vit" else None,
            vlm_layers=resolved_layers if attention_source == "vlm" else None,
            action_layers=resolved_layers if attention_source == "action" else None,
            num_steps=num_steps,
            action_steps=action_steps,
        )
        all_relevance.append(collected.normalized_by_source[attention_source])
        all_masks.append(collected.valid_mask)
        all_masses.append(collected.visual_mass_by_source[attention_source])

    relevance_arr = np.concatenate(all_relevance, axis=0)
    valid_mask_arr = np.concatenate(all_masks, axis=0)
    visual_mass_arr = np.concatenate(all_masses, axis=0)

    summary, hard_tokens_per_sample = token_budget.compute_token_budget_summary(
        relevance_arr,
        valid_mask_arr,
        coverage_target=coverage_target,
        sample_quantile=sample_quantile,
        alignment=alignment,
        global_coverage_upper=global_coverage_upper,
        global_min_tokens=global_min_tokens,
        visual_attention_mass=visual_mass_arr,
    )

    model_config_dict = pi0_pytorch.build_model_config_dict(teacher_model_cfg)
    report = token_budget.build_budget_report(
        config_name=config_name,
        repo_id=repo_id,
        checkpoint=str(ckpt_path),
        attention_source=attention_source,
        model_config_dict=model_config_dict,
        summary=summary,
        layers=resolved_layers,
        seed=seed,
        sample_indices_source="uniform_without_replacement",
        num_steps=num_steps if attention_source == "action" else None,
        action_steps=action_steps if attention_source == "action" else None,
    )

    saved_json, saved_npz = token_budget.save_budget_artifacts(
        json_path,
        report,
        sample_indices=sample_indices,
        relevance=relevance_arr,
        valid_mask=valid_mask_arr,
        hard_tokens_per_sample=hard_tokens_per_sample,
        visual_attention_mass=visual_mass_arr,
        output_npz_path=npz_path,
    )
    logging.info(
        "Calibration complete (%s): H=%d, G=%d, total=%d -> %s",
        attention_source,
        summary["recommended_hard_tokens"],
        summary["recommended_global_tokens"],
        summary["recommended_total_tokens"],
        saved_json,
    )
    return saved_json, saved_npz, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate fixed H/G VisionRouter token budget from teacher attention.")
    parser.add_argument("--config-name", "--config_name", default="pi05_libero")
    parser.add_argument(
        "--checkpoint-dir",
        "--checkpoint_dir",
        default="models/pi05_libero_pytorch",
        help="Path to router-free PyTorch teacher checkpoint directory or safetensors file.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        "--output-dir",
        "--output_dir",
        required=True,
        help="Output directory or JSON file path for token_budget_<source>.json/.npz.",
    )
    parser.add_argument(
        "--attention-source",
        "--attention_source",
        "--source",
        choices=("vit", "vlm", "action"),
        default="vit",
    )
    parser.add_argument("--num-samples", "--num_samples", type=int, default=1024)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=2)
    parser.add_argument("--num-workers", "--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-target", "--coverage_target", "--coverage", type=float, default=0.80)
    parser.add_argument("--sample-quantile", "--sample_quantile", type=float, default=0.95)
    parser.add_argument("--alignment", type=int, default=32)
    parser.add_argument("--global-coverage-upper", "--global_coverage_upper", type=float, default=0.90)
    parser.add_argument("--global-min-tokens", "--global_min_tokens", type=int, default=64)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
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
    run_calibration(
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_path,
        config_name=args.config_name,
        attention_source=args.attention_source,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        coverage_target=args.coverage_target,
        sample_quantile=args.sample_quantile,
        alignment=args.alignment,
        global_coverage_upper=args.global_coverage_upper,
        global_min_tokens=args.global_min_tokens,
        layers=args.layers,
        num_steps=args.num_steps,
        action_steps=args.action_steps,
        assets_base_dir=args.assets_base_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
