from collections.abc import Sequence
import dataclasses
import json
import logging
import math
import os
import pathlib
from typing import Any

import safetensors
import safetensors.torch
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.shared import torch_runtime

MODEL_CONFIG_FILENAME = "model_config.json"
MODEL_CONFIG_FORMAT_VERSION = 1


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "npu" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    # Ascend has no Dirichlet sampling kernel; transfer only the small batch of sampled times.
    sample_device = "cpu" if torch.device(device).type == "npu" else device
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=sample_device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=sample_device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,)).to(device)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def relaxed_topk(
    scores: Tensor,
    k: int,
    temperature: float = 1.0,
    *,
    ranked_indices: Tensor | None = None,
    order: Tensor | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Deterministic relaxed sorting straight-through Top-K assignment matrix [B, k, N]."""
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    bsize, num_tokens = scores.shape
    if k <= 0 or k > num_tokens:
        raise ValueError(f"k must be in [1, {num_tokens}], got {k}")

    scores_fp32 = scores.float()
    if valid_mask is not None:
        if valid_mask.shape != scores.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with the same shape as scores")
        valid_counts = valid_mask.float().sum(dim=-1)
        if (valid_counts < k).any():
            raise ValueError("Top-K requires at least k valid patches in every sample")
    if ranked_indices is None:
        ranking_scores = scores_fp32 if valid_mask is None else scores_fp32.masked_fill(~valid_mask, -torch.inf)
        _, ranked_indices = torch.topk(ranking_scores, k=k, dim=-1, largest=True, sorted=True)
    if order is None:
        # Patch indices are exactly representable in FP32; integer sorting uses slow Ascend AICPU.
        order = torch.argsort(ranked_indices.float() if scores.device.type == "npu" else ranked_indices, dim=-1)

    pairwise_abs_diff = torch.abs(scores_fp32.unsqueeze(2) - scores_fp32.unsqueeze(1))
    if valid_mask is not None:
        # Valid-subset NeuralSort, padded to N; never put infinities into pairwise differences.
        pairwise_abs_diff = pairwise_abs_diff * valid_mask.unsqueeze(1)
    pairwise_abs_diff = pairwise_abs_diff.sum(dim=2)
    ranks = torch.arange(1, k + 1, device=scores.device, dtype=torch.float32).view(1, k, 1)
    size = float(num_tokens) if valid_mask is None else valid_counts[:, None, None]
    scale = size + 1.0 - 2.0 * ranks
    logits = (scale * scores_fp32.unsqueeze(1) - pairwise_abs_diff.unsqueeze(1)) / float(temperature)
    if valid_mask is not None:
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -torch.inf)
    p_soft = F.softmax(logits, dim=-1)
    p_hard = F.one_hot(ranked_indices, num_classes=num_tokens).to(dtype=p_soft.dtype)

    row_order = order.unsqueeze(-1).expand(bsize, k, num_tokens)
    p_soft = torch.gather(p_soft, dim=1, index=row_order)
    p_hard = torch.gather(p_hard, dim=1, index=row_order)
    return p_hard + (p_soft - p_soft.detach())


@dataclasses.dataclass
class VisionRouterOutput:
    tokens: Tensor
    output_mask: Tensor
    camera_weights: Tensor
    selected_indices: Tensor
    assignment: Tensor
    scores: Tensor | None = None


class VisionRouter(nn.Module):
    """Language-conditioned Hard Top-K + Global/Soft aggregation router over projected camera patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_cameras: int = 3,
        hard_tokens: int = 96,
        soft_tokens: int = 32,
        *,
        hard_enabled: bool = True,
        soft_enabled: bool = True,
        selection_temperature: float = 1.0,
        router_dim: int | None = None,
        mask_invalid_cameras: bool = False,
    ):
        super().__init__()
        if not hard_enabled and not soft_enabled:
            raise ValueError("VisionRouter requires at least one enabled branch (hard_enabled or soft_enabled).")
        if hard_enabled and hard_tokens <= 0:
            raise ValueError(f"hard_tokens must be > 0 when hard_enabled is True, got {hard_tokens}")
        if not hard_enabled and hard_tokens < 0:
            raise ValueError(f"hard_tokens must be >= 0, got {hard_tokens}")
        if soft_enabled and soft_tokens <= 0:
            raise ValueError(f"soft_tokens must be > 0 when soft_enabled is True, got {soft_tokens}")
        if not soft_enabled and soft_tokens < 0:
            raise ValueError(f"soft_tokens must be >= 0, got {soft_tokens}")
        if selection_temperature <= 0:
            raise ValueError(f"selection_temperature must be > 0, got {selection_temperature}")
        if num_cameras <= 0:
            raise ValueError(f"num_cameras must be > 0, got {num_cameras}")

        self.embed_dim = int(embed_dim)
        self.num_cameras = int(num_cameras)
        self.hard_enabled = bool(hard_enabled)
        self.soft_enabled = bool(soft_enabled)
        self.hard_tokens = int(hard_tokens) if self.hard_enabled else 0
        self.soft_tokens = int(soft_tokens) if self.soft_enabled else 0
        self.selection_temperature = float(selection_temperature)
        self.mask_invalid_cameras = bool(mask_invalid_cameras)

        expected_router_dim = self.hard_tokens + self.soft_tokens
        if router_dim is None:
            router_dim = expected_router_dim
        elif int(router_dim) != expected_router_dim:
            raise ValueError(f"router_dim ({router_dim}) must equal enabled H + G ({expected_router_dim})")
        self.router_dim = int(router_dim)

        self.camera_embedding = nn.Parameter(torch.zeros(self.num_cameras, 1, self.embed_dim))
        self.image_norm = nn.LayerNorm(self.embed_dim)
        self.prompt_norm = nn.LayerNorm(self.embed_dim)
        self.image_proj = nn.Linear(self.embed_dim, self.router_dim)
        self.prompt_proj = nn.Linear(self.embed_dim, self.router_dim)

        if self.hard_enabled:
            self.selection_score: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.router_dim, self.router_dim),
                nn.SiLU(),
                nn.Linear(self.router_dim, 1),
            )
        else:
            self.selection_score = None

        if self.soft_enabled:
            self.compression_assignment: nn.Sequential | None = nn.Sequential(
                nn.Linear(self.router_dim, self.router_dim),
                nn.SiLU(),
                nn.Linear(self.router_dim, self.soft_tokens),
            )
        else:
            self.compression_assignment = None

    def relaxed_topk(
        self,
        scores: Tensor,
        k: int | None = None,
        temperature: float | None = None,
        *,
        ranked_indices: Tensor | None = None,
        order: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return relaxed_topk(
            scores,
            self.hard_tokens if k is None else k,
            self.selection_temperature if temperature is None else temperature,
            ranked_indices=ranked_indices,
            order=order,
            valid_mask=valid_mask,
        )

    def forward(
        self,
        image_embs: Sequence[Tensor],
        prompt_emb: Tensor,
        prompt_mask: Tensor | None = None,
        image_masks: Sequence[Tensor] | None = None,
    ) -> VisionRouterOutput:
        if len(image_embs) != self.num_cameras:
            raise ValueError(f"Expected {self.num_cameras} camera embeddings, got {len(image_embs)}")

        input_dtype = image_embs[0].dtype
        router_dtype = self.image_proj.weight.dtype

        stacked_images = torch.stack([img.to(dtype=router_dtype) for img in image_embs], dim=1)
        bsize, num_cameras, patches_per_camera, embed_dim = stacked_images.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected image embedding dim {self.embed_dim}, got {embed_dim}")
        total_patches = num_cameras * patches_per_camera
        valid_mask = None
        if self.mask_invalid_cameras:
            if image_masks is None or len(image_masks) != num_cameras:
                raise ValueError("Camera-masked Router requires one image mask per camera")
            camera_mask = torch.stack([m.to(device=stacked_images.device, dtype=torch.bool) for m in image_masks], dim=1)
            if camera_mask.shape != (bsize, num_cameras):
                raise ValueError("Camera masks must have shape [batch] for each camera")
            valid_mask = camera_mask.unsqueeze(-1).expand(bsize, num_cameras, patches_per_camera).reshape(bsize, total_patches)
            if (valid_mask.float().sum(dim=-1) < max(self.hard_tokens, 1)).any():
                raise ValueError("Camera-masked Router has insufficient valid patches")

        camera_values = stacked_images + self.camera_embedding.to(dtype=router_dtype).unsqueeze(0)

        prompt_cast = prompt_emb.to(dtype=router_dtype)
        if prompt_mask is None:
            pooled_prompt = prompt_cast.mean(dim=1)
        else:
            mask_f = prompt_mask.to(device=prompt_cast.device, dtype=router_dtype).unsqueeze(-1)
            denom = mask_f.sum(dim=1).clamp_min(1.0)
            pooled_prompt = (prompt_cast * mask_f).sum(dim=1) / denom

        joint_features = F.silu(
            self.image_proj(self.image_norm(camera_values))
            + self.prompt_proj(self.prompt_norm(pooled_prompt))[:, None, None, :]
        )

        flat_values = camera_values.reshape(bsize, total_patches, embed_dim)
        flat_features = joint_features.reshape(bsize, total_patches, self.router_dim)

        token_parts: list[Tensor] = []
        camera_weight_parts: list[Tensor] = []
        scores = None

        if self.hard_enabled:
            if self.hard_tokens > total_patches:
                raise ValueError(f"hard_tokens ({self.hard_tokens}) exceeds total visual patches ({total_patches})")
            assert self.selection_score is not None
            scores = self.selection_score(flat_features).squeeze(-1)
            ranking_scores = scores if valid_mask is None else scores.masked_fill(~valid_mask, -torch.inf)
            _, ranked_indices = torch.topk(ranking_scores, k=self.hard_tokens, dim=-1, largest=True, sorted=True)
            order = torch.argsort(ranked_indices.float() if scores.device.type == "npu" else ranked_indices, dim=-1)
            selected_indices = torch.gather(ranked_indices, dim=-1, index=order)

            if self.training:
                p_st = self.relaxed_topk(
                    scores,
                    self.hard_tokens,
                    self.selection_temperature,
                    ranked_indices=ranked_indices,
                    order=order,
                    valid_mask=valid_mask,
                )
                hard_tokens = torch.bmm(p_st.to(dtype=flat_values.dtype), flat_values)
            else:
                gather_index = selected_indices.unsqueeze(-1).expand(bsize, self.hard_tokens, embed_dim)
                hard_tokens = torch.gather(flat_values, dim=1, index=gather_index)

            hard_cameras = torch.div(selected_indices, patches_per_camera, rounding_mode="floor")
            hard_camera_weights = F.one_hot(hard_cameras, num_classes=num_cameras).to(dtype=router_dtype)
            token_parts.append(hard_tokens)
            camera_weight_parts.append(hard_camera_weights)
        else:
            selected_indices = torch.empty((bsize, 0), dtype=torch.long, device=flat_values.device)

        if self.soft_enabled:
            assert self.compression_assignment is not None
            assignment_logits = self.compression_assignment(flat_features)
            if valid_mask is not None:
                assignment_logits = assignment_logits.masked_fill(~valid_mask.unsqueeze(-1), -torch.inf)
            assignment = F.softmax(assignment_logits.float(), dim=1).to(dtype=router_dtype)
            soft_tokens = torch.einsum("bnk,bnd->bkd", assignment, flat_values)
            soft_camera_weights = (
                assignment.reshape(bsize, num_cameras, patches_per_camera, self.soft_tokens)
                .sum(dim=2)
                .transpose(1, 2)
                .contiguous()
            )
            token_parts.append(soft_tokens)
            camera_weight_parts.append(soft_camera_weights)
        else:
            assignment = torch.empty((bsize, total_patches, 0), dtype=router_dtype, device=flat_values.device)

        tokens = torch.cat(token_parts, dim=1).to(dtype=input_dtype)
        camera_weights = torch.cat(camera_weight_parts, dim=1)
        output_mask = torch.ones((bsize, tokens.shape[1]), dtype=torch.bool, device=tokens.device)

        return VisionRouterOutput(
            tokens=tokens,
            output_mask=output_mask,
            camera_weights=camera_weights,
            selected_indices=selected_indices,
            assignment=assignment,
            scores=scores,
        )


def _resolve_safetensors_and_config_paths(weight_path: str | os.PathLike) -> tuple[pathlib.Path, pathlib.Path]:
    path = pathlib.Path(weight_path)
    if path.is_dir():
        return path / "model.safetensors", path / MODEL_CONFIG_FILENAME
    return path, path.parent / MODEL_CONFIG_FILENAME


def build_model_config_dict(config) -> dict[str, Any]:
    """Build the standardized model_config.json dictionary for a Pi0Config."""
    pi05 = bool(getattr(config, "pi05", False))
    hard_enabled = bool(pi05 and getattr(config, "vision_router_hard_enabled", False))
    soft_enabled = bool(pi05 and getattr(config, "vision_router_soft_enabled", False))
    hard_tokens = int(getattr(config, "vision_router_hard_tokens", 96)) if hard_enabled else 0
    soft_tokens = int(getattr(config, "vision_router_soft_tokens", 32)) if soft_enabled else 0
    router_dim = hard_tokens + soft_tokens

    payload = {
        "format_version": MODEL_CONFIG_FORMAT_VERSION,
        "model": {
            "action_dim": int(config.action_dim),
            "action_horizon": int(config.action_horizon),
            "max_token_len": int(config.max_token_len),
            "dtype": str(config.dtype),
            "paligemma_variant": str(config.paligemma_variant),
            "action_expert_variant": str(config.action_expert_variant),
            "pi05": pi05,
            "discrete_state_input": bool(config.discrete_state_input),
        },
        "vision_router": {
            "hard_enabled": hard_enabled,
            "soft_enabled": soft_enabled,
            "hard_tokens": hard_tokens,
            "soft_tokens": soft_tokens,
            "router_dim": router_dim,
            "num_cameras": len(_preprocessing.IMAGE_KEYS),
            "selection_temperature": float(getattr(config, "vision_router_selection_temperature", 1.0)),
            "camera_loss_weight": 0.0,
        },
    }
    # Omit the default to preserve historical exact model_config dictionary comparisons.
    if (hard_enabled or soft_enabled) and getattr(config, "vision_router_mask_invalid_cameras", False):
        payload["format_version"] = 2  # Old loaders must reject this semantic variant, not silently ignore its mask.
        payload["vision_router"]["mask_invalid_cameras"] = True
    if getattr(config, "vision_router_preserve_hard_rope", False):
        payload["format_version"] = 3  # Old loaders must not silently renumber this checkpoint.
        payload["vision_router"]["preserve_hard_rope"] = True
    return payload


def save_model_config(config, output_path: str | os.PathLike) -> pathlib.Path:
    """Save model_config.json to a directory or file path."""
    target = pathlib.Path(output_path)
    if target.suffix != ".json":
        target.mkdir(parents=True, exist_ok=True)
        target = target / MODEL_CONFIG_FILENAME
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_model_config_dict(config)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def load_model_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load model_config.json from a directory or file path."""
    target = pathlib.Path(path)
    if target.is_dir():
        target = target / MODEL_CONFIG_FILENAME
    payload = json.loads(target.read_text())
    if not isinstance(payload, dict) or payload.get("format_version") not in (MODEL_CONFIG_FORMAT_VERSION, 2, 3):
        raise ValueError(f"Unsupported model_config format in {target}")
    router = payload.get("vision_router")
    if isinstance(router, dict):
        if router.get("mask_invalid_cameras") is True and payload["format_version"] not in (2, 3):
            raise ValueError("Camera-masked checkpoints require model_config format_version=2 or 3")
        preserve = router.get("preserve_hard_rope", False)
        if not isinstance(preserve, bool):
            raise ValueError("model_config.json preserve_hard_rope must be boolean")
        if preserve != (payload["format_version"] == 3):
            raise ValueError("Original Hard RoPE requires model_config format_version=3")
        if preserve and not (router.get("hard_enabled") is True and router.get("mask_invalid_cameras") is True):
            raise ValueError("Original Hard RoPE requires enabled Hard routing and valid-camera masking")
    return payload


def _inspect_safetensors_shapes(safetensors_path: pathlib.Path) -> dict[str, tuple[int, ...]]:
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {safetensors_path}")
    with safetensors.safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
        return {key: tuple(f.get_slice(key).get_shape()) for key in f.keys()}


def checkpoint_has_vision_router(weight_path: str | os.PathLike) -> bool:
    """Return True if the checkpoint contains any VisionRouter parameters or enabled router metadata."""
    safetensors_path, config_json_path = _resolve_safetensors_and_config_paths(weight_path)
    shapes = _inspect_safetensors_shapes(safetensors_path)
    if any(key.startswith("vision_router.") for key in shapes):
        return True
    if config_json_path.exists():
        payload = load_model_config(config_json_path)
        vr = payload.get("vision_router", {})
        if vr.get("hard_enabled", False) or vr.get("soft_enabled", False):
            return True
    return False


def resolve_checkpoint_model_config(model_config, weight_path: str | os.PathLike):
    """Resolve VisionRouter topology and token counts from checkpoint weights and optional model_config.json."""
    safetensors_path, config_json_path = _resolve_safetensors_and_config_paths(weight_path)
    tensor_shapes = _inspect_safetensors_shapes(safetensors_path)

    if any(key == "vision_router.logits" or key.startswith("vision_router.logits.") for key in tensor_shapes):
        raise ValueError(
            f"Unsupported legacy camera-weight router checkpoint ('vision_router.logits.*') in {safetensors_path}"
        )

    router_keys = {key for key in tensor_shapes if key.startswith("vision_router.")}
    router_meta: dict[str, Any] | None = None
    if config_json_path.exists():
        payload = load_model_config(config_json_path)
        vr_meta = payload.get("vision_router")
        if not isinstance(vr_meta, dict):
            raise ValueError(f"Invalid vision_router section in {config_json_path}")
        router_meta = vr_meta

    if not router_keys:
        if router_meta is not None:
            if router_meta.get("hard_enabled", False) or router_meta.get("soft_enabled", False):
                raise ValueError(
                    f"model_config.json enables vision_router, but {safetensors_path} has no vision_router weights"
                )
            if int(router_meta.get("router_dim", 0)) != 0:
                raise ValueError(f"model_config.json router_dim must be 0 when router is disabled in {config_json_path}")
        if hasattr(model_config, "vision_router_hard_enabled"):
            return dataclasses.replace(
                model_config,
                vision_router_hard_enabled=False,
                vision_router_soft_enabled=False,
                vision_router_mask_invalid_cameras=False,
                vision_router_preserve_hard_rope=False,
            )
        return model_config

    if not getattr(model_config, "pi05", False):
        raise ValueError("VisionRouter checkpoints require a pi05=True model config.")

    base_keys = {
        "vision_router.camera_embedding",
        "vision_router.image_norm.weight",
        "vision_router.image_norm.bias",
        "vision_router.prompt_norm.weight",
        "vision_router.prompt_norm.bias",
        "vision_router.image_proj.weight",
        "vision_router.image_proj.bias",
        "vision_router.prompt_proj.weight",
        "vision_router.prompt_proj.bias",
    }
    hard_keys = {
        "vision_router.selection_score.0.weight",
        "vision_router.selection_score.0.bias",
        "vision_router.selection_score.2.weight",
        "vision_router.selection_score.2.bias",
    }
    soft_keys = {
        "vision_router.compression_assignment.0.weight",
        "vision_router.compression_assignment.0.bias",
        "vision_router.compression_assignment.2.weight",
        "vision_router.compression_assignment.2.bias",
    }

    if not base_keys.issubset(router_keys):
        missing_base = sorted(base_keys - router_keys)
        raise ValueError(f"Checkpoint {safetensors_path} is missing base VisionRouter keys: {missing_base}")

    has_any_hard = any(key.startswith("vision_router.selection_score.") for key in router_keys)
    has_any_soft = any(key.startswith("vision_router.compression_assignment.") for key in router_keys)
    if has_any_hard and not hard_keys.issubset(router_keys):
        raise ValueError(f"Checkpoint {safetensors_path} has incomplete selection_score weights")
    if has_any_soft and not soft_keys.issubset(router_keys):
        raise ValueError(f"Checkpoint {safetensors_path} has incomplete compression_assignment weights")

    allowed_keys = base_keys | (hard_keys if has_any_hard else set()) | (soft_keys if has_any_soft else set())
    unknown_router_keys = sorted(router_keys - allowed_keys)
    if unknown_router_keys:
        raise ValueError(f"Checkpoint {safetensors_path} has unexpected VisionRouter keys: {unknown_router_keys}")
    if not has_any_hard and not has_any_soft:
        raise ValueError(f"Checkpoint {safetensors_path} has VisionRouter base weights but no active branch weights")

    expected_cameras = len(_preprocessing.IMAGE_KEYS)
    cam_shape = tensor_shapes["vision_router.camera_embedding"]
    if len(cam_shape) != 3 or cam_shape[0] != expected_cameras or cam_shape[1] != 1:
        raise ValueError(
            f"Invalid vision_router.camera_embedding shape {cam_shape}; expected ({expected_cameras}, 1, D)"
        )
    embed_dim = cam_shape[2]
    image_proj_shape = tensor_shapes["vision_router.image_proj.weight"]
    prompt_proj_shape = tensor_shapes["vision_router.prompt_proj.weight"]
    if len(image_proj_shape) != 2 or len(prompt_proj_shape) != 2:
        raise ValueError("Invalid VisionRouter projection weight rank")
    weight_router_dim = image_proj_shape[0]
    if prompt_proj_shape[0] != weight_router_dim or image_proj_shape[1] != embed_dim or prompt_proj_shape[1] != embed_dim:
        raise ValueError("Inconsistent VisionRouter projection shapes in checkpoint")

    weight_soft_tokens = (
        tensor_shapes["vision_router.compression_assignment.2.weight"][0] if has_any_soft else 0
    )

    if router_meta is not None:
        hard_enabled = bool(router_meta.get("hard_enabled", False))
        soft_enabled = bool(router_meta.get("soft_enabled", False))
        if hard_enabled != has_any_hard or soft_enabled != has_any_soft:
            raise ValueError(
                f"model_config.json branch flags (hard={hard_enabled}, soft={soft_enabled}) do not match "
                f"weights (hard={has_any_hard}, soft={has_any_soft})"
            )
        meta_num_cameras = int(router_meta.get("num_cameras", expected_cameras))
        if meta_num_cameras != expected_cameras:
            raise ValueError(f"model_config.json num_cameras ({meta_num_cameras}) != {expected_cameras}")
        hard_tokens = int(router_meta.get("hard_tokens", 0)) if hard_enabled else 0
        soft_tokens = int(router_meta.get("soft_tokens", 0)) if soft_enabled else 0
        if hard_enabled and hard_tokens <= 0:
            raise ValueError(f"model_config.json hard_tokens must be > 0 when hard_enabled is True, got {hard_tokens}")
        if soft_enabled and soft_tokens <= 0:
            raise ValueError(f"model_config.json soft_tokens must be > 0 when soft_enabled is True, got {soft_tokens}")
        if soft_enabled and soft_tokens != weight_soft_tokens:
            raise ValueError(
                f"model_config.json soft_tokens ({soft_tokens}) != compression_assignment weight shape ({weight_soft_tokens})"
            )
        meta_router_dim = int(router_meta.get("router_dim", -1))
        if meta_router_dim != hard_tokens + soft_tokens or meta_router_dim != weight_router_dim:
            raise ValueError(
                f"Inconsistent router_dim: meta={meta_router_dim}, H+G={hard_tokens + soft_tokens}, weight={weight_router_dim}"
            )
        selection_temperature = float(
            router_meta.get("selection_temperature", getattr(model_config, "vision_router_selection_temperature", 1.0))
        )
        mask_invalid_cameras = router_meta.get("mask_invalid_cameras", False)
        preserve_hard_rope = router_meta.get("preserve_hard_rope", False)
        if not isinstance(mask_invalid_cameras, bool):
            raise ValueError("model_config.json mask_invalid_cameras must be boolean")
    else:
        hard_enabled = has_any_hard
        soft_enabled = has_any_soft
        hard_tokens = 128 if hard_enabled else 0
        soft_tokens = weight_soft_tokens if soft_enabled else 0
        if weight_router_dim != hard_tokens + soft_tokens:
            raise ValueError(
                f"Cannot infer VisionRouter token counts without {MODEL_CONFIG_FILENAME}: "
                f"weight router_dim={weight_router_dim} != inferred H({hard_tokens}) + G({soft_tokens})"
            )
        selection_temperature = float(getattr(model_config, "vision_router_selection_temperature", 1.0))
        mask_invalid_cameras = False
        preserve_hard_rope = False

    return dataclasses.replace(
        model_config,
        vision_router_hard_enabled=hard_enabled,
        vision_router_soft_enabled=soft_enabled,
        vision_router_hard_tokens=hard_tokens if hard_enabled else 0,
        vision_router_soft_tokens=soft_tokens if soft_enabled else 0,
        vision_router_selection_temperature=selection_temperature,
        vision_router_mask_invalid_cameras=mask_invalid_cameras,
        vision_router_preserve_hard_rope=preserve_hard_rope,
    )


def load_pretrained_weights(
    model: nn.Module,
    weight_path: str | os.PathLike,
    device: str | int | torch.device | None = None,
) -> tuple[set[str], list[str]]:
    """Load pretrained weights into PI0Pytorch, allowing only the entire new VisionRouter to be missing."""
    target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    safetensors_path, _ = _resolve_safetensors_and_config_paths(weight_path)
    if not safetensors_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found at {safetensors_path}")

    shapes = _inspect_safetensors_shapes(safetensors_path)
    if any(key == "vision_router.logits" or key.startswith("vision_router.logits.") for key in shapes):
        raise ValueError(
            f"Unsupported legacy camera-weight router checkpoint ('vision_router.logits.*') in {safetensors_path}"
        )

    if device is None:
        first_param = next(target_model.parameters(), None)
        device_arg: str | int = str(first_param.device) if first_param is not None else "cpu"
    else:
        device_arg = str(device) if isinstance(device, torch.device) else device

    missing, unexpected = safetensors.torch.load_model(
        target_model,
        str(safetensors_path),
        strict=False,
        device="cpu" if str(device_arg).startswith("npu") else device_arg,
    )
    missing_set = set(missing)
    if unexpected:
        raise RuntimeError(f"Unexpected key(s) in state_dict: {sorted(unexpected)}")

    if not missing_set:
        return missing_set, unexpected

    vision_router = getattr(target_model, "vision_router", None)
    expected_router_keys = (
        {f"vision_router.{k}" for k in vision_router.state_dict().keys()} if vision_router is not None else set()
    )
    if not expected_router_keys or missing_set != expected_router_keys:
        raise RuntimeError(
            f"Missing key(s) in state_dict do not match full VisionRouter parameter set: "
            f"missing={sorted(missing_set)}, expected_router={sorted(expected_router_keys)}"
        )
    return missing_set, unexpected


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        hard_enabled = bool(getattr(config, "vision_router_hard_enabled", True))
        soft_enabled = bool(getattr(config, "vision_router_soft_enabled", True))
        if self.pi05 and (hard_enabled or soft_enabled):
            self.vision_router: VisionRouter | None = VisionRouter(
                embed_dim=paligemma_config.width,
                num_cameras=len(_preprocessing.IMAGE_KEYS),
                hard_tokens=int(getattr(config, "vision_router_hard_tokens", 96)),
                soft_tokens=int(getattr(config, "vision_router_soft_tokens", 32)),
                hard_enabled=hard_enabled,
                soft_enabled=soft_enabled,
                selection_temperature=float(getattr(config, "vision_router_selection_temperature", 1.0)),
                mask_invalid_cameras=bool(getattr(config, "vision_router_mask_invalid_cameras", False)),
            )
        else:
            self.vision_router = None
        self.last_router_output: VisionRouterOutput | None = None

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None and torch_runtime.device_type() != "npu":
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True, return_labels=False):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        result = (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
        return (*result, observation.router_label_masks) if return_labels else result

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, *, return_positions=False
    ) -> tuple[torch.Tensor, ...]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        if self.config.vision_router_preserve_hard_rope and not return_positions:
            raise ValueError("Original Hard RoPE callers must request explicit positions with return_positions=True")
        image_embs = []
        for img in images:

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            image_embs.append(img_emb)

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs = []
        pad_masks = []
        att_masks = []

        if self.vision_router is not None:
            router_output = self.vision_router(image_embs, lang_emb, lang_masks, image_masks=img_masks)
            self.last_router_output = router_output
            num_routed = router_output.tokens.shape[1]
            embs.append(router_output.tokens)
            pad_masks.append(router_output.output_mask)
            att_masks += [0] * num_routed
        else:
            self.last_router_output = None
            for img_emb, img_mask in zip(image_embs, img_masks, strict=True):
                bsize, num_img_embs = img_emb.shape[:2]
                embs.append(img_emb)
                pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
                att_masks += [0] * num_img_embs

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        prefix_position_ids = torch.cumsum(pad_masks, dim=1) - 1
        action_position_offset = pad_masks.sum(dim=1, keepdim=True)
        if self.config.vision_router_preserve_hard_rope:
            assert self.vision_router is not None
            hard, global_tokens = self.vision_router.hard_tokens, self.vision_router.soft_tokens
            original_visual_mask = torch.cat([
                mask[:, None].expand(-1, image.shape[1])
                for image, mask in zip(image_embs, img_masks, strict=True)
            ], dim=1)
            original_visual_positions = torch.cumsum(original_visual_mask, dim=1) - 1
            visual_count = original_visual_mask.sum(dim=1, keepdim=True)
            hard_positions = original_visual_positions.gather(1, router_output.selected_indices)
            text_positions = visual_count + torch.cumsum(lang_masks, dim=1) - 1
            action_position_offset = visual_count + lang_masks.sum(dim=1, keepdim=True)
            # Global has no original coordinate: append new IDs after the original action range.
            global_positions = action_position_offset + self.config.action_horizon + torch.arange(
                global_tokens, device=pad_masks.device
            )[None, :]
            prefix_position_ids = torch.cat([hard_positions, text_positions, global_positions], dim=1)
            embs = torch.cat([embs[:, :hard], embs[:, hard + global_tokens:], embs[:, hard:hard + global_tokens]], dim=1)
            pad_masks = torch.cat([pad_masks[:, :hard], lang_masks, pad_masks[:, hard:hard + global_tokens]], dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        if return_positions:
            return embs, pad_masks, att_masks, prefix_position_ids, action_position_offset
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(
        self,
        observation,
        actions,
        noise=None,
        time=None,
        *,
        return_loss_components: bool = False,
        router_label_weight: float = 0.0,
    ) -> Tensor | dict[str, Tensor]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state, labels = self._preprocess_observation(
            observation, train=True, return_labels=True
        )

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_positions, action_position_offset = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, return_positions=True
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        suffix_positions = action_position_offset + torch.cumsum(suffix_pad_masks, dim=1) - 1
        position_ids = torch.cat([prefix_positions, suffix_positions], dim=1)

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        if torch_runtime.device_type() == "npu":
            # Gemma already checkpoints each block; nesting the whole pass repeats that work.
            suffix_out = forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        else:
            suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
            )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        losses = F.mse_loss(u_t, v_t, reduction="none")
        if not return_loss_components:
            return losses

        action_loss = losses.mean()
        router_camera_loss = torch.zeros((), dtype=action_loss.dtype, device=action_loss.device)
        result = {
            "loss": action_loss + router_camera_loss,
            "action_loss": action_loss,
            "router_camera_loss": router_camera_loss,
        }
        if labels is not None and router_label_weight > 0:
            from openpi.training.router_labels import positive_distribution_loss

            if self.last_router_output is None or self.last_router_output.scores is None:
                raise ValueError("Router label supervision requires an enabled Hard branch")
            targets = torch.cat([
                F.avg_pool2d(labels[key][:, None].float(), kernel_size=14, stride=14).flatten(1)
                for key in _preprocessing.IMAGE_KEYS
            ], dim=1)
            valid = torch.cat([mask[:, None].expand(-1, 256) for mask in img_masks], dim=1)
            hard_loss, count = positive_distribution_loss(self.last_router_output.scores, targets, valid)
            result.update(loss=result['loss']+router_label_weight*hard_loss,
                          router_label_loss=hard_loss, router_labeled_count=count)
        return result

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks, prefix_position_ids, action_position_offset = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, return_positions=True
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        position_kwargs = (
            {"prefix_position_offset": action_position_offset} if self.config.vision_router_preserve_hard_rope else {}
        )
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
                **position_kwargs,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        *,
        prefix_position_offset=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if prefix_position_offset is None:
            if self.config.vision_router_preserve_hard_rope:
                raise ValueError("Original Hard RoPE requires the original action position offset with cached KV")
            prefix_position_offset = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        prefix_offsets = prefix_position_offset
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
