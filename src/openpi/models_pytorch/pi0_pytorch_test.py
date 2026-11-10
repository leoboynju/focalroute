import dataclasses
import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch


@pytest.mark.parametrize("global_tokens", [0, 3])
def test_original_rope_keeps_masked_teacher_positions_and_appends_global(patched_tiny_backbone, global_tokens):
    cfg = pi0_config.Pi0Config(
        pi05=True, dtype="float32", action_dim=8, action_horizon=4, max_token_len=4,
        vision_router_hard_tokens=5, vision_router_soft_enabled=global_tokens > 0,
        vision_router_soft_tokens=global_tokens, vision_router_mask_invalid_cameras=True,
        vision_router_preserve_hard_rope=True, pytorch_compile_mode=None,
    )
    model = pi0_pytorch.PI0Pytorch(cfg).eval()
    images = [torch.randn(2, 3, 224, 224) for _ in range(3)]
    masks = [torch.tensor([True, False]), torch.tensor([False, True]), torch.tensor([True, False])]
    text_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])
    with pytest.raises(ValueError, match="return_positions=True"):
        model.embed_prefix(images, masks, torch.ones(2, 4, dtype=torch.long), text_mask)
    embs, padding, _, positions, action_offset = model.embed_prefix(
        images, masks, torch.ones(2, 4, dtype=torch.long), text_mask, return_positions=True
    )
    original_mask = torch.cat([m[:, None].expand(-1, 16) for m in masks], dim=1)
    original_positions = torch.cumsum(original_mask, dim=1) - 1
    selected = model.last_router_output.selected_indices
    torch.testing.assert_close(positions[:, :5], original_positions.gather(1, selected))
    assert original_mask.gather(1, selected).all()
    torch.testing.assert_close(positions[:, 5:9], torch.tensor([[32, 33, 33, 33], [15, 15, 15, 15]]))
    torch.testing.assert_close(action_offset, torch.tensor([[34], [16]]))
    torch.testing.assert_close(embs[:, :5], model.last_router_output.tokens[:, :5])
    assert torch.equal(padding[:, 5:9], text_mask)
    if global_tokens:
        torch.testing.assert_close(positions[:, -global_tokens:], torch.tensor([[38, 39, 40], [20, 21, 22]]))
        torch.testing.assert_close(embs[:, -global_tokens:], model.last_router_output.tokens[:, 5:])
    with pytest.raises(ValueError, match="original action position offset"):
        model.denoise_step(torch.zeros(2, 8), padding, None, torch.zeros(2, 4, 8), torch.ones(2))


def test_original_rope_metadata_roundtrip_and_legacy_default(patched_tiny_backbone, tmp_path):
    cfg = pi0_config.Pi0Config(
        pi05=True, dtype="float32", vision_router_hard_tokens=4, vision_router_soft_tokens=4,
        vision_router_mask_invalid_cameras=True, vision_router_preserve_hard_rope=True, pytorch_compile_mode=None,
    )
    model = pi0_pytorch.PI0Pytorch(cfg)
    safetensors.torch.save_file(model.state_dict(), tmp_path / "model.safetensors")
    path = pi0_pytorch.save_model_config(cfg, tmp_path)
    restored = pi0_pytorch.resolve_checkpoint_model_config(dataclasses.replace(cfg, vision_router_preserve_hard_rope=False), tmp_path)
    assert restored.vision_router_preserve_hard_rope and json.loads(path.read_text())["format_version"] == 3
    payload = json.loads(path.read_text())
    payload["format_version"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="format_version=3"):
        pi0_pytorch.load_model_config(path)
    payload["vision_router"].pop("preserve_hard_rope")
    path.write_text(json.dumps(payload))
    assert not pi0_pytorch.resolve_checkpoint_model_config(cfg, tmp_path).vision_router_preserve_hard_rope
    with pytest.raises(ValueError, match="valid-camera"):
        dataclasses.replace(cfg, vision_router_mask_invalid_cameras=False)


class _DummySelfAttn(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.head_dim = width


class _DummyLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _DummySelfAttn(width)


class _DummyLanguageModel(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.layers = nn.ModuleList([_DummyLayer(width)])
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.gradient_checkpointing = False


class _DummyVisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False


class _DummyPaliGemma(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.language_model = _DummyLanguageModel(width)
        self.vision_tower = _DummyVisionTower()


class _DummyExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.gradient_checkpointing = False


class _DummyGemmaExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyExpertModel()


class _TinyPaliGemmaWithExpert(nn.Module):
    """Lightweight replacement for PaliGemmaWithExpertModel to test PI0Pytorch routing and checkpointing quickly."""

    patches_per_camera = 16

    def __init__(self, vlm_config, action_expert_config, use_adarms=None, precision="float32"):
        super().__init__()
        self.vlm_width = vlm_config.width
        self.expert_width = action_expert_config.width
        self.paligemma = _DummyPaliGemma(self.vlm_width)
        self.gemma_expert = _DummyGemmaExpert()
        self.image_encoder = nn.Linear(3, self.vlm_width)
        self.token_embed = nn.Embedding(2048, self.vlm_width)
        self.prefix_proj = nn.Linear(self.vlm_width, self.vlm_width)
        self.suffix_proj = nn.Linear(self.expert_width, self.expert_width)
        self.last_forward_kwargs: dict = {}

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        bsize = image.shape[0]
        pooled = image.float().reshape(bsize, 3, -1).permute(0, 2, 1)[:, : self.patches_per_camera, :]
        return self.image_encoder(pooled)

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embed(tokens.clamp(min=0, max=2047))

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        self.last_forward_kwargs = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
        }
        prefix_embs, suffix_embs = inputs_embeds
        if suffix_embs is None:
            prefix_out = self.prefix_proj(prefix_embs.float())
            cache = [(prefix_out.detach(), prefix_out.detach())]
            return [prefix_out, None], cache
        if prefix_embs is None:
            suffix_out = self.suffix_proj(suffix_embs.float())
            return [None, suffix_out], None
        prefix_out = self.prefix_proj(prefix_embs.float())
        suffix_out = self.suffix_proj(suffix_embs.float())
        return [prefix_out, suffix_out], None


@pytest.fixture
def patched_tiny_backbone(monkeypatch):
    monkeypatch.setattr(
        pi0_pytorch._gemma,
        "get_config",
        lambda variant: SimpleNamespace(width=32, depth=2, mlp_dim=64, num_heads=2, num_kv_heads=1, head_dim=16),
    )
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", _TinyPaliGemmaWithExpert)


def _make_observation(batch_size: int = 2, max_token_len: int = 12, action_dim: int = 8):
    images = {
        key: torch.randn(batch_size, 3, 224, 224, dtype=torch.float32)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool),
        "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
    }
    prompt = torch.randint(1, 100, (batch_size, max_token_len), dtype=torch.int32)
    prompt_mask = torch.ones(batch_size, max_token_len, dtype=torch.bool)
    prompt_mask[:, -3:] = False
    state = torch.randn(batch_size, action_dim, dtype=torch.float32)
    return SimpleNamespace(
        images=images,
        image_masks=image_masks,
        tokenized_prompt=prompt,
        tokenized_prompt_mask=prompt_mask,
        state=state,
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _expected_router_param_count(d: int, h: int, g: int, hard_enabled: bool, soft_enabled: bool) -> int:
    r = (h if hard_enabled else 0) + (g if soft_enabled else 0)
    total = 7 * d + 2 * d * r + 2 * r
    if hard_enabled:
        total += r * r + 2 * r + 1
    if soft_enabled:
        total += r * r + r + r * g + g
    return total


@pytest.mark.parametrize(
    ("hard_enabled", "soft_enabled", "h", "g", "expected_params"),
    [
        (True, False, 128, 0, 555_521),
        (True, False, 256, 0, 1_129_473),
        (False, True, 0, 128, 571_904),
        (False, True, 0, 256, 1_195_008),
        (True, True, 128, 128, 1_228_161),
        (True, True, 64, 128, 900_161),
        (True, True, 128, 64, 887_809),
        (True, True, 128, 256, 1_982_593),
        (True, True, 256, 128, 1_933_313),
        (True, True, 224, 96, 1_562_273),
    ],
)
def test_vision_router_topologies_and_parameter_counts(hard_enabled, soft_enabled, h, g, expected_params):
    router = pi0_pytorch.VisionRouter(
        embed_dim=2048,
        num_cameras=3,
        hard_tokens=h if hard_enabled else 128,
        soft_tokens=g if soft_enabled else 128,
        hard_enabled=hard_enabled,
        soft_enabled=soft_enabled,
    )
    param_count = sum(p.numel() for p in router.parameters())
    assert param_count == expected_params
    assert param_count == _expected_router_param_count(2048, h, g, hard_enabled, soft_enabled)
    assert (router.selection_score is not None) is hard_enabled
    assert (router.compression_assignment is not None) is soft_enabled


def test_vision_router_forward_exact_gather_and_gradients():
    torch.manual_seed(0)
    b, p, d, h, g = 2, 12, 16, 5, 4
    router = pi0_pytorch.VisionRouter(
        embed_dim=d,
        num_cameras=3,
        hard_tokens=h,
        soft_tokens=g,
        hard_enabled=True,
        soft_enabled=True,
        selection_temperature=0.7,
    )
    router.train()
    with torch.no_grad():
        router.camera_embedding.normal_(mean=0.0, std=0.2)

    images = [torch.randn(b, p, d, requires_grad=True) for _ in range(3)]
    prompt = torch.randn(b, 8, d, requires_grad=True)
    prompt_mask = torch.tensor(
        [[True, True, True, True, True, False, False, False], [True, True, True, False, False, False, False, False]]
    )

    out = router(images, prompt, prompt_mask)
    assert out.tokens.shape == (b, h + g, d)
    assert out.output_mask.shape == (b, h + g)
    assert torch.all(out.output_mask)
    assert out.selected_indices.shape == (b, h)
    assert out.assignment.shape == (b, 3 * p, g)
    assert out.camera_weights.shape == (b, h + g, 3)

    # Selected indices must be strictly ascending along original patch order.
    assert torch.all(out.selected_indices[:, 1:] > out.selected_indices[:, :-1])

    # Training hard forward values must exactly equal gathering from (image + camera_embedding).
    stacked = torch.stack([img.detach() for img in images], dim=1) + router.camera_embedding.detach().unsqueeze(0)
    flat_values = stacked.reshape(b, 3 * p, d)
    expected_hard = torch.gather(
        flat_values,
        dim=1,
        index=out.selected_indices.unsqueeze(-1).expand(b, h, d),
    )
    torch.testing.assert_close(out.tokens[:, :h, :], expected_hard)

    # Soft values must match einsum("bnk,bnd->bkd", assignment, flat_values).
    expected_soft = torch.einsum("bnk,bnd->bkd", out.assignment, flat_values)
    torch.testing.assert_close(out.tokens[:, h:, :], expected_soft)

    # Assignment sums to 1 over N; camera_weights sum to 1 over cameras.
    torch.testing.assert_close(out.assignment.sum(dim=1), torch.ones(b, g))
    torch.testing.assert_close(out.camera_weights.sum(dim=-1), torch.ones(b, h + g))

    # Verify gradients reach selection_score (via hard tokens only), prompt, and images.
    hard_only_loss = (out.tokens[:, :h, :] * torch.randn_like(out.tokens[:, :h, :])).sum()
    hard_only_loss.backward()
    assert router.selection_score[0].weight.grad is not None
    assert torch.any(router.selection_score[0].weight.grad != 0)
    assert router.selection_score[2].weight.grad is not None
    assert torch.any(router.selection_score[2].weight.grad != 0)
    assert prompt.grad is not None and torch.any(prompt.grad[:, :3] != 0)
    assert torch.all(prompt.grad[:, 5:] == 0)
    for img in images:
        assert img.grad is not None and torch.any(img.grad != 0)

    # Direct check that unselected logits also receive non-zero gradient in relaxed_topk.
    scores = torch.tensor([[2.5, -0.3, 1.2, 0.4, -1.1, 0.8]], requires_grad=True)
    p_st = pi0_pytorch.relaxed_topk(scores, k=2, temperature=1.0)
    weights = torch.tensor([[[1.0], [2.0]]])
    (p_st @ torch.arange(6, dtype=torch.float32).view(1, 6, 1) * weights).sum().backward()
    assert scores.grad is not None
    assert torch.all(scores.grad != 0)


def test_vision_router_eval_skips_relaxed_topk(monkeypatch):
    router = pi0_pytorch.VisionRouter(
        embed_dim=16,
        num_cameras=3,
        hard_tokens=4,
        soft_tokens=3,
        hard_enabled=True,
        soft_enabled=True,
    )
    router.eval()

    def _forbid_relaxed_topk(*args, **kwargs):
        raise AssertionError("relaxed_topk must not be called in eval() mode")

    monkeypatch.setattr(router, "relaxed_topk", _forbid_relaxed_topk)
    monkeypatch.setattr(pi0_pytorch, "relaxed_topk", _forbid_relaxed_topk)

    images = [torch.randn(2, 8, 16) for _ in range(3)]
    prompt = torch.randn(2, 5, 16)
    out = router(images, prompt, torch.ones(2, 5, dtype=torch.bool))
    assert out.tokens.shape == (2, 7, 16)
    assert torch.all(out.selected_indices[:, 1:] > out.selected_indices[:, :-1])


def test_embed_prefix_mask_behavior_and_layout(patched_tiny_backbone):
    cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=8,
        action_horizon=4,
        max_token_len=10,
        pytorch_compile_mode=None,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=True,
        vision_router_hard_tokens=6,
        vision_router_soft_tokens=4,
    )
    model = pi0_pytorch.PI0Pytorch(cfg)
    model.eval()

    obs = _make_observation(batch_size=2, max_token_len=10, action_dim=8)
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(obs, train=False)

    # 1. Camera masks do not alter router output when VisionRouter is enabled.
    embs_a, pad_a, att_a = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    flipped_img_masks = [torch.zeros_like(m) for m in img_masks]
    embs_b, pad_b, att_b = model.embed_prefix(images, flipped_img_masks, lang_tokens, lang_masks)
    torch.testing.assert_close(embs_a, embs_b)
    assert torch.equal(pad_a, pad_b)
    assert torch.equal(att_a, att_b)

    # 2. Prompt padding tokens must not affect pooled prompt conditioning in VisionRouter.
    lang_tokens_changed_pad = lang_tokens.clone()
    lang_tokens_changed_pad[:, -3:] = 999
    embs_c, _, _ = model.embed_prefix(images, img_masks, lang_tokens_changed_pad, lang_masks)
    torch.testing.assert_close(embs_a[:, :10, :], embs_c[:, :10, :])

    # 3. Prefix order is strictly [Hard(6), Global(4), prompt(10)] and position_ids are contiguous cumsum(pad_mask)-1.
    assert embs_a.shape == (2, 6 + 4 + 10, 32)
    assert torch.all(pad_a[:, :10])
    assert torch.equal(pad_a[:, 10:], lang_masks)
    assert not torch.any(att_a)

    actions = torch.randn(2, 4, 8)
    loss_dict = model(obs, actions, return_loss_components=True)
    assert set(loss_dict.keys()) == {"loss", "action_loss", "router_camera_loss"}
    assert loss_dict["router_camera_loss"].item() == 0.0
    pos_ids = model.paligemma_with_expert.last_forward_kwargs["position_ids"]
    assert pos_ids.shape == (2, 10 + 10 + 4)

    sampled = model.sample_actions("cpu", obs, num_steps=2)
    assert sampled.shape == (2, 4, 8)


def test_masked_relaxed_topk_matches_valid_subsets_and_gradients():
    torch.manual_seed(31)
    scores = torch.randn(3, 9, requires_grad=True)
    valid = torch.tensor([[1,1,0,0,0,1,0,0,0], [0,1,1,1,1,0,0,0,1], [1]*9], dtype=torch.bool)
    weights = torch.randn(3, 2, 9)
    output = pi0_pytorch.relaxed_topk(scores, 2, .7, valid_mask=valid)
    (output * weights).sum().backward()
    assert torch.isfinite(scores.grad).all() and (scores.grad[~valid] == 0).all()
    for row in range(3):
        reference = scores.detach()[row, valid[row]].clone().requires_grad_()
        assignment = pi0_pytorch.relaxed_topk(reference[None], 2, .7)
        torch.testing.assert_close(output[row, :, valid[row]], assignment[0], rtol=0, atol=0)
        (assignment[0] * weights[row, :, valid[row]]).sum().backward()
        torch.testing.assert_close(scores.grad[row, valid[row]], reference.grad, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("training", [False, True])
def test_masked_router_excludes_invalid_cameras_and_gradients(training):
    torch.manual_seed(17)
    router = pi0_pytorch.VisionRouter(8, hard_tokens=3, soft_tokens=2, mask_invalid_cameras=True)
    router.train(training)
    images = [torch.randn(2, 4, 8, requires_grad=True) for _ in range(3)]
    masks = [torch.full((2,), i < 2, dtype=torch.bool) for i in range(3)]
    prompt = torch.randn(2, 3, 8)
    out = router(images, prompt, image_masks=masks)
    assert (out.selected_indices < 8).all() and (out.assignment[:, 8:] == 0).all()
    torch.testing.assert_close(out.assignment.sum(dim=1), torch.ones(2, 2))
    assert torch.isfinite(out.tokens).all() and (out.camera_weights[:, :, 2] == 0).all()
    (out.tokens * torch.randn_like(out.tokens)).sum().backward()
    assert (images[2].grad == 0).all() and (router.camera_embedding.grad[2] == 0).all()
    assert sum(p.grad.abs().sum() for p in router.compression_assignment.parameters()) > 0
    changed = [x.detach().clone() for x in images]
    changed[2].add_(10000)
    torch.testing.assert_close(router(changed, prompt, image_masks=masks).tokens, out.tokens, rtol=0, atol=0)


def test_masked_router_supports_different_camera_masks_and_rejects_empty():
    router = pi0_pytorch.VisionRouter(8, hard_tokens=3, soft_tokens=2, mask_invalid_cameras=True)
    images = [torch.randn(3, 4, 8) for _ in range(3)]
    prompt = torch.randn(3, 3, 8)
    cameras = torch.tensor([[1,1,0], [0,1,0], [1,0,1]], dtype=torch.bool)
    out = router(images, prompt, image_masks=list(cameras.unbind(dim=1)))
    valid = cameras[:, :, None].expand(3, 3, 4).reshape(3, 12)
    assert valid.gather(1, out.selected_indices).all()
    assert (out.assignment[~valid] == 0).all()
    with pytest.raises(ValueError, match="requires one image mask"):
        router(images, prompt)
    with pytest.raises(ValueError, match="insufficient valid"):
        router(images, prompt, image_masks=[torch.zeros(3, dtype=torch.bool)]*3)


def test_all_valid_mask_keeps_legacy_output_exact():
    router = pi0_pytorch.VisionRouter(8, hard_tokens=3, soft_tokens=2)
    images = [torch.randn(2, 4, 8) for _ in range(3)]
    prompt = torch.randn(2, 3, 8)
    original = router(images, prompt)
    router.mask_invalid_cameras = True
    masked = router(images, prompt, image_masks=[torch.ones(2, dtype=torch.bool)]*3)
    torch.testing.assert_close(masked.tokens, original.tokens, rtol=0, atol=0)
    torch.testing.assert_close(masked.assignment, original.assignment, rtol=0, atol=0)


def test_mask_option_metadata_and_model_prefix(tmp_path, patched_tiny_backbone):
    cfg = pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=8, action_horizon=4, max_token_len=10,
                             pytorch_compile_mode=None, vision_router_hard_tokens=6, vision_router_soft_tokens=4)
    assert "mask_invalid_cameras" not in pi0_pytorch.build_model_config_dict(cfg)["vision_router"]
    cfg = dataclasses.replace(cfg, vision_router_mask_invalid_cameras=True)
    assert pi0_pytorch.build_model_config_dict(cfg)["format_version"] == 2
    model = pi0_pytorch.PI0Pytorch(cfg).eval()
    obs = _make_observation(max_token_len=10)
    images, masks, tokens, token_masks, _ = model._preprocess_observation(obs, train=False)
    prefix, _, _ = model.embed_prefix(images, masks, tokens, token_masks)
    assert prefix.shape[1] == 20 and (model.last_router_output.selected_indices < 32).all()
    safetensors.torch.save_model(model, str(tmp_path / "model.safetensors"))
    pi0_pytorch.save_model_config(cfg, tmp_path)
    base = dataclasses.replace(cfg, vision_router_mask_invalid_cameras=False)
    restored = pi0_pytorch.resolve_checkpoint_model_config(base, tmp_path)
    assert restored.vision_router_mask_invalid_cameras
    loaded = base.load_pytorch(SimpleNamespace(model=base), str(tmp_path))
    assert loaded.vision_router.mask_invalid_cameras
    pi0_pytorch.save_model_config(base, tmp_path)
    assert not pi0_pytorch.resolve_checkpoint_model_config(cfg, tmp_path).vision_router_mask_invalid_cameras


def test_router_off_restores_full_visual_prefix(patched_tiny_backbone):
    cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=8,
        action_horizon=4,
        max_token_len=10,
        pytorch_compile_mode=None,
        vision_router_hard_enabled=False,
        vision_router_soft_enabled=False,
    )
    model = pi0_pytorch.PI0Pytorch(cfg)
    assert model.vision_router is None

    obs = _make_observation(batch_size=2, max_token_len=10, action_dim=8)
    images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(obs, train=False)
    embs, pad_masks, _ = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    p = _TinyPaliGemmaWithExpert.patches_per_camera
    assert embs.shape == (2, 3 * p + 10, 32)
    # Third camera mask is False, so its patch positions in pad_masks must be False.
    assert torch.all(pad_masks[:, : 2 * p])
    assert not torch.any(pad_masks[:, 2 * p : 3 * p])


def test_load_pretrained_weights_and_checkpoint_resolution(tmp_path, patched_tiny_backbone):
    base_cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=8,
        action_horizon=4,
        max_token_len=10,
        pytorch_compile_mode=None,
        vision_router_hard_enabled=False,
        vision_router_soft_enabled=False,
    )
    teacher_model = pi0_pytorch.PI0Pytorch(base_cfg)
    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    safetensors.torch.save_model(teacher_model, str(teacher_dir / "model.safetensors"))
    pi0_pytorch.save_model_config(base_cfg, teacher_dir)

    # Initializing a new Dual Router model from router-free teacher weights succeeds.
    routed_cfg = dataclasses.replace(
        base_cfg,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=True,
        vision_router_hard_tokens=8,
        vision_router_soft_tokens=6,
    )
    routed_model = pi0_pytorch.PI0Pytorch(routed_cfg)
    missing, unexpected = pi0_pytorch.load_pretrained_weights(routed_model, teacher_dir)
    assert not unexpected
    assert len(missing) == len(routed_model.vision_router.state_dict())

    # Partial router weights or missing backbone weights must fail.
    broken_state = teacher_model.state_dict()
    broken_state.pop("action_in_proj.weight")
    broken_path = tmp_path / "broken_backbone.safetensors"
    safetensors.torch.save_file(broken_state, str(broken_path))
    with pytest.raises(RuntimeError, match="Missing key"):
        pi0_pytorch.load_pretrained_weights(routed_model, broken_path)

    partial_router_state = routed_model.state_dict()
    partial_router_state.pop("vision_router.selection_score.2.weight")
    partial_router_path = tmp_path / "partial_router.safetensors"
    safetensors.torch.save_file(partial_router_state, str(partial_router_path))
    with pytest.raises(RuntimeError, match="Missing key"):
        pi0_pytorch.load_pretrained_weights(routed_model, partial_router_path)

    # Legacy camera-weight router keys must be rejected.
    legacy_state = teacher_model.state_dict()
    legacy_state["vision_router.logits"] = torch.zeros(3)
    legacy_path = tmp_path / "legacy_logits.safetensors"
    safetensors.torch.save_file(legacy_state, str(legacy_path))
    with pytest.raises(ValueError, match="vision_router.logits"):
        pi0_pytorch.resolve_checkpoint_model_config(routed_cfg, legacy_path)

    # Saving and resolving custom H/G and single-branch checkpoints via BaseModelConfig.load_pytorch.
    custom_dir = tmp_path / "custom_ckpt"
    custom_dir.mkdir()
    safetensors.torch.save_model(routed_model, str(custom_dir / "model.safetensors"))
    pi0_pytorch.save_model_config(routed_cfg, custom_dir)

    default_train_cfg = SimpleNamespace(model=pi0_config.Pi0Config(pi05=True, dtype="float32", action_dim=8, action_horizon=4, max_token_len=10, pytorch_compile_mode=None))
    loaded = default_train_cfg.model.load_pytorch(default_train_cfg, str(custom_dir))
    assert loaded.vision_router is not None
    assert loaded.vision_router.hard_tokens == 8
    assert loaded.vision_router.soft_tokens == 6
    assert loaded.vision_router.router_dim == 14

    # Router-free checkpoint resolves to full-token model (vision_router=None).
    loaded_teacher = default_train_cfg.model.load_pytorch(default_train_cfg, str(teacher_dir))
    assert loaded_teacher.vision_router is None


def test_legacy_checkpoint_without_model_config_json_and_mismatch_guards(tmp_path, patched_tiny_backbone):
    base_cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=8,
        action_horizon=4,
        max_token_len=10,
        pytorch_compile_mode=None,
    )

    # Legacy Dual Router checkpoint without model_config.json (H=128 default, G=64 inferred from soft weights -> R=192).
    legacy_dual_cfg = dataclasses.replace(
        base_cfg,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=True,
        vision_router_hard_tokens=128,
        vision_router_soft_tokens=64,
    )
    legacy_dual_model = pi0_pytorch.PI0Pytorch(legacy_dual_cfg)
    legacy_dual_path = tmp_path / "legacy_dual.safetensors"
    safetensors.torch.save_model(legacy_dual_model, str(legacy_dual_path))

    resolved_dual = pi0_pytorch.resolve_checkpoint_model_config(base_cfg, legacy_dual_path)
    assert resolved_dual.vision_router_hard_enabled is True
    assert resolved_dual.vision_router_soft_enabled is True
    assert resolved_dual.vision_router_hard_tokens == 128
    assert resolved_dual.vision_router_soft_tokens == 64
    assert resolved_dual.vision_router_dim == 192

    # Non-128 Hard checkpoint without model_config.json must fail resolution.
    non128_cfg = dataclasses.replace(
        base_cfg,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=True,
        vision_router_hard_tokens=224,
        vision_router_soft_tokens=96,
    )
    non128_model = pi0_pytorch.PI0Pytorch(non128_cfg)
    non128_path = tmp_path / "non128_no_json.safetensors"
    safetensors.torch.save_model(non128_model, str(non128_path))
    with pytest.raises(ValueError, match="Cannot infer VisionRouter token counts"):
        pi0_pytorch.resolve_checkpoint_model_config(base_cfg, non128_path)

    # Mismatched model_config.json branch flags vs safetensors weights must fail.
    mismatch_dir = tmp_path / "mismatch_ckpt"
    mismatch_dir.mkdir()
    safetensors.torch.save_model(non128_model, str(mismatch_dir / "model.safetensors"))
    hard_only_cfg = dataclasses.replace(
        base_cfg,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=False,
        vision_router_hard_tokens=224,
        vision_router_soft_tokens=0,
    )
    pi0_pytorch.save_model_config(hard_only_cfg, mismatch_dir)
    with pytest.raises(ValueError, match="do not match weights"):
        pi0_pytorch.resolve_checkpoint_model_config(base_cfg, mismatch_dir)


def test_trainer_budget_archiving_final_step_save_and_resume_topology(tmp_path, monkeypatch, patched_tiny_backbone):
    import numpy as np
    from openpi.training import config as _config
    from openpi.training import token_budget
    import scripts.train_pytorch as train_pytorch

    valid_mask = np.ones((8, 512), dtype=bool)
    relevance = np.ones((8, 512), dtype=np.float32) / 512.0
    summary, hard_per_sample = token_budget.compute_token_budget_summary(
        relevance,
        valid_mask,
        coverage_target=0.25,
        sample_quantile=0.95,
        alignment=16,
        global_coverage_upper=0.50,
        global_min_tokens=16,
    )
    budget_dir = tmp_path / "budget_src"
    report = token_budget.build_budget_report(
        config_name="pi05_libero",
        repo_id="fake",
        checkpoint="/tmp/teacher",
        attention_source="vit",
        model_config_dict={"format_version": 1},
        summary=summary,
        layers=(0, 1, 2, 3),
    )
    budget_json, _ = token_budget.save_budget_artifacts(
        budget_dir / "token_budget_vit.json",
        report,
        sample_indices=np.arange(8),
        relevance=relevance,
        valid_mask=valid_mask,
        hard_tokens_per_sample=hard_per_sample,
    )

    run_out_dir = tmp_path / "run_out"
    monkeypatch.setenv("RUN_OUTPUT_DIR", str(run_out_dir))

    base_train_cfg = _config.TrainConfig(
        name="pi05_libero",
        exp_name="unit_test_exp",
        checkpoint_base_dir=str(tmp_path / "ckpts"),
        assets_base_dir=str(tmp_path / "assets"),
        pytorch_training_precision="float32",
        num_train_steps=5,
        save_interval=10,
        overwrite=True,
        wandb_enabled=False,
        vision_router_budget_path=str(budget_json),
        vision_router_calibration_source="vit",
        vision_router_attention_coverage=0.25,
        vision_router_budget_quantile=0.95,
        vision_router_global_coverage_upper=0.50,
        vision_router_global_min_tokens=16,
        vision_router_token_alignment=16,
        model=pi0_config.Pi0Config(
            pi05=True,
            dtype="float32",
            action_dim=8,
            action_horizon=4,
            max_token_len=10,
            pytorch_compile_mode=None,
        ),
    )

    resolved_cfg, model_cfg, resuming = train_pytorch.prepare_training_config_and_checkpoint_dir(
        base_train_cfg,
        is_main=True,
        use_ddp=False,
    )
    assert not resuming
    assert model_cfg.vision_router_hard_tokens == summary["recommended_hard_tokens"]
    assert model_cfg.vision_router_soft_tokens == summary["recommended_global_tokens"]
    assert resolved_cfg.vision_router_budget_path == str(
        resolved_cfg.checkpoint_dir / "calibration" / "token_budget_vit.json"
    )
    assert (resolved_cfg.checkpoint_dir / "calibration" / "training_budget.json").exists()
    assert (run_out_dir / "calibration" / "training_budget.json").exists()

    model = pi0_pytorch.PI0Pytorch(model_cfg)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dummy_data_cfg = SimpleNamespace(norm_stats=None, asset_id=None)

    # Final step global_step == num_train_steps (5) must save checkpoint even when save_interval=10.
    train_pytorch.save_checkpoint(model, optim, 5, resolved_cfg, True, dummy_data_cfg)
    step5_dir = resolved_cfg.checkpoint_dir / "5"
    assert (step5_dir / "model.safetensors").exists()
    assert (step5_dir / "model_config.json").exists()
    assert (step5_dir / "optimizer.pt").exists()
    assert (step5_dir / "metadata.pt").exists()

    # Resume must recover the saved H/G topology without budget path.
    resume_cfg = dataclasses.replace(
        base_train_cfg,
        overwrite=False,
        resume=True,
        vision_router_budget_path=None,
        model=pi0_config.Pi0Config(
            pi05=True,
            dtype="float32",
            action_dim=8,
            action_horizon=4,
            max_token_len=10,
            pytorch_compile_mode=None,
            vision_router_hard_tokens=128,
            vision_router_soft_tokens=128,
        ),
    )
    _, resumed_model_cfg, is_resuming = train_pytorch.prepare_training_config_and_checkpoint_dir(
        resume_cfg,
        is_main=True,
        use_ddp=False,
    )
    assert is_resuming
    assert resumed_model_cfg.vision_router_hard_tokens == model_cfg.vision_router_hard_tokens
    assert resumed_model_cfg.vision_router_soft_tokens == model_cfg.vision_router_soft_tokens
