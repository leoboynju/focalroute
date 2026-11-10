import dataclasses
import json
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models import pi0_config
from openpi.models_pytorch import pi0_pytorch
from openpi.training import attention_visualization as _vis
from openpi.training import token_budget
import scripts.render_token_attention as render_mod
import scripts.rerender_token_attention as rerender_mod


class _MockSiglipSelfAttn(nn.Module):
    def __init__(self, width: int, num_patches: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.num_patches = num_patches

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        b = hidden_states.shape[0]
        out = self.q_proj(hidden_states)
        if not output_attentions:
            return out, None
        grid = torch.arange(1, self.num_patches + 1, device=hidden_states.device, dtype=torch.float32)
        weights = grid / grid.sum()
        attn = weights.view(1, 1, 1, self.num_patches).expand(b, 2, self.num_patches, self.num_patches).contiguous()
        return out, attn


class _MockSiglipLayer(nn.Module):
    def __init__(self, width: int, num_patches: int):
        super().__init__()
        self.self_attn = _MockSiglipSelfAttn(width, num_patches)


class _MockGemmaSelfAttn(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.head_dim = width

    def forward(self, hidden_states, total_keys: int):
        b, q_len, _ = hidden_states.shape
        out = self.q_proj(hidden_states)
        grid = torch.arange(1, total_keys + 1, device=hidden_states.device, dtype=torch.float32)
        weights = (grid / grid.sum()).view(1, 1, 1, total_keys).expand(b, 2, q_len, total_keys).contiguous()
        return out, weights


class _MockGemmaLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.self_attn = _MockGemmaSelfAttn(width)


class _MockPaliGemmaWithExpert(nn.Module):
    patches_per_camera = 16

    def __init__(self, vlm_config, action_expert_config, use_adarms=None, precision="float32"):
        super().__init__()
        w = vlm_config.width
        ew = action_expert_config.width
        vit_layers = nn.ModuleList([_MockSiglipLayer(w, self.patches_per_camera) for _ in range(4)])
        vision_model = SimpleNamespace(
            encoder=SimpleNamespace(layers=vit_layers),
            config=SimpleNamespace(_attn_implementation="sdpa", output_attentions=False),
        )
        vision_tower = SimpleNamespace(
            vision_model=vision_model,
            config=SimpleNamespace(_attn_implementation="sdpa", output_attentions=False),
            gradient_checkpointing=False,
        )
        vlm_layers = nn.ModuleList([_MockGemmaLayer(w) for _ in range(4)])
        language_model = SimpleNamespace(
            layers=vlm_layers,
            config=SimpleNamespace(_attn_implementation="sdpa"),
            gradient_checkpointing=False,
        )
        self.paligemma = SimpleNamespace(vision_tower=vision_tower, language_model=language_model)

        expert_layers = nn.ModuleList([_MockGemmaLayer(ew) for _ in range(4)])
        expert_model = SimpleNamespace(
            layers=expert_layers,
            config=SimpleNamespace(_attn_implementation="sdpa"),
            gradient_checkpointing=False,
        )
        self.gemma_expert = SimpleNamespace(model=expert_model)

        self.vit_layers = vit_layers
        self.vlm_layers = vlm_layers
        self.expert_layers = expert_layers
        self.img_proj = nn.Linear(3, w)
        self.tok_emb = nn.Embedding(512, w)

    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        b = image.shape[0]
        x = image.float().reshape(b, 3, -1).permute(0, 2, 1)[:, : self.patches_per_camera, :]
        h = self.img_proj(x)
        out_attn = bool(self.paligemma.vision_tower.config.output_attentions)
        for layer in self.vit_layers:
            h, _ = layer.self_attn(h, output_attentions=out_attn)
        return h

    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(tokens.clamp(min=0, max=511))

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        adarms_cond=None,
    ):
        prefix_embs, suffix_embs = inputs_embeds
        if suffix_embs is None:
            h = prefix_embs.float()
            total_k = h.shape[1]
            for layer in self.vlm_layers:
                h, _ = layer.self_attn(h, total_k)
            cache = [(h.detach(), h.detach())]
            return [h, None], cache
        h = suffix_embs.float()
        prefix_len = past_key_values[0][0].shape[1] if past_key_values else 0
        total_k = prefix_len + h.shape[1]
        for layer in self.expert_layers:
            h, _ = layer.self_attn(h, total_k)
        return [None, h], None


class _FakeEpisodeDataset:
    def __init__(self, num_frames: int = 4, max_token_len: int = 8, action_dim: int = 8, action_horizon: int = 4):
        self.num_frames = num_frames
        self.max_token_len = max_token_len
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.episode_data_index = {"from": np.array([0, 2]), "to": np.array([2, 4])}

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(100 + int(idx))
        img = rng.uniform(-1.0, 1.0, size=(224, 224, 3)).astype(np.float32)
        zero_img = np.zeros((224, 224, 3), dtype=np.float32)
        prompt = np.ones((self.max_token_len,), dtype=np.int32)
        prompt_mask = np.array([True] * (self.max_token_len - 2) + [False, False], dtype=bool)
        return {
            "image": {
                "base_0_rgb": img,
                "left_wrist_0_rgb": img,
                "right_wrist_0_rgb": zero_img,
            },
            "image_mask": {
                "base_0_rgb": np.bool_(True),
                "left_wrist_0_rgb": np.bool_(True),
                "right_wrist_0_rgb": np.bool_(False),
            },
            "state": np.zeros((self.action_dim,), dtype=np.float32),
            "tokenized_prompt": prompt,
            "tokenized_prompt_mask": prompt_mask,
            "actions": np.zeros((self.action_horizon, self.action_dim), dtype=np.float32),
        }


@pytest.fixture
def patched_attention_env(monkeypatch):
    monkeypatch.setattr(
        pi0_pytorch._gemma,
        "get_config",
        lambda variant: SimpleNamespace(width=16, depth=4, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8),
    )
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", _MockPaliGemmaWithExpert)


def test_visual_token_layout_scatter_and_source_contribution():
    layout = _vis.VisualTokenLayout(num_cameras=3, grid_height=4, grid_width=4, image_height=224, image_width=224)
    assert layout.patches_per_camera == 16
    assert layout.total_visual_tokens == 48

    routed_attn = np.array([[0.2, 0.3, 0.1, 0.15, 0.25]], dtype=np.float32)
    selected_idx = np.array([[2, 19, 35]], dtype=np.int64)
    hard_map, soft_mass = layout.scatter_routed_attention(
        routed_attn,
        selected_idx,
        hard_tokens=3,
        soft_tokens=2,
    )
    assert hard_map.shape == (1, 48)
    assert np.isclose(hard_map[0, 2], 0.2)
    assert np.isclose(hard_map[0, 19], 0.3)
    assert np.isclose(hard_map[0, 35], 0.1)
    assert np.isclose(hard_map.sum(), 0.6)
    assert np.isclose(soft_mass[0], 0.4)

    assignment = np.ones((2, 48, 4), dtype=np.float32) / 48.0
    contrib = layout.soft_source_contribution(assignment)
    assert contrib.shape == (2, 48)
    assert np.allclose(contrib, 1.0 / 48.0)


def test_render_and_rerender_token_attention_end_to_end(tmp_path, monkeypatch, patched_attention_env):
    base_train_cfg = render_mod._config.get_config("pi05_libero")
    fake_train_cfg = dataclasses.replace(
        base_train_cfg,
        assets_base_dir=str(tmp_path / "assets"),
        seed=42,
        model=pi0_config.Pi0Config(
            pi05=True,
            dtype="float32",
            action_dim=8,
            action_horizon=4,
            max_token_len=8,
            pytorch_compile_mode=None,
            vision_router_hard_enabled=False,
            vision_router_soft_enabled=False,
        ),
    )
    monkeypatch.setattr(render_mod._config, "get_config", lambda name: fake_train_cfg)
    monkeypatch.setattr(rerender_mod._config, "get_config", lambda name: fake_train_cfg)

    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    teacher_model = pi0_pytorch.PI0Pytorch(fake_train_cfg.model)
    safetensors.torch.save_model(teacher_model, str(teacher_dir / "model.safetensors"))
    pi0_pytorch.save_model_config(fake_train_cfg.model, teacher_dir)

    dataset = _FakeEpisodeDataset()
    out_c80 = tmp_path / "attention_c80"
    manifest_c80 = render_mod.render_episodes(
        checkpoint_dir=teacher_dir,
        output_dir=out_c80,
        config_name="pi05_libero",
        episode_ids=[0, 1],
        batch_size=2,
        fps=10,
        coverage=0.80,
        alignment=8,
        global_coverage_upper=0.90,
        global_min_tokens=8,
        dataset=dataset,
    )
    assert manifest_c80["status"] == "completed"
    assert manifest_c80["total_frames"] == 4
    for s in ("vit", "vlm", "action"):
        assert (out_c80 / f"token_budget_{s}.json").exists()
        assert (out_c80 / f"token_budget_{s}.npz").exists()

    out_c50 = tmp_path / "attention_c50"
    manifest_c50 = rerender_mod.rerender_from_cache(
        source_dir=out_c80,
        output_dir=out_c50,
        coverage=0.50,
        expected_checkpoint_dir=teacher_dir,
        dataset=dataset,
    )
    assert manifest_c50["coverage"] == 0.50

    for s in ("vit", "vlm", "action"):
        with np.load(out_c80 / "episodes" / "episode_0000.npz") as d80, np.load(
            out_c50 / "episodes" / "episode_0000.npz"
        ) as d50:
            np.testing.assert_allclose(d80[f"{s}_raw_scores"], d50[f"{s}_raw_scores"])
            np.testing.assert_allclose(d80[f"{s}_relevance"], d50[f"{s}_relevance"])
            assert np.all(d50[f"{s}_selected_counts"] <= d80[f"{s}_selected_counts"])
            assert np.all(d50[f"{s}_selected_counts"] < d80[f"{s}_selected_counts"])

    # Reject router-containing checkpoints in render and rerender.
    routed_cfg = dataclasses.replace(
        fake_train_cfg.model,
        vision_router_hard_enabled=True,
        vision_router_soft_enabled=True,
        vision_router_hard_tokens=4,
        vision_router_soft_tokens=4,
    )
    routed_model = pi0_pytorch.PI0Pytorch(routed_cfg)
    routed_dir = tmp_path / "routed_ckpt"
    routed_dir.mkdir()
    safetensors.torch.save_model(routed_model, str(routed_dir / "model.safetensors"))
    pi0_pytorch.save_model_config(routed_cfg, routed_dir)

    with pytest.raises(ValueError, match="router-free"):
        render_mod.render_episodes(
            checkpoint_dir=routed_dir,
            output_dir=tmp_path / "bad_out",
            dataset=dataset,
        )

    with pytest.raises(ValueError, match="expected_checkpoint_dir"):
        rerender_mod.rerender_from_cache(
            source_dir=out_c80,
            output_dir=tmp_path / "bad_rerender",
            coverage=0.50,
            expected_checkpoint_dir=routed_dir,
            dataset=dataset,
        )

    # Reject manifest with mismatched model_config schema or incomplete status.
    tampered_dir = tmp_path / "tampered_src"
    tampered_dir.mkdir()
    manifest_tampered = json.loads((out_c80 / "manifest.json").read_text())
    manifest_tampered["model_config"]["model"].pop("discrete_state_input")
    (tampered_dir / "manifest.json").write_text(json.dumps(manifest_tampered))
    with pytest.raises(ValueError, match="model_config"):
        rerender_mod.rerender_from_cache(
            source_dir=tampered_dir,
            output_dir=tmp_path / "bad_schema_out",
            coverage=0.50,
            expected_checkpoint_dir=teacher_dir,
            dataset=dataset,
        )
