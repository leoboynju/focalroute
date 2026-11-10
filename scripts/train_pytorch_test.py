import dataclasses
from types import SimpleNamespace

import pytest
import torch

from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as config_lib
from scripts import train_pytorch as trainer


def test_original_rope_full_backend_real_gradients_cache_and_resume(small_model, tmp_path):
    model = small_model
    model.config = dataclasses.replace(model.config, vision_router_mask_invalid_cameras=True, vision_router_preserve_hard_rope=True)
    model.vision_router.mask_invalid_cameras = True
    trainer.configure_trainable_parameters(model, "router_llm_expert")
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    assert frozen and all("vision_tower" in n or "multi_modal_projector" in n for n in frozen)
    for module in (model.vision_router, model.paligemma_with_expert.paligemma.language_model,
                   model.paligemma_with_expert.gemma_expert, model.action_in_proj, model.action_out_proj,
                   model.time_mlp_in, model.time_mlp_out):
        assert all(p.requires_grad for p in module.parameters())
    images = {key: torch.rand(2, 3, 224, 224) * 2 - 1 for key in model_lib.IMAGE_KEYS}
    observation = model_lib.Observation(
        images=images, image_masks={k: torch.tensor([i < 2, i == 1]) for i, k in enumerate(images)},
        state=torch.zeros(2, 8), tokenized_prompt=torch.arange(8)[None].expand(2, -1),
        tokenized_prompt_mask=torch.tensor([[True] * 5 + [False] * 3, [True] * 3 + [False] * 5]),
    )
    actions, noise, time = torch.randn(2, 4, 8), torch.randn(2, 4, 8), torch.tensor([.4, .6])
    model(observation, actions, noise=noise, time=time).mean().backward()
    for layers in (model.paligemma_with_expert.paligemma.language_model.layers, model.paligemma_with_expert.gemma_expert.model.layers):
        for layer in layers:
            assert sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None) > 0
    for module in (model.vision_router.selection_score, model.vision_router.compression_assignment,
                   model.paligemma_with_expert.paligemma.language_model.embed_tokens,
                   model.action_in_proj, model.action_out_proj, model.time_mlp_in, model.time_mlp_out):
        assert sum(p.grad.abs().sum() for p in module.parameters() if p.grad is not None) > 0
    optimizer = torch.optim.AdamW(trainer.get_model_parameters(model), lr=1e-3)
    torch.nn.utils.clip_grad_norm_(list(trainer.get_model_parameters(model)), 1., error_if_nonfinite=True)
    optimizer.step()
    assert all(torch.equal(dict(model.named_parameters())[n], v) for n, v in frozen.items())

    model.eval()
    with torch.no_grad():
        ims, masks, tokens, token_masks, state = model._preprocess_observation(observation, train=False)
        prefix, pm, pa, positions, offset = model.embed_prefix(ims, masks, tokens, token_masks, return_positions=True)
        suffix, sm, sa, cond = model.embed_suffix(state, noise, time)
        prefix_mask = model._prepare_attention_masks_4d(pi0_pytorch.make_att_2d_masks(pm, pa))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(attention_mask=prefix_mask, position_ids=positions,
                                                      inputs_embeds=[prefix, None], use_cache=True)
        cached = model.denoise_step(state, pm, cache, noise, time, prefix_position_offset=offset)
        full_mask = model._prepare_attention_masks_4d(pi0_pytorch.make_att_2d_masks(torch.cat([pm, sm], 1), torch.cat([pa, sa], 1)))
        full_pos = torch.cat([positions, offset + torch.cumsum(sm, 1) - 1], 1)
        (_, hidden), _ = model.paligemma_with_expert.forward(attention_mask=full_mask, position_ids=full_pos,
                                                            inputs_embeds=[prefix, suffix], use_cache=False, adarms_cond=[None, cond])
        full = model.action_out_proj(hidden[:, -model.config.action_horizon:].float())
        torch.testing.assert_close(cached, full, rtol=2e-5, atol=2e-5)
        sampled = model.sample_actions(torch.device("cpu"), observation, noise=noise, num_steps=2)
        assert sampled.shape == (2, 4, 8) and torch.isfinite(sampled).all()

    cfg = config_lib.TrainConfig(name="pi05_libero", exp_name="original_rope", model=model.config,
                                checkpoint_base_dir=str(tmp_path), pytorch_train_mode="router_llm_expert",
                                pytorch_training_precision="float32", num_train_steps=1, wandb_enabled=False)
    cfg, _, _ = trainer.prepare_training_config_and_checkpoint_dir(cfg, is_main=True, use_ddp=False)
    trainer.save_checkpoint(model, optimizer, 1, cfg, True, SimpleNamespace(norm_stats=None, asset_id=None))
    resumed, restored, _ = trainer.prepare_training_config_and_checkpoint_dir(
        dataclasses.replace(cfg, resume=True, pytorch_train_mode=None,
                            model=dataclasses.replace(cfg.model, vision_router_preserve_hard_rope=False)),
        is_main=True, use_ddp=False)
    assert restored.vision_router_preserve_hard_rope and resumed.pytorch_train_mode == "router_llm_expert"
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer.prepare_training_config_and_checkpoint_dir(dataclasses.replace(cfg, resume=True, pytorch_train_mode="router_first4"),
                                                           is_main=True, use_ddp=False)
    assert trainer.load_checkpoint(model, optimizer, cfg.checkpoint_dir, torch.device("cpu")) == 1

    legacy = dataclasses.replace(cfg.model, vision_router_preserve_hard_rope=False)
    pi0_pytorch.save_model_config(legacy, cfg.checkpoint_dir / "1")
    with pytest.raises(ValueError, match="Cannot enable original Hard RoPE"):
        trainer.prepare_training_config_and_checkpoint_dir(
            dataclasses.replace(cfg, resume=True), is_main=True, use_ddp=False,
        )


@pytest.fixture
def small_model(monkeypatch):
    # Use real SigLIP/Gemma attention and checkpointing, with two frozen blocks after block 3.
    monkeypatch.setattr(
        pi0_pytorch._gemma,  # noqa: SLF001
        "get_config",
        lambda _: SimpleNamespace(width=32, depth=6, mlp_dim=64, num_heads=8, num_kv_heads=1, head_dim=16),
    )
    paligemma_class = gemma_pytorch.PaliGemmaForConditionalGeneration
    expert_class = gemma_pytorch.GemmaForCausalLM

    def make_paligemma(config):
        config.text_config.vocab_size = 128
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.projection_dim = 32
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 2
        return paligemma_class(config)

    def make_expert(config):
        config.vocab_size = 128
        return expert_class(config)

    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", make_paligemma)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", make_expert)
    cfg = pi0_config.Pi0Config(
        pi05=True, dtype="float32", action_dim=8, action_horizon=4, max_token_len=8,
        vision_router_hard_tokens=4, vision_router_soft_tokens=4, pytorch_compile_mode=None,
    )
    model = pi0_pytorch.PI0Pytorch(cfg)
    model.gradient_checkpointing_enable()
    return model


def test_router_first4_flow_gradients_and_resume(small_model, tmp_path):
    model = small_model
    cfg = config_lib.TrainConfig(
        name="pi05_libero", exp_name="first4", model=model.config, checkpoint_base_dir=str(tmp_path),
        pytorch_training_precision="float32", pytorch_train_mode="router_first4",
        num_train_steps=1, save_interval=10, wandb_enabled=False,
    )
    cfg, _, _ = trainer.prepare_training_config_and_checkpoint_dir(cfg, is_main=True, use_ddp=False)
    trainer.configure_trainable_parameters(model, cfg.pytorch_train_mode)
    parameters = list(trainer.get_model_parameters(model))
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    trainable_before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    images = {key: torch.rand(1, 3, 224, 224) * 2 - 1 for key in model_lib.IMAGE_KEYS}
    observation = model_lib.Observation(
        images=images, image_masks={key: torch.tensor([i < 2]) for i, key in enumerate(images)},
        state=torch.zeros(1, 8), tokenized_prompt=torch.arange(8)[None],
        tokenized_prompt_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    actions = torch.randn(1, 4, 8)
    noise = torch.randn_like(actions)
    time = torch.tensor([0.4])

    def step():
        torch.manual_seed(17)
        losses = model(observation, actions, noise=noise, time=time, return_loss_components=True)
        torch.testing.assert_close(losses["loss"], losses["action_loss"])
        losses["loss"].backward()
        for head in (model.vision_router.selection_score, model.vision_router.compression_assignment):
            assert sum(p.grad.abs().sum() for p in head.parameters() if p.grad is not None) > 0
        for layers in (
            model.paligemma_with_expert.paligemma.language_model.layers,
            model.paligemma_with_expert.gemma_expert.model.layers,
        ):
            for i, layer in enumerate(layers):
                if i < 4:
                    assert sum(p.grad.abs().sum() for p in layer.parameters() if p.grad is not None) > 0
                else:
                    assert all(p.grad is None for p in layer.parameters())
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    step()
    for name, p in model.named_parameters():
        if name in frozen_before:
            assert torch.equal(p, frozen_before[name]), name
    assert any(not torch.equal(p, trainable_before[name]) for name, p in model.named_parameters() if p.requires_grad)
    assert set(optimizer.state) == set(parameters)
    trainer.save_checkpoint(model, optimizer, 1, cfg, True, SimpleNamespace(norm_stats=None, asset_id=None))

    resumed_cfg, _, resuming = trainer.prepare_training_config_and_checkpoint_dir(
        dataclasses.replace(cfg, resume=True, pytorch_train_mode=None), is_main=True, use_ddp=False,
    )
    assert resuming and resumed_cfg.pytorch_train_mode == "router_first4"
    with pytest.raises(ValueError, match="Cannot enable camera masking"):
        trainer.prepare_training_config_and_checkpoint_dir(
            dataclasses.replace(cfg, resume=True, model=dataclasses.replace(cfg.model, vision_router_mask_invalid_cameras=True)),
            is_main=True, use_ddp=False,
        )
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer.prepare_training_config_and_checkpoint_dir(
            dataclasses.replace(cfg, resume=True, pytorch_train_mode="full"), is_main=True, use_ddp=False,
        )

    step()
    expected = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(parameters, lr=9e-2)
    assert trainer.load_checkpoint(model, optimizer, cfg.checkpoint_dir, torch.device("cpu")) == 1
    step()
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, expected[name], rtol=0, atol=0)

    trainer.configure_trainable_parameters(model, "full")
    with pytest.raises(ValueError, match="names/order"):
        trainer.load_checkpoint(model, optimizer, cfg.checkpoint_dir, torch.device("cpu"))


def test_legacy_full_resume_and_partial_scope_validation(small_model, tmp_path):
    model = small_model
    cfg = config_lib.TrainConfig(
        name="pi05_libero", exp_name="legacy", model=model.config, checkpoint_base_dir=str(tmp_path),
        pytorch_training_precision="float32", num_train_steps=1, wandb_enabled=False,
    )
    cfg, _, _ = trainer.prepare_training_config_and_checkpoint_dir(cfg, is_main=True, use_ddp=False)
    assert cfg.pytorch_train_mode == "full"
    trainer.configure_trainable_parameters(model, "full")
    optimizer = torch.optim.AdamW(trainer.get_model_parameters(model))
    trainer.save_checkpoint(model, optimizer, 1, cfg, True, SimpleNamespace(norm_stats=None, asset_id=None))
    path = cfg.checkpoint_dir / "1" / "metadata.pt"
    metadata = torch.load(path, weights_only=False)
    del metadata["config"]["pytorch_train_mode"]
    del metadata["trainable_parameter_names"]
    torch.save(metadata, path)
    resumed_cfg, _, _ = trainer.prepare_training_config_and_checkpoint_dir(
        dataclasses.replace(cfg, resume=True, pytorch_train_mode=None), is_main=True, use_ddp=False,
    )
    assert resumed_cfg.pytorch_train_mode == "full"
    assert trainer.load_checkpoint(model, optimizer, cfg.checkpoint_dir, torch.device("cpu")) == 1
    trainer.configure_trainable_parameters(model, "router_first4")
    with pytest.raises(ValueError, match="Legacy checkpoints"):
        trainer.load_checkpoint(model, optimizer, cfg.checkpoint_dir, torch.device("cpu"))
    model.vision_router = None
    with pytest.raises(ValueError, match="VisionRouter enabled"):
        trainer.configure_trainable_parameters(model, "router_first4")


def test_sparse_hard_supervision_has_no_soft_auxiliary_gradient(small_model):
    model = small_model
    trainer.configure_trainable_parameters(model, 'router_first4')
    images = {key: torch.rand(2,3,224,224)*2-1 for key in model_lib.IMAGE_KEYS}
    targets = {key: torch.zeros(2,224,224) for key in images}
    targets['base_0_rgb'][0,40:100,80:150] = 1
    observation = model_lib.Observation(
        images=images, image_masks={k:torch.tensor([i<2,i<2]) for i,k in enumerate(images)},
        state=torch.zeros(2,8), tokenized_prompt=torch.arange(8)[None].expand(2,-1),
        tokenized_prompt_mask=torch.ones(2,8,dtype=torch.bool), router_label_masks=targets,
    )
    actions = torch.randn(2,4,8)
    losses = model(observation, actions, return_loss_components=True, router_label_weight=.01)
    assert losses['router_labeled_count'] == 1
    torch.testing.assert_close(losses['loss'], losses['action_loss']+.01*losses['router_label_loss'])
    losses['router_label_loss'].backward()
    assert sum(p.grad.abs().sum() for p in model.vision_router.selection_score.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.vision_router.compression_assignment.parameters())
    model.zero_grad(set_to_none=True)
    losses = model(observation, actions, return_loss_components=True, router_label_weight=.01)
    losses['loss'].backward()
    assert sum(p.grad.abs().sum() for p in model.vision_router.compression_assignment.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_only_npu_skips_redundant_outer_checkpoint(small_model, monkeypatch):
    model = small_model
    trainer.configure_trainable_parameters(model, "router_first4")
    images = {key: torch.rand(1, 3, 224, 224) * 2 - 1 for key in model_lib.IMAGE_KEYS}
    labels = {key: torch.zeros(1, 224, 224) for key in images}
    labels["base_0_rgb"][:, 40:100, 80:150] = 1
    observation = model_lib.Observation(
        images=images, image_masks={key: torch.tensor([i < 2]) for i, key in enumerate(images)},
        state=torch.zeros(1, 8), tokenized_prompt=torch.arange(8)[None],
        tokenized_prompt_mask=torch.ones(1, 8, dtype=torch.bool), router_label_masks=labels,
    )
    actions = torch.randn(1, 4, 8)
    noise = torch.randn_like(actions)
    calls = []
    original = model._apply_checkpoint

    def checkpoint(function, *args, **kwargs):
        calls.append(function.__name__)
        return original(function, *args, **kwargs)

    monkeypatch.setattr(model, "_apply_checkpoint", checkpoint)
    reference = None
    for backend in ("cpu", "cuda", "npu"):
        # Exercise backend dispatch with real CPU operators; actual NPU equivalence is checked separately.
        monkeypatch.setattr(pi0_pytorch.torch_runtime, "device_type", lambda: backend)
        calls.clear()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(17)
        output = model(observation, actions, noise=noise, time=torch.tensor([.4]),
                       return_loss_components=True, router_label_weight=.01)
        output["loss"].backward()
        assert ("forward_func" in calls) == (backend != "npu")
        assert {"image_embed_func", "lang_embed_func", "action_out_proj_func"} <= set(calls)
        actual = {name: p.grad.clone() for name, p in model.named_parameters() if p.requires_grad}
        actual.update({"loss/" + key: value.detach().clone() for key, value in output.items()})
        if reference is None:
            reference = actual
        else:
            for name in actual:
                torch.testing.assert_close(actual[name], reference[name], rtol=0, atol=0)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
