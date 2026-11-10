import json

import numpy as np
import pytest

from openpi.training import token_budget


def test_budget_validation_rejects_invalid_inputs():
    valid_mask = np.ones((2, 8), dtype=bool)
    scores = np.ones((2, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="NaN"):
        bad = scores.copy()
        bad[0, 0] = np.nan
        token_budget.compute_token_budget_summary(bad, valid_mask)

    with pytest.raises(ValueError, match="negative"):
        bad = scores.copy()
        bad[0, 1] = -0.1
        token_budget.compute_token_budget_summary(bad, valid_mask)

    with pytest.raises(ValueError, match="positive valid attention mass"):
        bad = np.zeros((2, 8), dtype=np.float32)
        token_budget.compute_token_budget_summary(bad, valid_mask)

    with pytest.raises(ValueError, match="coverage_target"):
        token_budget.compute_token_budget_summary(scores, valid_mask, coverage_target=0.0)

    with pytest.raises(ValueError, match="alignment"):
        token_budget.compute_token_budget_summary(scores, valid_mask, alignment=0)


def test_higher_quantile_alignment_after_aligned_hard_and_padding_exclusion():
    # 20 samples, 768 total tokens (512 valid, 256 padding on 3rd camera).
    num_samples = 20
    total_tokens = 768
    valid_tokens = 512
    valid_mask = np.zeros((num_samples, total_tokens), dtype=bool)
    valid_mask[:, :valid_tokens] = True

    # Construct synthetic relevance where K_f(0.8) = 193 and K_f(0.9) = 301 on the 95th percentile sample.
    relevance = np.zeros((num_samples, total_tokens), dtype=np.float64)
    for i in range(num_samples):
        k80 = 180 if i < 18 else 193
        k90 = 280 if i < 18 else 301
        relevance[i, :k80] = 0.80 / k80
        relevance[i, k80:k90] = 0.10 / (k90 - k80)
        relevance[i, k90:valid_tokens] = 0.10 / (valid_tokens - k90)
        # Put large junk scores on invalid 3rd camera positions to ensure they are excluded.
        relevance[i, valid_tokens:] = 999.0

    summary, hard_per_sample = token_budget.compute_token_budget_summary(
        relevance,
        valid_mask,
        coverage_target=0.80,
        sample_quantile=0.95,
        alignment=32,
        global_coverage_upper=0.90,
        global_min_tokens=64,
    )
    assert summary["raw_hard_quantile_tokens"] == 193
    assert summary["recommended_hard_tokens"] == 224
    assert summary["raw_upper_quantile_tokens"] == 301
    assert summary["raw_global_quantile_tokens"] == 77
    assert summary["recommended_global_tokens"] == 96
    assert summary["recommended_total_tokens"] == 320
    assert summary["global_token_method"] == "after_aligned_hard"
    assert summary["global_subtracted_hard_tokens"] == 224
    assert hard_per_sample[-1] == 193

    # Coverage 1.0 must never include the 256 masked padding tokens.
    k100 = token_budget.compute_tokens_for_coverage(relevance, valid_mask, 1.0)
    assert np.all(k100 == valid_tokens)

    # If aligned H exceeds max_valid_visual_tokens, raise ValueError.
    with pytest.raises(ValueError, match="exceeds maximum valid visual tokens"):
        token_budget.compute_token_budget_summary(
            relevance,
            valid_mask,
            coverage_target=1.0,
            alignment=300,
        )


def test_npz_recalculation_and_json_only_guards(tmp_path):
    num_samples = 20
    total_tokens = 768
    valid_tokens = 512
    valid_mask = np.zeros((num_samples, total_tokens), dtype=bool)
    valid_mask[:, :valid_tokens] = True

    relevance = np.zeros((num_samples, total_tokens), dtype=np.float64)
    for i in range(num_samples):
        # 0.70 reached at 122, 0.80 reached at 193, 0.85 reached at 241, 0.90 reached at 301.
        relevance[i, :122] = 0.701 / 122
        relevance[i, 122:193] = 0.100 / (193 - 122)
        relevance[i, 193:241] = 0.050 / (241 - 193)
        relevance[i, 241:301] = 0.050 / (301 - 241)
        relevance[i, 301:valid_tokens] = 0.099 / (valid_tokens - 301)

    summary, hard_per_sample = token_budget.compute_token_budget_summary(
        relevance,
        valid_mask,
        coverage_target=0.80,
        sample_quantile=0.95,
        alignment=32,
        global_coverage_upper=0.90,
        global_min_tokens=64,
    )
    report = token_budget.build_budget_report(
        config_name="pi05_libero",
        repo_id="physical-intelligence/libero",
        checkpoint="/tmp/teacher",
        attention_source="vit",
        model_config_dict={"format_version": 1},
        summary=summary,
        layers=(23, 24, 25, 26),
    )
    json_path, npz_path = token_budget.save_budget_artifacts(
        tmp_path / "token_budget_vit.json",
        report,
        sample_indices=np.arange(num_samples),
        relevance=relevance,
        valid_mask=valid_mask,
        hard_tokens_per_sample=hard_per_sample,
    )

    # Recompute 70% -> 85% from NPZ cache: Q95 K70=122 -> H=128, Q95 K85=241 -> residual 113 -> G=128.
    recomputed = token_budget.resolve_training_budget(
        json_path,
        expected_source="vit",
        expected_config_name="pi05_libero",
        expected_repo_id="physical-intelligence/libero",
        coverage_target=0.70,
        global_coverage_upper=0.85,
    )
    assert recomputed.hard_tokens == 128
    assert recomputed.soft_tokens == 128
    assert recomputed.summary["raw_hard_quantile_tokens"] == 122
    assert recomputed.summary["raw_upper_quantile_tokens"] == 241
    assert recomputed.summary["raw_global_quantile_tokens"] == 113

    # Source, config_name, and repo_id mismatches must fail.
    with pytest.raises(ValueError, match="attention_source"):
        token_budget.load_budget(tmp_path, expected_source="action")
    with pytest.raises(ValueError, match="config_name"):
        token_budget.load_budget(json_path, expected_config_name="pi0_aloha")
    with pytest.raises(ValueError, match="repo_id"):
        token_budget.load_budget(json_path, expected_repo_id="wrong/repo")

    # Without NPZ, alignment/min_tokens can be recalculated, but changing coverage_target fails.
    npz_path.unlink()
    json_only = token_budget.resolve_training_budget(
        json_path,
        expected_source="vit",
        alignment=64,
        global_min_tokens=128,
    )
    assert json_only.hard_tokens == 256
    assert json_only.soft_tokens == 128

    with pytest.raises(ValueError, match="without sibling NPZ cache"):
        token_budget.resolve_training_budget(json_path, coverage_target=0.70)


def test_archive_training_budget_preserves_original_json_and_guards_overwrite(tmp_path):
    valid_mask = np.ones((4, 64), dtype=bool)
    relevance = np.ones((4, 64), dtype=np.float32) / 64.0
    summary, hard_per_sample = token_budget.compute_token_budget_summary(
        relevance,
        valid_mask,
        coverage_target=0.50,
        sample_quantile=0.95,
        alignment=16,
        global_coverage_upper=0.75,
        global_min_tokens=16,
    )
    report = token_budget.build_budget_report(
        config_name="pi05_libero",
        repo_id="physical-intelligence/libero",
        checkpoint="/tmp/teacher",
        attention_source="vit",
        model_config_dict={"format_version": 1},
        summary=summary,
        layers=(23, 24, 25, 26),
    )
    src_dir = tmp_path / "src_cal"
    json_path, _ = token_budget.save_budget_artifacts(
        src_dir / "token_budget_vit.json",
        report,
        sample_indices=np.arange(4),
        relevance=relevance,
        valid_mask=valid_mask,
        hard_tokens_per_sample=hard_per_sample,
    )
    orig_bytes = json_path.read_bytes()

    resolved = token_budget.resolve_training_budget(
        json_path,
        coverage_target=0.25,
        global_coverage_upper=0.50,
        alignment=16,
        global_min_tokens=16,
    )
    ckpt_dir = tmp_path / "ckpt"
    run_out_dir = tmp_path / "run_out"
    archived_json = token_budget.archive_training_budget(resolved, ckpt_dir, run_output_dir=run_out_dir)

    assert archived_json.read_bytes() == orig_bytes
    tb = json.loads((ckpt_dir / "calibration" / "training_budget.json").read_text())
    assert tb["selection"]["coverage_target"] == 0.25
    assert (run_out_dir / "calibration" / "training_budget.json").exists()

    # Archiving conflicting selection into the same checkpoint directory must fail.
    conflicting = token_budget.resolve_training_budget(
        json_path,
        coverage_target=0.50,
        global_coverage_upper=0.75,
        alignment=16,
        global_min_tokens=16,
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        token_budget.archive_training_budget(conflicting, ckpt_dir)


def test_collect_teacher_attention_hook_cleanup_and_padding_query_exclusion(monkeypatch):
    from types import SimpleNamespace
    import torch
    from openpi.models import pi0_config
    from openpi.models_pytorch import pi0_pytorch
    import scripts.rerender_token_attention_test as rtest

    monkeypatch.setattr(
        pi0_pytorch._gemma,
        "get_config",
        lambda variant: SimpleNamespace(width=16, depth=4, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8),
    )
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", rtest._MockPaliGemmaWithExpert)

    cfg = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
        pytorch_compile_mode=None,
        vision_router_hard_enabled=False,
        vision_router_soft_enabled=False,
    )
    model = pi0_pytorch.PI0Pytorch(cfg)
    model.train()

    ds = rtest._FakeEpisodeDataset(num_frames=2, max_token_len=8, action_dim=8, action_horizon=4)
    batch = rtest.render_mod._collate_samples([ds[0], ds[1]])
    tensor_batch = {
        k: ({sk: torch.as_tensor(sv) for sk, sv in v.items()} if isinstance(v, dict) else torch.as_tensor(v))
        for k, v in batch.items()
    }
    obs = rtest.render_mod._model.Observation.from_dict(tensor_batch)

    collected = token_budget.collect_teacher_attention_batch(
        model,
        obs,
        sources=("vit", "vlm", "action"),
        num_steps=4,
        action_steps=(0, 2, 3),
    )
    # Restore training mode and sdpa config after collection.
    assert model.training is True
    assert model.paligemma_with_expert.paligemma.language_model.config._attn_implementation == "sdpa"
    for layer in model.paligemma_with_expert.vit_layers:
        assert len(layer.self_attn._forward_hooks) == 0
        assert len(layer.self_attn._forward_pre_hooks) == 0
    for s in ("vit", "vlm", "action"):
        # Masked 3rd camera (last 16 tokens) must have strictly 0 score.
        assert np.all(collected.raw_scores_by_source[s][:, 32:] == 0.0)
        assert np.allclose(collected.normalized_by_source[s][:, :32].sum(axis=1), 1.0)


def test_run_calibration_end_to_end_and_rejects_router_checkpoint(tmp_path, monkeypatch):
    import dataclasses
    from types import SimpleNamespace
    import safetensors.torch
    from openpi.models import pi0_config
    from openpi.models_pytorch import pi0_pytorch
    import scripts.calibrate_token_budget as cal_mod
    import scripts.rerender_token_attention_test as rtest

    monkeypatch.setattr(
        pi0_pytorch._gemma,
        "get_config",
        lambda variant: SimpleNamespace(width=16, depth=4, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8),
    )
    monkeypatch.setattr(pi0_pytorch, "PaliGemmaWithExpertModel", rtest._MockPaliGemmaWithExpert)

    base_train_cfg = cal_mod._config.get_config("pi05_libero")
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
    monkeypatch.setattr(cal_mod._config, "get_config", lambda name: fake_train_cfg)

    teacher_dir = tmp_path / "teacher"
    teacher_dir.mkdir()
    teacher_model = pi0_pytorch.PI0Pytorch(fake_train_cfg.model)
    safetensors.torch.save_model(teacher_model, str(teacher_dir / "model.safetensors"))
    pi0_pytorch.save_model_config(fake_train_cfg.model, teacher_dir)

    dataset = rtest._FakeEpisodeDataset(num_frames=4, max_token_len=8, action_dim=8, action_horizon=4)
    out_dir = tmp_path / "cal_out"
    saved_json, saved_npz, report = cal_mod.run_calibration(
        checkpoint_dir=teacher_dir,
        output_path=out_dir,
        config_name="pi05_libero",
        attention_source="vit",
        num_samples=4,
        batch_size=2,
        alignment=8,
        global_min_tokens=8,
        dataset=dataset,
    )
    assert saved_json.exists() and saved_npz.exists()
    assert report["attention_source"] == "vit"
    assert report["method"] == "full_vit_patch_attention"

    # Refuses to overwrite existing calibration output.
    with pytest.raises(FileExistsError):
        cal_mod.run_calibration(
            checkpoint_dir=teacher_dir,
            output_path=out_dir,
            attention_source="vit",
            num_samples=4,
            dataset=dataset,
        )

    # Refuses router-containing checkpoint.
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
        cal_mod.run_calibration(
            checkpoint_dir=routed_dir,
            output_path=tmp_path / "cal_out2",
            dataset=dataset,
        )
