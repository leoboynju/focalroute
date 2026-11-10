"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
from openpi.shared import torch_runtime
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training import token_budget


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch_runtime.get_device(local_rank)
    if device.type != "cpu":
        torch_runtime.accelerator().set_device(device)
    if use_ddp and not torch.distributed.is_initialized():
        backend = {"cuda": "nccl", "npu": "hccl", "cpu": "gloo"}[device.type]
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch_runtime.device_type() != "cpu":
        torch_runtime.accelerator().manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get only trainable parameters, handling DDP wrapper."""
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def configure_trainable_parameters(model, mode: str):
    """Set the training scope before constructing DDP and the optimizer."""
    if mode == "full":
        model.requires_grad_(True)
    elif mode == "router_llm_expert":
        if not model.pi05 or model.vision_router is None:
            raise ValueError("router_llm_expert requires a pi05 model with VisionRouter enabled.")
        model.requires_grad_(True)
        model.paligemma_with_expert.paligemma.vision_tower.requires_grad_(False)
        model.paligemma_with_expert.paligemma.multi_modal_projector.requires_grad_(False)
    elif mode == "router_first4":
        if not model.pi05 or model.vision_router is None:
            raise ValueError("router_first4 requires a pi05 model with VisionRouter enabled.")
        llm_layers = model.paligemma_with_expert.paligemma.language_model.layers
        expert_layers = model.paligemma_with_expert.gemma_expert.model.layers
        if min(len(llm_layers), len(expert_layers)) < 4:
            raise ValueError("router_first4 requires at least four LLM and four expert blocks.")
        model.requires_grad_(False)
        model.vision_router.requires_grad_(True)
        llm_layers[:4].requires_grad_(True)
        expert_layers[:4].requires_grad_(True)
    else:
        raise ValueError(f"Unsupported pytorch_train_mode: {mode}")

    # Frozen downstream blocks must still backpropagate to Router and the first four blocks.
    trainable = list(get_model_parameters(model))
    logging.info(
        "PyTorch train mode=%s: trainable=%d / total=%d parameters; gradient payload=%.3f GB",
        mode,
        sum(p.numel() for p in trainable),
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() * p.element_size() for p in trainable) / 1e9,
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, model_config.json, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state and model_config.json using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")
        openpi.models_pytorch.pi0_pytorch.save_model_config(model_to_save.config, tmp_ckpt_dir)

        # Save optimizer state using PyTorch format
        optimizer_state = optimizer.state_dict()
        if torch_runtime.device_type() == "npu":
            # CPU tensors keep checkpoints loadable on CUDA hosts without torch_npu.
            optimizer_state = jax.tree.map(lambda value: value.cpu() if isinstance(value, torch.Tensor) else value, optimizer_state)
        torch.save(optimizer_state, tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "trainable_parameter_names": [name for name, p in model_to_save.named_parameters() if p.requires_grad],
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    metadata = torch.load(ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False)
    model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    expected_names = metadata.get("trainable_parameter_names")
    actual_names = [name for name, p in model_to_load.named_parameters() if p.requires_grad]
    if expected_names is not None and expected_names != actual_names:
        raise ValueError("Checkpoint trainable parameter names/order do not match the current optimizer scope.")
    if expected_names is None and any(not p.requires_grad for p in model_to_load.parameters()):
        raise ValueError("Legacy checkpoints without a training scope can only resume full fine-tuning.")

    # Clear memory before loading checkpoints
    if torch_runtime.device_type() != "cpu":
        torch_runtime.accelerator().empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            safetensors.torch.load_model(
                model_to_load, safetensors_path, device="cpu" if device.type == "npu" else str(device)
            )
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch_runtime.accelerator().empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch_runtime.accelerator().empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch_runtime.accelerator().empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch_runtime.accelerator().empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if device.type == "cpu":
        return

    accelerator = torch_runtime.accelerator()
    memory_allocated = accelerator.memory_allocated(device) / 1e9
    memory_reserved = accelerator.memory_reserved(device) / 1e9
    memory_free = accelerator.memory_reserved(device) - accelerator.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = accelerator.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): {device.type} memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def _normalize_model_config(config: _config.TrainConfig) -> openpi.models.pi0_config.Pi0Config:
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        return openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    return dataclasses.replace(config.model, dtype=config.pytorch_training_precision)


def prepare_training_config_and_checkpoint_dir(
    config: _config.TrainConfig,
    *,
    is_main: bool,
    use_ddp: bool,
) -> tuple[_config.TrainConfig, openpi.models.pi0_config.Pi0Config, bool]:
    """Manage checkpoint directory on rank 0, apply token budget or resume topology, and archive budget files."""
    exp_checkpoint_dir = config.checkpoint_dir
    resuming = False
    latest_step: int | None = None

    if config.resume:
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    else:
        if is_main:
            if config.overwrite and exp_checkpoint_dir.exists():
                shutil.rmtree(exp_checkpoint_dir)
                logging.info(f"Overwriting checkpoint directory: {exp_checkpoint_dir}")
            exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
        if use_ddp and dist.is_initialized():
            dist.barrier()

    model_cfg = _normalize_model_config(config)
    train_mode = config.pytorch_train_mode or "full"

    if resuming:
        assert latest_step is not None
        latest_ckpt_dir = exp_checkpoint_dir / f"{latest_step}"
        requested_camera_mask = model_cfg.vision_router_mask_invalid_cameras
        requested_original_rope = model_cfg.vision_router_preserve_hard_rope
        model_cfg = openpi.models_pytorch.pi0_pytorch.resolve_checkpoint_model_config(model_cfg, latest_ckpt_dir)
        if requested_camera_mask and not model_cfg.vision_router_mask_invalid_cameras:
            raise ValueError("Cannot enable camera masking on resume; start a new experiment.")
        if requested_original_rope and not model_cfg.vision_router_preserve_hard_rope:
            raise ValueError("Cannot enable original Hard RoPE on resume; start a new experiment.")
        metadata = torch.load(latest_ckpt_dir / "metadata.pt", map_location="cpu", weights_only=False)
        saved_config = metadata.get("config", {})
        train_mode = saved_config.get("pytorch_train_mode") or "full"
        saved_labels = saved_config.get("vision_router_label_manifest")
        saved_label_end = saved_config.get("vision_router_label_end_step")
        if config.vision_router_label_end_step is not None and config.vision_router_label_end_step != saved_label_end:
            raise ValueError("Cannot change the sparse label cutoff on resume; start a new experiment.")
        config = dataclasses.replace(config, vision_router_label_end_step=saved_label_end)
        if config.vision_router_label_manifest is not None and config.vision_router_label_manifest != saved_labels:
            raise ValueError("Cannot change sparse label source on resume; start a new experiment.")
        if saved_labels is not None:
            config = dataclasses.replace(
                config, vision_router_label_manifest=saved_labels,
                vision_router_label_variant=metadata["config"]["vision_router_label_variant"],
                vision_router_label_weight=metadata["config"]["vision_router_label_weight"],
            )
        if config.pytorch_train_mode is not None and config.pytorch_train_mode != train_mode:
            raise ValueError(
                f"Cannot resume {train_mode} checkpoint with pytorch_train_mode={config.pytorch_train_mode}. "
                "Start a new experiment with pytorch_weight_path to change the training scope."
            )
        config = dataclasses.replace(config, model=model_cfg)
    elif config.vision_router_budget_path is not None:
        if not model_cfg.pi05 or not (model_cfg.vision_router_hard_enabled and model_cfg.vision_router_soft_enabled):
            raise ValueError("Token budget calibration only supports new pi05 dual-branch VisionRouter training.")
        data_config = config.data.create(config.assets_dirs, model_cfg)
        resolved_budget = token_budget.resolve_training_budget(
            config.vision_router_budget_path,
            expected_source=config.vision_router_calibration_source,
            expected_config_name=config.name,
            expected_repo_id=data_config.repo_id,
            coverage_target=config.vision_router_attention_coverage,
            sample_quantile=config.vision_router_budget_quantile,
            alignment=config.vision_router_token_alignment,
            global_coverage_upper=config.vision_router_global_coverage_upper,
            global_min_tokens=config.vision_router_global_min_tokens,
        )
        model_cfg = dataclasses.replace(
            model_cfg,
            vision_router_hard_tokens=resolved_budget.hard_tokens,
            vision_router_soft_tokens=resolved_budget.soft_tokens,
        )
        if (
            model_cfg.vision_router_hard_tokens != resolved_budget.selection["hard_tokens"]
            or model_cfg.vision_router_soft_tokens != resolved_budget.selection["soft_tokens"]
        ):
            raise RuntimeError("Resolved training budget selection does not match model VisionRouter token counts")

        archived_json_path = exp_checkpoint_dir / "calibration" / resolved_budget.source_json_path.name
        if is_main:
            archived_json_path = token_budget.archive_training_budget(
                resolved_budget,
                exp_checkpoint_dir,
                run_output_dir=os.environ.get("RUN_OUTPUT_DIR"),
            )
        if use_ddp and dist.is_initialized():
            dist.barrier()

        config = dataclasses.replace(
            config,
            model=model_cfg,
            vision_router_budget_path=str(archived_json_path),
        )
        logging.info(
            "Applied VisionRouter token budget (%s): H=%d, G=%d, R=%d (archived at %s)",
            resolved_budget.selection["attention_source"],
            model_cfg.vision_router_hard_tokens,
            model_cfg.vision_router_soft_tokens,
            model_cfg.vision_router_dim,
            archived_json_path,
        )
    else:
        config = dataclasses.replace(config, model=model_cfg)

    config = dataclasses.replace(config, pytorch_train_mode=train_mode)
    return config, model_cfg, resuming


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    config, model_cfg, resuming = prepare_training_config_and_checkpoint_dir(
        config,
        is_main=is_main,
        use_ddp=use_ddp,
    )

    # Initialize wandb (only on main process)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)

    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = observation.to_dict()
        sample_batch["actions"] = actions

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if device.type != "cpu":
            torch_runtime.accelerator().empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    configure_trainable_parameters(model, config.pytorch_train_mode)
    trainable_parameters = list(get_model_parameters(model))
    if getattr(model, "vision_router", None) is not None:
        if (
            model.vision_router.hard_tokens != (model_cfg.vision_router_hard_tokens if model_cfg.vision_router_hard_enabled else 0)
            or model.vision_router.soft_tokens != (model_cfg.vision_router_soft_tokens if model_cfg.vision_router_soft_enabled else 0)
        ):
            raise RuntimeError("Constructed VisionRouter token counts do not match resolved model_cfg")

    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and device.type != "cpu":
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8 and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type != "cpu" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from pretrained checkpoint if starting a new fine-tuning run
    if not resuming and config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        openpi.models_pytorch.pi0_pytorch.load_pretrained_weights(
            model,
            config.pytorch_weight_path,
            device=device,
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        trainable_parameters,
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if resuming
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            # The unified data loader returns (observation, actions) tuple
            observation = jax.tree.map(lambda x: x.to(device, non_blocking=True), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            actions = actions.to(device, non_blocking=True)  # noqa: PLW2901

            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            label_weight = (
                config.router_label_weight_for_step(global_step)
                if config.vision_router_label_manifest is not None else 0.0
            )
            label_kwargs = (
                {"router_label_weight": label_weight}
                if config.vision_router_label_manifest is not None else {}
            )
            loss_outputs = model(observation, actions, return_loss_components=True, **label_kwargs)
            if isinstance(loss_outputs, dict):
                loss = loss_outputs["loss"]
                action_loss = loss_outputs.get("action_loss", loss)
                router_camera_loss = loss_outputs.get(
                    "router_camera_loss", torch.zeros((), dtype=loss.dtype, device=loss.device)
                )
            else:
                if isinstance(loss_outputs, list | tuple):
                    loss_outputs = torch.stack(loss_outputs)
                elif not isinstance(loss_outputs, torch.Tensor):
                    loss_outputs = torch.tensor(loss_outputs, device=device, dtype=torch.float32)
                loss = loss_outputs.mean()
                action_loss = loss
                router_camera_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and device.type != "cpu":
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=config.optimizer.clip_gradient_norm, error_if_nonfinite=True
            )

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # Collect stats
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "action_loss": action_loss.item(),
                        "router_camera_loss": router_camera_loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                if isinstance(loss_outputs, dict) and config.vision_router_label_manifest is not None:
                    logging.info(
                        "router_supervision step=%d loss=%.6f global_labeled_frames=%.0f weight=%.4f",
                        global_step, float(loss_outputs.get("router_label_loss", 0.0)),
                        float(loss_outputs.get("router_labeled_count", 0.0)), label_weight,
                    )

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_action_loss = sum(info["action_loss"] for info in infos) / len(infos)
                avg_camera_loss = sum(info["router_camera_loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} action_loss={avg_action_loss:.4f} camera_loss={avg_camera_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} action_loss={avg_action_loss:.4f} camera_loss={avg_camera_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )

                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "action_loss": avg_action_loss,
                        "router_camera_loss": avg_camera_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
