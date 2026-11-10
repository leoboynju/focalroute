"""Device selection shared by CUDA training and optional Ascend deployment."""

from functools import lru_cache
import importlib.util
import os

import torch


@lru_cache(maxsize=1)
def device_type() -> str:
    requested = os.environ.get("OPENPI_DEVICE", "auto")
    if requested not in ("auto", "cpu", "cuda", "npu"):
        raise ValueError(f"Unsupported OPENPI_DEVICE={requested}")
    if requested == "cpu":
        return "cpu"
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    if requested in ("auto", "npu") and importlib.util.find_spec("torch_npu") is not None:
        import torch_npu  # noqa: F401

        if torch.npu.is_available():
            torch.npu.set_compile_mode(jit_compile=False)
            return "npu"
    if requested != "auto":
        raise RuntimeError(f"Requested {requested} accelerator is unavailable")
    return "cpu"


def accelerator():
    # CUDA's unavailable-device checks and empty_cache are safe for the CPU fallback.
    return torch.npu if device_type() == "npu" else torch.cuda


def get_device(index: int = 0) -> torch.device:
    kind = device_type()
    return torch.device("cpu" if kind == "cpu" else f"{kind}:{index}")


def npu_attention(query, key, value, attention_mask, scale):
    """Native GQA attention; OpenPI uses zero / negative padding-block masks."""
    import torch_npu

    blocked = attention_mask < 0 if attention_mask is not None else None
    result = torch_npu.npu_fusion_attention(
        query, key, value, head_num=query.shape[1], input_layout="BNSD",
        atten_mask=blocked, scale=scale, keep_prob=1.0,
    )[0]
    return result.transpose(1, 2).contiguous()
