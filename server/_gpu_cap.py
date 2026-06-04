"""Hard 8 GB VRAM cap across all GPU engines.

Test box has 16 GB (T4) but we cap to 8 GB so numbers defensibly transfer
to consumer-tier hardware (RTX 2060 6 GB, RTX 2070 8 GB, mobile 2070 Max-Q).
Anything that runs past 8 GB during a benchmark is by definition not
deployable on the laptop tier Bob's customers buy.

Two knobs because we have two GPU runtimes in the mix:
  - PyTorch (NeMo, pyannote, sortformer, parakeet-nemo): set_per_process_memory_fraction
  - ONNX Runtime CUDA EP (sherpa-onnx → nemotron-streaming): gpu_mem_limit option

Both pin to the same VRAM_LIMIT_BYTES so the floor is uniform.
"""
from __future__ import annotations

import os

VRAM_LIMIT_BYTES = 8 * 1024 * 1024 * 1024  # 8 GiB


def apply() -> dict:
    """Apply the PyTorch cap (idempotent). Returns a dict for logging."""
    info: dict = {"limit_gib": 8, "applied": []}
    try:
        import torch
    except ImportError:
        info["torch"] = "not installed — nothing to cap"
        return info

    if not torch.cuda.is_available():
        info["torch"] = "CUDA not available — nothing to cap"
        return info

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, VRAM_LIMIT_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    info["torch"] = f"capped to {fraction:.2%} of {total / 2**30:.1f} GiB"
    info["applied"].append("torch.cuda.set_per_process_memory_fraction")
    return info


def sherpa_cuda_provider_options() -> list[dict]:
    """Pass to sherpa_onnx OnlineRecognizer if provider='cuda'."""
    return [
        {
            "device_id": 0,
            "gpu_mem_limit": VRAM_LIMIT_BYTES,
            "arena_extend_strategy": "kSameAsRequested",
        }
    ]
