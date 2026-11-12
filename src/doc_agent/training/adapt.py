"""Stage 8 — affordable adaptation — LoRA / quantization"""
from __future__ import annotations

import torch
from torch import nn

from ..contracts import *  # noqa


def apply_lora(model, cfg: dict):
    """Wrap a transformer component (e.g. the TrOCR decoder or the embedder) with LoRA adapters per
    cfg['adapt']['lora']: only the low-rank matrices on target_modules are trainable."""
    from peft import LoraConfig, get_peft_model

    c = cfg["adapt"]["lora"]
    lc = LoraConfig(r=c["r"], lora_alpha=c["alpha"], lora_dropout=c["dropout"],
                    target_modules=c["target_modules"])
    return get_peft_model(model, lc)
def quantize(model, cfg: dict):
    """Post-training dynamic quantization of Linear layers (CPU inference), dtype from
    cfg['adapt']['quant']['dtype'] (qint8 | float16)."""
    dtype = getattr(torch, cfg["adapt"]["quant"]["dtype"])
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=dtype)
