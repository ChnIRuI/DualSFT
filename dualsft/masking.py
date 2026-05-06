from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn as nn


@dataclass
class ParameterSlice:
    name: str
    param: nn.Parameter
    start: int
    end: int
    shape: torch.Size

    @property
    def numel(self) -> int:
        return self.end - self.start


def get_trainable_layout(model: nn.Module) -> List[ParameterSlice]:
    layout: List[ParameterSlice] = []
    cursor = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        layout.append(
            ParameterSlice(
                name=name,
                param=p,
                start=cursor,
                end=cursor + n,
                shape=p.shape,
            )
        )
        cursor += n
    return layout


def total_trainable_dim(layout: Iterable[ParameterSlice]) -> int:
    items = list(layout)
    if not items:
        return 0
    return items[-1].end


def zeros_vector(layout: List[ParameterSlice], device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.zeros(total_trainable_dim(layout), device=device, dtype=dtype)


def grads_to_vector(layout: List[ParameterSlice], grads: List[torch.Tensor | None], device: torch.device) -> torch.Tensor:
    vec = zeros_vector(layout, device=device)
    for item, g in zip(layout, grads):
        if g is None:
            continue
        vec[item.start : item.end] = g.detach().reshape(-1).to(device=device, dtype=torch.float32)
    return vec


def topk_mask(scores: torch.Tensor, k: int, use_abs: bool = False) -> torch.Tensor:
    k = max(1, min(int(k), scores.numel()))
    work = scores.abs() if use_abs else scores
    indices = torch.topk(work, k=k, largest=True).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask[indices] = True
    return mask


def vector_mask_to_named_tensors(mask_vector: torch.Tensor, layout: List[ParameterSlice], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for item in layout:
        m = mask_vector[item.start : item.end].reshape(item.shape)
        out[item.name] = m.to(device=device)
    return out


def apply_grad_mask(model: nn.Module, named_mask: Dict[str, torch.Tensor]) -> None:
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        m = named_mask.get(name)
        if m is None:
            continue
        p.grad.mul_(m)


def extract_adamw_second_moment(
    layout: List[ParameterSlice],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    c = zeros_vector(layout, device=device, dtype=dtype)
    for item in layout:
        state = optimizer.state.get(item.param, {})
        exp_avg_sq = state.get("exp_avg_sq", None)
        if exp_avg_sq is None:
            continue
        c[item.start : item.end] = exp_avg_sq.detach().reshape(-1).to(device=device, dtype=dtype)
    return c


def split_named_vector(vec: torch.Tensor, layout: List[ParameterSlice]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for item in layout:
        out[item.name] = vec[item.start : item.end].reshape(item.shape)
    return out
