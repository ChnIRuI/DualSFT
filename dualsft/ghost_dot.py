from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .masking import ParameterSlice


@dataclass
class _LinearProjectionSpec:
    u_weight_cpu: Optional[torch.Tensor]
    u_bias_cpu: Optional[torch.Tensor]
    u_weight_cache: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)
    u_bias_cache: Dict[Tuple[str, str], torch.Tensor] = field(default_factory=dict)


class GhostDotLinearScorer:
    """Streaming estimator for phi_data_n = <u, g_n> on affine layers.

    This follows the paper's Ghost Dot Product idea for linear layers:
    e_n^T U a_n (plus bias term), accumulated during one backward pass.
    """

    def __init__(self, model: nn.Module, layout: List[ParameterSlice], direction_u: torch.Tensor):
        self.model = model
        self._active = False
        self._batch_scores: Optional[torch.Tensor] = None
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        name_to_slice: Dict[str, ParameterSlice] = {item.name: item for item in layout}
        self.total_dim = int(direction_u.numel())
        covered = torch.zeros(self.total_dim, dtype=torch.bool)

        self._spec_by_module: Dict[nn.Module, _LinearProjectionSpec] = {}

        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            weight_name = f"{module_name}.weight" if module_name else "weight"
            bias_name = f"{module_name}.bias" if module_name else "bias"

            u_weight = None
            u_bias = None

            weight_slice = name_to_slice.get(weight_name)
            if weight_slice is not None:
                u_weight = direction_u[weight_slice.start : weight_slice.end].reshape(weight_slice.shape).detach().cpu()
                covered[weight_slice.start : weight_slice.end] = True

            bias_slice = name_to_slice.get(bias_name)
            if bias_slice is not None:
                u_bias = direction_u[bias_slice.start : bias_slice.end].reshape(bias_slice.shape).detach().cpu()
                covered[bias_slice.start : bias_slice.end] = True

            if u_weight is None and u_bias is None:
                continue

            self._spec_by_module[module] = _LinearProjectionSpec(
                u_weight_cpu=u_weight,
                u_bias_cpu=u_bias,
            )
            self._handles.append(module.register_forward_hook(self._forward_hook))
            self._handles.append(module.register_full_backward_hook(self._backward_hook))

        self.covered_ratio = covered.float().mean().item() if covered.numel() > 0 else 0.0

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for spec in self._spec_by_module.values():
            spec.u_weight_cache.clear()
            spec.u_bias_cache.clear()
        self._spec_by_module.clear()

    def start_batch(self, batch_size: int, device: torch.device) -> None:
        self._batch_scores = torch.zeros(batch_size, dtype=torch.float32, device=device)
        self._active = True

    def consume_batch_scores(self) -> torch.Tensor:
        if self._batch_scores is None:
            raise RuntimeError("No batch scores available")
        out = self._batch_scores.detach()
        self._batch_scores = None
        self._active = False
        return out

    def _forward_hook(self, module: nn.Module, args, output) -> None:
        if not self._active:
            return
        if len(args) == 0:
            return
        setattr(module, "_ghost_input", args[0].detach())

    def _backward_hook(self, module: nn.Module, grad_input, grad_output) -> None:
        if not self._active:
            return
        if self._batch_scores is None:
            return
        if len(grad_output) == 0 or grad_output[0] is None:
            return

        spec = self._spec_by_module.get(module)
        if spec is None:
            return

        inp = getattr(module, "_ghost_input", None)
        if inp is None:
            return

        gout = grad_output[0].detach()
        batch_size = inp.shape[0]
        contrib = torch.zeros(batch_size, dtype=torch.float32, device=gout.device)

        if spec.u_weight_cpu is not None:
            uw = self._cached_to_device(
                cpu_tensor=spec.u_weight_cpu,
                cache=spec.u_weight_cache,
                device=gout.device,
                dtype=inp.dtype,
            )
            proj = torch.matmul(inp, uw.transpose(0, 1))
            contrib = contrib + (gout * proj).reshape(batch_size, -1).sum(dim=-1).float()

        if spec.u_bias_cpu is not None:
            ub = self._cached_to_device(
                cpu_tensor=spec.u_bias_cpu,
                cache=spec.u_bias_cache,
                device=gout.device,
                dtype=gout.dtype,
            )
            contrib = contrib + (gout * ub).reshape(batch_size, -1).sum(dim=-1).float()

        self._batch_scores.add_(contrib)

        if hasattr(module, "_ghost_input"):
            delattr(module, "_ghost_input")

    @staticmethod
    def _cached_to_device(
        cpu_tensor: torch.Tensor,
        cache: Dict[Tuple[str, str], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (str(device), str(dtype))
        cached = cache.get(key)
        if cached is None:
            cached = cpu_tensor.to(device=device, dtype=dtype)
            cache[key] = cached
        return cached
