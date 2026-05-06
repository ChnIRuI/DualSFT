from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DualSFTConfig:
    model_name_or_path: str
    train_file: str
    validation_file: str
    output_dir: str
    stage: str = "all"  # all|warmup|select|finetune

    text_field: str = "text"
    prompt_field: str = "prompt"
    response_field: str = "response"
    prompt_response_separator: str = "\n"
    train_on_prompt: bool = False
    max_length: int = 1024

    seed: int = 42
    device: str = "cuda"
    teacher_device: str = "cpu"
    num_workers: int = 4
    prefetch_factor: int = 2

    learning_rate: float = 2e-5
    final_learning_rate: Optional[float] = None
    warmup_epochs: int = 1
    final_epochs: int = 3
    warmup_batch_size: int = 2
    score_batch_size: int = 2
    final_batch_size: int = 4
    eval_batch_size: int = 2
    weight_decay: float = 0.0

    warmup_ratio: float = 0.05
    anchor_ratio: float = 0.05
    score_pool_ratio: float = 1.0

    data_budget_ratio: float = 0.10
    param_budget_ratio: float = 0.05
    data_budget: Optional[int] = None
    param_budget: Optional[int] = None

    lambda_new: float = 1.0
    lambda_prior: float = 1.0
    tau: float = 2.0
    confidence_mode: str = "max_prob"
    use_diagonal_second_order: bool = True
    gradient_normalize: str = "mean"

    score_method: str = "ghost_linear"
    score_dtype: str = "float16"
    ghost_fallback_exact: bool = False
    data_rerank_topm: int = 0
    topk_use_abs: bool = False
    save_param_scores: bool = False
    save_selected_data: bool = True
    save_data_scores: bool = False
    save_full_intermediates: bool = False
    bf16: bool = False
    fp16: bool = False
    quota_enable: bool = False
    quota_layer_min_ratio: float = 0.0
    quota_module_min_ratio: float = 0.0
    selection_dir: str | None = None
    final_model_tag: Optional[str] = None

    # gradient accumulation
    warmup_grad_accum_steps: int = 1
    final_grad_accum_steps: int = 1

    def ensure_paths(self) -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
