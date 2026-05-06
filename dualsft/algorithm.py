from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DualSFTConfig
from .data import (
    IndexedTextDataset,
    load_records,
    load_text_dataset,
    make_dataloader,
    save_records_jsonl,
    set_seed,
)
from .ghost_dot import GhostDotLinearScorer
from .masking import (
    ParameterSlice,
    extract_adamw_second_moment,
    get_trainable_layout,
    topk_mask,
    vector_mask_to_named_tensors,
)


def _first_existing(base: Path, names: list[str]) -> Path | None:
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None


class DualSFTRunner:
    def __init__(self, cfg: DualSFTConfig):
        self.cfg = cfg
        self.cfg.ensure_paths()

        if cfg.device == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(cfg.device)

        if cfg.teacher_device == "cuda" and not torch.cuda.is_available():
            self.teacher_device = torch.device("cpu")
        else:
            self.teacher_device = torch.device(cfg.teacher_device)

        if cfg.bf16 and cfg.fp16:
            raise ValueError("`--bf16` and `--fp16` are mutually exclusive.")
        if cfg.bf16:
            self.dtype = torch.bfloat16
        elif cfg.fp16:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.score_dtype = self._resolve_score_dtype(cfg.score_dtype)
        self.tokenizer = None

        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    def _log(self, msg: str) -> None:
        print(f"[DualSFT] {msg}")

    @staticmethod
    def _resolve_score_dtype(name: str) -> torch.dtype:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if name not in mapping:
            raise ValueError(f"Unsupported score_dtype: {name}")
        return mapping[name]

    def _stage1_dir(self) -> Path:
        return Path(self.cfg.output_dir) / "stage1"

    def _stage2_dir(self) -> Path:
        return Path(self.cfg.output_dir) / "stage2"

    def _stage3_dir(self) -> Path:
        return Path(self.cfg.output_dir) / "stage3"

    def _warmup_model_dir(self) -> Path:
        return self._stage1_dir() / "warmup_model"

    def _warmup_artifact_file(self) -> Path:
        return self._stage1_dir() / "warmup_artifacts.pt"

    def _selection_artifact_file(self) -> Path:
        return self._stage2_dir() / "selection_artifacts.pt"

    def _selection_summary_file(self) -> Path:
        return self._stage2_dir() / "selection_summary.json"

    def _splits_file(self) -> Path:
        return self._stage1_dir() / "splits.json"

    def _selected_data_indices_file(self) -> Path:
        return self._stage3_dir() / "selected_data_indices.json"

    def _selected_data_jsonl_file(self) -> Path:
        return self._stage3_dir() / "selected_data.jsonl"

    def _selected_param_mask_file(self) -> Path:
        return self._stage3_dir() / "selected_param_mask.pt"

    def _selected_param_summary_file(self) -> Path:
        return self._stage3_dir() / "selected_param_summary.json"

    def _data_scores_file(self) -> Path:
        return self._stage2_dir() / "data_scores.json"

    def _selection_diagnostics_file(self) -> Path:
        return self._stage2_dir() / "selection_diagnostics.json"

    def _param_scores_file(self) -> Path:
        return self._stage2_dir() / "param_scores.pt"

    @staticmethod
    def _fmt_float_for_name(x: float) -> str:
        return format(float(x), ".6g").replace("+", "")

    def _auto_final_model_tag(self) -> str:
        lr = self.cfg.final_learning_rate or self.cfg.learning_rate
        data_budget = (
            f"n{int(self.cfg.data_budget)}"
            if self.cfg.data_budget is not None
            else f"r{self._fmt_float_for_name(self.cfg.data_budget_ratio)}"
        )
        param_budget = (
            f"n{int(self.cfg.param_budget)}"
            if self.cfg.param_budget is not None
            else f"r{self._fmt_float_for_name(self.cfg.param_budget_ratio)}"
        )
        return (
            f"lr{self._fmt_float_for_name(lr)}"
            f"_bs{int(self.cfg.final_batch_size)}x{int(self.cfg.final_grad_accum_steps)}"
            f"_ep{int(self.cfg.final_epochs)}"
            f"_data{data_budget}"
            f"_param{param_budget}"
            f"_seed{int(self.cfg.seed)}"
        )

    def _final_model_dir(self) -> Path:
        tag = (self.cfg.final_model_tag or "").strip()
        if not tag:
            tag = self._auto_final_model_tag()
        return self._stage3_dir() / "final_models" / f"final_model_{tag}"

    def _run_summary_file(self) -> Path:
        return self._stage3_dir() / "run_summary.json"

    @staticmethod
    def _write_json(path: Path, obj: Dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _first_existing_file(paths: List[Path]) -> Path | None:
        for p in paths:
            if p.exists() and p.is_file():
                return p
        return None

    @staticmethod
    def _first_existing_dir(paths: List[Path]) -> Path | None:
        for p in paths:
            if p.exists() and p.is_dir():
                return p
        return None

    def _legacy_warmup_artifact_paths(self) -> List[Path]:
        root = Path(self.cfg.output_dir)
        return [
            self._warmup_artifact_file(),
            root / "stage1_warmup_artifacts.pt",
        ]

    def _legacy_warmup_model_paths(self) -> List[Path]:
        root = Path(self.cfg.output_dir)
        return [
            self._warmup_model_dir(),
            root / "stage1_warmup_model",
            root / "warmup_model",
        ]

    def _load_tokenizer(self, model_name_or_path: str | None = None):
        path = model_name_or_path or self.cfg.model_name_or_path
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok

    def _load_model(self, model_name_or_path: str | None = None, device: torch.device | None = None) -> nn.Module:
        path = model_name_or_path or self.cfg.model_name_or_path
        target_device = device or self.device
        torch_dtype = self.dtype if self.dtype != torch.float32 else None
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )
        model.config.use_cache = False
        model.to(target_device)
        return model

    def _build_adamw(self, params, lr: float) -> torch.optim.Optimizer:
        params = list(params)
        kwargs = {
            "lr": lr,
            "weight_decay": self.cfg.weight_decay,
        }
        if self.device.type == "cuda":
            try:
                return AdamW(params, fused=True, **kwargs)
            except Exception:
                pass
        return AdamW(params, **kwargs)

    def _load_teacher_model(self) -> nn.Module:
        model = self._load_model(self.cfg.model_name_or_path, self.teacher_device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    @staticmethod
    def _move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in batch.items():
            if k == "indices":
                out[k] = v
            else:
                out[k] = v.to(device)
        return out

    @staticmethod
    def _per_sample_nll(model: nn.Module, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        )
        logits = outputs.logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = batch["labels"][:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shift_labels.size())

        valid = shift_labels.ne(-100)
        denom = valid.sum(dim=1).clamp_min(1)
        per_sample = (token_loss * valid).sum(dim=1) / denom
        return per_sample

    def _new_score_vector(self, layout) -> torch.Tensor:
        total = layout[-1].end if layout else 0
        return torch.zeros(total, device="cpu", dtype=self.score_dtype)

    @staticmethod
    def _accumulate_grads_to_cpu_vector(
        acc: torch.Tensor,
        layout,
        grads,
        scale: float = 1.0,
    ) -> None:
        for item, g in zip(layout, grads):
            if g is None:
                continue
            v = g.detach().reshape(-1).to(device="cpu", dtype=acc.dtype)
            if scale != 1.0:
                v = v * scale
            acc[item.start : item.end].add_(v)

    def _normalize(self, vec: torch.Tensor, denom: float) -> torch.Tensor:
        if self.cfg.gradient_normalize == "mean":
            denom = max(float(denom), 1.0)
            return vec / denom
        return vec

    def _resolve_budget(self, total: int, ratio: float, explicit: int | None) -> int:
        if explicit is not None:
            return max(1, min(int(explicit), total))
        return max(1, min(int(total * ratio), total))

    def _sample_indices(self, total: int, ratio: float, seed: int) -> List[int]:
        count = max(1, min(int(total * ratio), total))
        rng = random.Random(seed)
        return rng.sample(list(range(total)), count)

    def _build_splits(self, n_train: int) -> Dict[str, List[int]]:
        warm_indices = self._sample_indices(n_train, self.cfg.warmup_ratio, self.cfg.seed)
        anchor_indices = self._sample_indices(n_train, self.cfg.anchor_ratio, self.cfg.seed + 1)

        if self.cfg.score_pool_ratio >= 1.0:
            score_indices = list(range(n_train))
        else:
            score_indices = self._sample_indices(n_train, self.cfg.score_pool_ratio, self.cfg.seed + 2)

        return {
            "warm_indices": warm_indices,
            "anchor_indices": anchor_indices,
            "score_indices": score_indices,
        }

    def _warmup(self, model: nn.Module, warm_dataset: IndexedTextDataset):
        layout = get_trainable_layout(model)
        optimizer = self._build_adamw(model.parameters(), lr=self.cfg.learning_rate)
        loader = make_dataloader(
            warm_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.warmup_batch_size,
            max_length=self.cfg.max_length,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_steps = max(1, int(self.cfg.warmup_grad_accum_steps))
        micro_step = 0

        for epoch in range(self.cfg.warmup_epochs):
            pbar = tqdm(loader, desc=f"Warmup epoch {epoch + 1}")
            for batch in pbar:
                batch = self._move_batch(batch, self.device)
                per_sample_loss = self._per_sample_nll(model, batch)
                loss = per_sample_loss.mean()

                (loss / accum_steps).backward()
                micro_step += 1

                if micro_step % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(loss=float(loss.item()))

            if micro_step % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        c = extract_adamw_second_moment(
            layout,
            optimizer,
            device=torch.device("cpu"),
            dtype=self.score_dtype,
        )
        return c

    def _compute_vnew(self, model: nn.Module, val_dataset: IndexedTextDataset) -> torch.Tensor:
        model.eval()
        layout = get_trainable_layout(model)
        params = [x.param for x in layout]
        loader = make_dataloader(
            val_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.eval_batch_size,
            max_length=self.cfg.max_length,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )
        acc = self._new_score_vector(layout)
        total = 0
        for batch in tqdm(loader, desc="Compute v_new"):
            batch = self._move_batch(batch, self.device)
            per_sample_loss = self._per_sample_nll(model, batch)
            scalar = per_sample_loss.sum()
            grads = torch.autograd.grad(
                scalar,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            self._accumulate_grads_to_cpu_vector(acc, layout, grads)
            total += int(per_sample_loss.numel())
        return self._normalize(acc, total)

    def _confidence_weights(self, q: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        mode = self.cfg.confidence_mode
        if mode == "uniform":
            return torch.ones(q.size(0), device=q.device, dtype=torch.float32)

        if mode == "max_prob":
            conf_tok = q.max(dim=-1).values
        elif mode == "entropy_inv":
            entropy = -(q * q.clamp_min(1e-12).log()).sum(dim=-1)
            conf_tok = 1.0 - entropy / math.log(q.size(-1))
        else:
            raise ValueError(f"Unsupported confidence mode: {mode}")

        valid = valid_mask.to(conf_tok.dtype)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (conf_tok * valid).sum(dim=1) / denom

    def _compute_vprior(self, model: nn.Module, teacher_model: nn.Module, anchor_dataset: IndexedTextDataset) -> torch.Tensor:
        model.eval()
        layout = get_trainable_layout(model)
        params = [x.param for x in layout]
        loader = make_dataloader(
            anchor_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.eval_batch_size,
            max_length=self.cfg.max_length,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        acc = self._new_score_vector(layout)
        total_weight = 0.0

        for batch in tqdm(loader, desc="Compute v_prior (CWSD)"):
            batch_student = self._move_batch(batch, self.device)
            batch_teacher = self._move_batch(batch, self.teacher_device)

            with torch.no_grad():
                t_out = teacher_model(
                    input_ids=batch_teacher["input_ids"],
                    attention_mask=batch_teacher["attention_mask"],
                    use_cache=False,
                )
                teacher_logits = t_out.logits[:, :-1, :].float() / self.cfg.tau

            s_out = model(
                input_ids=batch_student["input_ids"],
                attention_mask=batch_student["attention_mask"],
                use_cache=False,
            )
            student_logits = s_out.logits[:, :-1, :].float() / self.cfg.tau

            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(student_logits.device)

            q = torch.softmax(teacher_logits, dim=-1)
            log_q = torch.log_softmax(teacher_logits, dim=-1)
            log_p = torch.log_softmax(student_logits, dim=-1)

            valid = batch_student["labels"][:, 1:].ne(-100)

            kl_tok = (q * (log_q - log_p)).sum(dim=-1)
            denom = valid.sum(dim=1).clamp_min(1)
            per_sample_kl = (kl_tok * valid).sum(dim=1) / denom

            weights = self._confidence_weights(q, valid)
            weighted = (weights * per_sample_kl).sum() * (self.cfg.tau ** 2)

            grads = torch.autograd.grad(
                weighted,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            self._accumulate_grads_to_cpu_vector(acc, layout, grads)
            total_weight += float(weights.sum().item())

        return self._normalize(acc, total_weight)

    def _accumulate_G(self, model: nn.Module, score_dataset: IndexedTextDataset) -> torch.Tensor:
        model.eval()
        layout = get_trainable_layout(model)
        params = [x.param for x in layout]
        loader = make_dataloader(
            score_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.score_batch_size,
            max_length=self.cfg.max_length,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        G = self._new_score_vector(layout)
        for batch in tqdm(loader, desc="Pass-1 accumulate G"):
            batch = self._move_batch(batch, self.device)
            per_sample_loss = self._per_sample_nll(model, batch)
            scalar = per_sample_loss.sum()
            grads = torch.autograd.grad(
                scalar,
                params,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            self._accumulate_grads_to_cpu_vector(G, layout, grads)
        return G

    def _score_data_exact(
        self,
        model: nn.Module,
        score_dataset: IndexedTextDataset,
        u: torch.Tensor,
    ) -> Dict[int, float]:
        model.eval()
        layout = get_trainable_layout(model)
        loader = make_dataloader(
            score_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.score_batch_size,
            max_length=self.cfg.max_length,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        scores: Dict[int, float] = {}
        params = [x.param for x in layout]

        for batch in tqdm(loader, desc="Pass-2 data score (exact)"):
            indices = batch["indices"].tolist()
            batch = self._move_batch(batch, self.device)
            per_sample_loss = self._per_sample_nll(model, batch)

            for i, idx in enumerate(indices):
                grads = torch.autograd.grad(
                    per_sample_loss[i],
                    params,
                    retain_graph=i < len(indices) - 1,
                    create_graph=False,
                    allow_unused=True,
                )
                score = 0.0
                for item, g in zip(layout, grads):
                    if g is None:
                        continue
                    u_slice = u[item.start : item.end].to(device=g.device, dtype=g.dtype)
                    score += float((u_slice * g.detach().reshape(-1)).sum().item())
                scores[int(idx)] = float(score)

        return scores

    def _score_data_ghost(
        self,
        model: nn.Module,
        score_dataset: IndexedTextDataset,
        u: torch.Tensor,
    ) -> Dict[int, float]:
        model.eval()
        layout = get_trainable_layout(model)
        scorer = GhostDotLinearScorer(model, layout, u)

        if scorer.covered_ratio < 0.999:
            self._log(
                f"Ghost scorer covers {scorer.covered_ratio * 100:.2f}% trainable params."
            )
            if self.cfg.ghost_fallback_exact:
                scorer.close()
                self._log("Falling back to exact data scoring (slow).")
                return self._score_data_exact(model, score_dataset, u)

        loader = make_dataloader(
            score_dataset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.score_batch_size,
            max_length=self.cfg.max_length,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        scores: Dict[int, float] = {}
        for batch in tqdm(loader, desc="Pass-2 data score (ghost)"):
            indices = batch["indices"].tolist()
            batch = self._move_batch(batch, self.device)

            model.zero_grad(set_to_none=True)
            scorer.start_batch(batch_size=len(indices), device=self.device)
            per_sample_loss = self._per_sample_nll(model, batch)
            scalar = per_sample_loss.sum()
            scalar.backward()
            batch_scores = scorer.consume_batch_scores().cpu().tolist()

            for idx, s in zip(indices, batch_scores):
                scores[int(idx)] = float(s)

        scorer.close()
        return scores

    def _select_top_data(self, scores: Dict[int, float], budget: int) -> List[int]:
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:budget]]

    @staticmethod
    def _scores_dict_to_tensors(scores: Dict[int, float]) -> Tuple[torch.Tensor, torch.Tensor]:
        items = sorted(scores.items(), key=lambda x: x[0])
        indices = torch.tensor([int(i) for i, _ in items], dtype=torch.long)
        values = torch.tensor([float(s) for _, s in items], dtype=torch.float32)
        return indices, values

    @staticmethod
    def _topk_positions(values: torch.Tensor, k: int) -> torch.Tensor:
        if values.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long)
        k = min(int(k), values.numel())
        return torch.topk(values, k=k, largest=True).indices

    def _maybe_exact_rerank_topm(
        self,
        model: nn.Module,
        score_dataset: IndexedTextDataset,
        ghost_scores: Dict[int, float],
        u: torch.Tensor,
    ):
        topm = max(0, int(self.cfg.data_rerank_topm))
        if topm <= 0 or len(ghost_scores) == 0:
            return None

        candidate_budget = min(topm, len(ghost_scores))
        candidate_orig_indices = self._select_top_data(ghost_scores, candidate_budget)
        local_lookup = {int(orig_idx): pos for pos, orig_idx in enumerate(score_dataset.indices)}
        candidate_local_indices = [local_lookup[idx] for idx in candidate_orig_indices if idx in local_lookup]
        if not candidate_local_indices:
            return None

        rerank_dataset = score_dataset.subset(candidate_local_indices)
        rerank_scores = self._score_data_exact(model, rerank_dataset, u)
        rerank_indices, rerank_values = self._scores_dict_to_tensors(rerank_scores)
        return {
            "candidate_count": int(len(candidate_local_indices)),
            "indices": rerank_indices,
            "values": rerank_values,
        }

    @staticmethod
    def _select_indices_from_scores(
        score_indices: torch.Tensor,
        score_values: torch.Tensor,
        budget: int,
    ) -> List[int]:
        top_pos = DualSFTRunner._topk_positions(score_values, budget)
        return score_indices[top_pos].tolist()

    @staticmethod
    def _sample_tensor(values: torch.Tensor, max_points: int = 100000) -> torch.Tensor:
        flat = values.detach().reshape(-1).to(device="cpu")
        total = int(flat.numel())
        if total == 0:
            return flat.to(dtype=torch.float32)
        if total <= max_points:
            return flat.to(dtype=torch.float32)
        sample_idx = torch.linspace(0, total - 1, steps=max_points, dtype=torch.float32).round().to(dtype=torch.long)
        return flat.index_select(0, sample_idx).to(dtype=torch.float32)

    @staticmethod
    def _tensor_summary(values: torch.Tensor, max_points: int = 100000) -> Dict:
        sample = DualSFTRunner._sample_tensor(values, max_points=max_points)
        total = int(values.numel())
        if total == 0:
            return {
                "count_total": 0,
                "count_sampled": 0,
                "sampled": False,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "mean_abs": None,
                "positive_fraction": None,
                "negative_fraction": None,
                "zero_fraction": None,
                "quantiles": {},
                "abs_quantiles": {},
            }

        abs_sample = sample.abs()
        q_points = [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]
        q_tensor = torch.tensor(q_points, dtype=torch.float32)
        raw_q = torch.quantile(sample, q_tensor)
        abs_q = torch.quantile(abs_sample, q_tensor)
        q_keys = ["q0", "q1", "q5", "q50", "q95", "q99", "q100"]

        return {
            "count_total": total,
            "count_sampled": int(sample.numel()),
            "sampled": bool(sample.numel() != total),
            "min": float(sample.min().item()),
            "max": float(sample.max().item()),
            "mean": float(sample.mean().item()),
            "std": float(sample.std(unbiased=False).item()),
            "mean_abs": float(abs_sample.mean().item()),
            "positive_fraction": float((sample > 0).float().mean().item()),
            "negative_fraction": float((sample < 0).float().mean().item()),
            "zero_fraction": float((sample == 0).float().mean().item()),
            "quantiles": {k: float(v.item()) for k, v in zip(q_keys, raw_q)},
            "abs_quantiles": {k: float(v.item()) for k, v in zip(q_keys, abs_q)},
        }

    @staticmethod
    def _sample_cosine(x: torch.Tensor, y: torch.Tensor, max_points: int = 100000):
        xs = DualSFTRunner._sample_tensor(x, max_points=max_points)
        ys = DualSFTRunner._sample_tensor(y, max_points=max_points)
        if xs.numel() == 0 or ys.numel() == 0 or xs.numel() != ys.numel():
            return None
        x_centered = xs - xs.mean()
        y_centered = ys - ys.mean()
        denom = torch.linalg.vector_norm(x_centered) * torch.linalg.vector_norm(y_centered)
        denom_val = float(denom.item())
        if denom_val == 0.0:
            return None
        return float((x_centered * y_centered).sum().item() / denom_val)

    @staticmethod
    def _sample_l2(values: torch.Tensor, max_points: int = 100000) -> float:
        sample = DualSFTRunner._sample_tensor(values, max_points=max_points)
        if sample.numel() == 0:
            return 0.0
        return float(torch.linalg.vector_norm(sample).item())

    @staticmethod
    def _score_topk_preview(
        score_indices: torch.Tensor,
        score_values: torch.Tensor,
        budget: int,
    ) -> Dict:
        if score_values.numel() == 0 or budget <= 0:
            return {
                "budget": 0,
                "threshold": None,
                "mean_score": None,
                "top10_indices": [],
                "top10_scores": [],
            }
        budget = min(int(budget), int(score_values.numel()))
        top_pos = DualSFTRunner._topk_positions(score_values, budget)
        top_scores = score_values[top_pos].to(dtype=torch.float32)
        top_indices = score_indices[top_pos]
        preview_count = min(10, budget)
        return {
            "budget": budget,
            "threshold": float(top_scores.min().item()),
            "mean_score": float(top_scores.mean().item()),
            "top10_indices": [int(x) for x in top_indices[:preview_count].tolist()],
            "top10_scores": [float(x) for x in top_scores[:preview_count].tolist()],
        }

    @staticmethod
    def _overlap_stats(a: List[int], b: List[int]) -> Dict:
        set_a = set(int(x) for x in a)
        set_b = set(int(x) for x in b)
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        return {
            "overlap_count": int(inter),
            "overlap_ratio_vs_a": float(inter / max(len(set_a), 1)),
            "overlap_ratio_vs_b": float(inter / max(len(set_b), 1)),
            "jaccard": float(inter / max(union, 1)),
        }

    def _build_selection_diagnostics(
        self,
        *,
        phi_param: torch.Tensor,
        data_score_indices: torch.Tensor,
        data_score_values: torch.Tensor,
        rerank_payload,
        v_new: torch.Tensor,
        v_prior: torch.Tensor,
        G: torch.Tensor,
        u: torch.Tensor,
        c: torch.Tensor,
        splits: Dict[str, List[int]],
        train_size: int,
        val_size: int,
        anchor_size: int,
        score_pool_size: int,
    ) -> Dict:
        param_budget = self._resolve_budget(
            total=int(phi_param.numel()),
            ratio=self.cfg.param_budget_ratio,
            explicit=self.cfg.param_budget,
        )
        data_budget = self._resolve_budget(
            total=int(data_score_values.numel()),
            ratio=self.cfg.data_budget_ratio,
            explicit=self.cfg.data_budget,
        )

        raw_top_data = self._select_indices_from_scores(data_score_indices, data_score_values, data_budget)
        diagnostics = {
            "config": {
                "learning_rate": float(self.cfg.learning_rate),
                "lambda_new": float(self.cfg.lambda_new),
                "lambda_prior": float(self.cfg.lambda_prior),
                "tau": float(self.cfg.tau),
                "score_method": self.cfg.score_method,
                "data_rerank_topm": int(self.cfg.data_rerank_topm),
                "topk_use_abs": bool(self.cfg.topk_use_abs),
                "use_diagonal_second_order": bool(self.cfg.use_diagonal_second_order),
                "data_budget_ratio": float(self.cfg.data_budget_ratio),
                "param_budget_ratio": float(self.cfg.param_budget_ratio),
                "data_budget": int(data_budget),
                "param_budget": int(param_budget),
            },
            "split_sizes": {
                "train": int(train_size),
                "validation": int(val_size),
                "warmup": int(len(splits.get("warm_indices", []))),
                "anchor": int(anchor_size),
                "score_pool": int(score_pool_size),
            },
            "vector_metrics": {
                "sample_l2": {
                    "v_new": self._sample_l2(v_new),
                    "v_prior": self._sample_l2(v_prior),
                    "scaled_v_new": abs(float(self.cfg.lambda_new)) * self._sample_l2(v_new),
                    "scaled_v_prior": abs(float(self.cfg.lambda_prior)) * self._sample_l2(v_prior),
                    "G": self._sample_l2(G),
                    "u": self._sample_l2(u),
                    "phi_param": self._sample_l2(phi_param),
                    "c": self._sample_l2(c),
                },
                "sample_correlations": {
                    "vnew_vs_vprior": self._sample_cosine(v_new, v_prior),
                    "u_vs_G": self._sample_cosine(u, G),
                    "u_vs_phi_param": self._sample_cosine(u, phi_param),
                },
            },
            "param_scores": {
                "selection_mode": "abs_topk" if self.cfg.topk_use_abs else "raw_topk",
                "summary": self._tensor_summary(phi_param),
            },
            "data_scores": {
                "full_pool_summary": self._tensor_summary(data_score_values, max_points=200000),
                "current_budget_preview": self._score_topk_preview(data_score_indices, data_score_values, data_budget),
            },
            "rerank": {
                "enabled": bool(rerank_payload is not None),
            },
        }

        if rerank_payload is not None:
            rerank_indices = rerank_payload["indices"].to(device="cpu", dtype=torch.long)
            rerank_values = rerank_payload["values"].to(device="cpu", dtype=torch.float32)
            rerank_budget = min(int(data_budget), int(rerank_values.numel()))
            rerank_top = self._select_indices_from_scores(rerank_indices, rerank_values, rerank_budget)
            ghost_lookup = {int(i): float(s) for i, s in zip(data_score_indices.tolist(), data_score_values.tolist())}
            ghost_on_candidates = torch.tensor(
                [ghost_lookup[int(idx)] for idx in rerank_indices.tolist()],
                dtype=torch.float32,
            )
            exact_on_candidates = rerank_values.to(dtype=torch.float32)

            diagnostics["rerank"] = {
                "enabled": True,
                "candidate_count": int(rerank_values.numel()),
                "candidate_ratio_vs_pool": float(rerank_values.numel() / max(int(data_score_values.numel()), 1)),
                "candidate_covers_budget": bool(rerank_values.numel() >= data_budget),
                "score_summary": self._tensor_summary(rerank_values, max_points=200000),
                "current_budget_preview": self._score_topk_preview(rerank_indices, rerank_values, rerank_budget),
                "raw_vs_exact_overlap_at_budget": self._overlap_stats(raw_top_data, rerank_top),
                "ghost_vs_exact_candidate_correlation": self._sample_cosine(ghost_on_candidates, exact_on_candidates, max_points=200000),
            }

        return diagnostics

    def _save_param_summary(
        self,
        model: nn.Module,
        mask_vector: torch.Tensor,
        layout: List[ParameterSlice] | None = None,
    ) -> None:
        if layout is None:
            layout = get_trainable_layout(model)
        named_mask = vector_mask_to_named_tensors(mask_vector, layout, device=torch.device("cpu"))
        mask_path = self._selected_param_mask_file()
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(named_mask, mask_path)

        summary = []
        for item in layout:
            mask = named_mask[item.name]
            selected = int(mask.sum().item())
            total = int(mask.numel())
            summary.append(
                {
                    "name": item.name,
                    "selected": selected,
                    "total": total,
                    "density": float(selected / max(total, 1)),
                }
            )

        self._write_json(
            self._selected_param_summary_file(),
            {
                "num_tensors": len(summary),
                "total_selected": int(mask_vector.sum().item()),
                "total_trainable": int(mask_vector.numel()),
                "tensor_stats": summary,
            },
        )

    def _save_selected_data(self, selected_indices: List[int]) -> None:
        self._write_json(self._selected_data_indices_file(), {"indices": selected_indices})
        if not self.cfg.save_selected_data:
            return

        records = load_records(self.cfg.train_file)
        selected_records = [records[i] for i in selected_indices]
        save_records_jsonl(selected_records, str(self._selected_data_jsonl_file()))

    def _load_warmup_bundle(self):
        warmup_artifact = self._first_existing_file(self._legacy_warmup_artifact_paths())
        warmup_model_dir = self._first_existing_dir(self._legacy_warmup_model_paths())

        if warmup_artifact is None or warmup_model_dir is None:
            raise FileNotFoundError(
                "Stage-1 artifacts not found. Please run `--stage warmup` first."
            )

        payload = torch.load(warmup_artifact, map_location="cpu")
        c = payload["c"].to(device="cpu", dtype=self.score_dtype)
        splits = payload["splits"]
        train_size = payload.get("train_size")

        self.tokenizer = self._load_tokenizer(str(warmup_model_dir))
        model = self._load_model(str(warmup_model_dir), self.device)
        return model, c, splits, train_size

    @staticmethod
    def _pick_first(obj: dict, keys: list[str]):
        for k in keys:
            if k in obj and obj[k] is not None:
                return obj[k]
        return None

    def _load_selection_bundle(self):
        search_dirs: list[Path] = []
        if getattr(self.cfg, "selection_dir", None):
            search_dirs.append(Path(self.cfg.selection_dir))
        search_dirs.append(Path(self.cfg.output_dir))

        # Remove duplicate search directories.
        uniq = []
        seen = set()
        for d in search_dirs:
            s = str(d.resolve())
            if s not in seen:
                seen.add(s)
                uniq.append(d)
        search_dirs = uniq

        for d in search_dirs:
            if not d.exists():
                continue

            # 1) Prefer consolidated artifacts.
            merged_candidates = [
                d / "stage2" / "selection_artifacts.pt",
                d / "stage2_selection_artifacts.pt",
            ]
            merged = self._first_existing_file(merged_candidates)
            if merged is not None:
                obj = torch.load(merged, map_location="cpu")

                phi_param = self._pick_first(obj, ["phi_param", "param_scores"])
                data_score_values = self._pick_first(obj, ["data_score_values", "data_scores", "phi_data"])
                data_score_indices = self._pick_first(obj, ["data_score_indices"])
                data_rerank_values = self._pick_first(obj, ["data_rerank_values"])
                data_rerank_indices = self._pick_first(obj, ["data_rerank_indices"])

                if data_score_indices is None and data_score_values is not None:
                    data_score_indices = torch.arange(data_score_values.numel(), dtype=torch.long)

                if data_rerank_values is None or data_rerank_indices is None:
                    data_rerank_values = None
                    data_rerank_indices = None

                if phi_param is not None and data_score_values is not None:
                    mode = "full_scores"
                    summary_candidates = [
                        d / "stage2" / "selection_summary.json",
                        d / "stage2_selection_summary.json",
                    ]
                    smy = self._first_existing_file(summary_candidates)
                    if smy is not None:
                        try:
                            mode = json.loads(smy.read_text()).get("mode", mode)
                        except Exception:
                            pass
                    return {
                        "mode": mode,
                        "phi_param": phi_param,
                        "data_score_values": data_score_values,
                        "data_score_indices": data_score_indices.long(),
                        "data_rerank_values": data_rerank_values.float() if data_rerank_values is not None else None,
                        "data_rerank_indices": data_rerank_indices.long() if data_rerank_indices is not None else None,
                        "data_rerank_candidate_count": int(obj.get("data_rerank_candidate_count", 0)),
                        "base_dir": str(d),
                    }

            # 2) Fall back to split files for legacy layouts.
            p_param = self._first_existing_file([d / "stage2" / "param_scores.pt", d / "param_scores.pt"])
            p_data_json = self._first_existing_file([d / "stage2" / "data_scores.json", d / "data_scores.json"])
            p_data_idx = self._first_existing_file([d / "stage3" / "selected_data_indices.json", d / "selected_data_indices.json"])

            if p_param is not None and p_data_json is not None:
                phi_param = torch.load(p_param, map_location="cpu")
                with open(p_data_json, "r", encoding="utf-8") as f:
                    ds = json.load(f)

                if isinstance(ds, dict) and "scores" in ds:
                    data_items = ds.get("scores", [])
                    data_score_indices = torch.tensor([int(x["index"]) for x in data_items], dtype=torch.long)
                    data_score_values = torch.tensor([float(x["score"]) for x in data_items], dtype=torch.float32)
                else:
                    data_score_values = torch.tensor(ds, dtype=torch.float32)
                    data_score_indices = None

                if p_data_idx is not None:
                    with open(p_data_idx, "r", encoding="utf-8") as f:
                        idx = json.load(f)
                    if isinstance(idx, dict) and "indices" in idx:
                        idx = idx["indices"]
                    idx_tensor = torch.tensor(idx, dtype=torch.long)
                    if idx_tensor.numel() == data_score_values.numel():
                        data_score_indices = idx_tensor
                if data_score_indices is None:
                    data_score_indices = torch.arange(data_score_values.numel(), dtype=torch.long)

                return {
                    "mode": "full_scores",
                    "phi_param": phi_param,
                    "data_score_values": data_score_values,
                    "data_score_indices": data_score_indices,
                    "base_dir": str(d),
                }

        raise FileNotFoundError(
            f"Stage-2 artifacts not found. searched_dirs={[str(x) for x in search_dirs]}"
        )

    def _restricted_finetune(
        self,
        selected_indices: List[int],
        mask_vector: torch.Tensor,
        train_dataset: IndexedTextDataset,
    ) -> Tuple[nn.Module, Dict[str, torch.Tensor], List[ParameterSlice]]:
        self._log("Reloading base model for restricted fine-tuning from theta_old")
        model = self._load_model(self.cfg.model_name_or_path, self.device)
        layout = get_trainable_layout(model)
        named_mask = vector_mask_to_named_tensors(mask_vector, layout, device=self.device)
        for name, p in model.named_parameters():
            m = named_mask.get(name)
            if m is None or not bool(m.any().item()):
                p.requires_grad_(False)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters remain after masking.")
        grad_hooks = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            m = named_mask.get(name)
            if m is None:
                continue
            if bool(m.all().item()):
                continue
            grad_hooks.append(p.register_hook(lambda grad, mask=m: grad * mask))

        subset = train_dataset.subset(selected_indices)
        loader = make_dataloader(
            subset,
            tokenizer=self.tokenizer,
            batch_size=self.cfg.final_batch_size,
            max_length=self.cfg.max_length,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            prefetch_factor=self.cfg.prefetch_factor,
            mask_prompt_loss=not self.cfg.train_on_prompt,
        )

        lr = self.cfg.final_learning_rate or self.cfg.learning_rate
        optimizer = self._build_adamw(trainable_params, lr=lr)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        accum_steps = max(1, int(self.cfg.final_grad_accum_steps))  # NEW
        micro_step = 0  # NEW: count dataloader steps

        for epoch in range(self.cfg.final_epochs):
            pbar = tqdm(loader, desc=f"Restricted FT epoch {epoch + 1}")
            for batch in pbar:
                batch = self._move_batch(batch, self.device)

                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    use_cache=False,
                )
                loss = outputs.loss

                # NEW: scale loss for accumulation
                (loss / accum_steps).backward()
                micro_step += 1

                # step every accum_steps
                if micro_step % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                pbar.set_postfix(loss=float(loss.item()))

            # NEW: flush remainder at end of epoch
            if micro_step % accum_steps != 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        for h in grad_hooks:
            h.remove()

        return model, named_mask, layout

    def stage_warmup(self) -> None:
        set_seed(self.cfg.seed)
        self.tokenizer = self._load_tokenizer(self.cfg.model_name_or_path)

        train_dataset = load_text_dataset(self.cfg.train_file, self.cfg)
        n_train = len(train_dataset)
        splits = self._build_splits(n_train)

        warm_dataset = train_dataset.subset(splits["warm_indices"])
        self._log(f"Stage-1 warmup: train={n_train}, warmup={len(warm_dataset)}")

        model = self._load_model(self.cfg.model_name_or_path, self.device)
        c = self._warmup(model, warm_dataset)

        warmup_model_dir = self._warmup_model_dir()
        warmup_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(warmup_model_dir)
        self.tokenizer.save_pretrained(warmup_model_dir)

        warmup_artifact_file = self._warmup_artifact_file()
        warmup_artifact_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "c": c.detach().cpu(),
                "splits": splits,
                "train_size": n_train,
            },
            warmup_artifact_file,
        )
        self._write_json(self._splits_file(), splits)
        self._log(f"Stage-1 done. Saved warmup model to {warmup_model_dir}")

    def stage_select(self) -> None:
        set_seed(self.cfg.seed)

        train_dataset = load_text_dataset(self.cfg.train_file, self.cfg)
        val_dataset = load_text_dataset(self.cfg.validation_file, self.cfg)

        model, c, splits, train_size = self._load_warmup_bundle()
        if train_size is not None and len(train_dataset) != int(train_size):
            raise ValueError(
                f"Train set size mismatch: stage1 used {train_size}, current train size is {len(train_dataset)}"
            )

        anchor_dataset = train_dataset.subset(splits["anchor_indices"])
        score_dataset = train_dataset.subset(splits["score_indices"])

        self._log(
            f"Stage-2 select: val={len(val_dataset)}, anchor={len(anchor_dataset)}, score_pool={len(score_dataset)}"
        )

        v_new = self._compute_vnew(model, val_dataset)
        teacher_model = self._load_teacher_model()
        v_prior = self._compute_vprior(model, teacher_model, anchor_dataset)
        del teacher_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        G = self._accumulate_G(model, score_dataset)

        u = self.cfg.learning_rate * (
            self.cfg.lambda_new * v_new + self.cfg.lambda_prior * v_prior
        )
        if self.cfg.use_diagonal_second_order:
            u = u - 0.5 * (self.cfg.learning_rate ** 2) * (c * G)

        phi_param = u * G
        rerank_payload = None
        if self.cfg.score_method == "ghost_linear":
            data_scores = self._score_data_ghost(model, score_dataset, u)
            rerank_payload = self._maybe_exact_rerank_topm(model, score_dataset, data_scores, u)
        elif self.cfg.score_method == "exact":
            data_scores = self._score_data_exact(model, score_dataset, u)
        else:
            raise ValueError(f"Unsupported score_method: {self.cfg.score_method}")

        data_score_indices, data_score_values = self._scores_dict_to_tensors(data_scores)

        selection_payload = {
            "phi_param": phi_param.detach().cpu(),
            "data_score_indices": data_score_indices,
            "data_score_values": data_score_values,
        }
        if rerank_payload is not None:
            selection_payload.update(
                {
                    "data_rerank_indices": rerank_payload["indices"],
                    "data_rerank_values": rerank_payload["values"],
                    "data_rerank_candidate_count": int(rerank_payload["candidate_count"]),
                }
            )
        if self.cfg.save_full_intermediates:
            selection_payload.update(
                {
                    "u": u.detach().cpu(),
                    "G": G.detach().cpu(),
                    "v_new": v_new.detach().cpu(),
                    "v_prior": v_prior.detach().cpu(),
                    "c": c.detach().cpu(),
                    "splits": splits,
                }
            )

        selection_file = self._selection_artifact_file()
        selection_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(selection_payload, selection_file)

        if self.cfg.save_param_scores:
            param_score_file = self._param_scores_file()
            param_score_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"phi_param": phi_param.detach().cpu()}, param_score_file)

        if self.cfg.save_data_scores:
            sorted_scores = sorted(data_scores.items(), key=lambda x: x[1], reverse=True)
            data_scores_payload = {
                "score_method": self.cfg.score_method,
                "scores": [{"index": int(i), "score": float(s)} for i, s in sorted_scores],
            }
            if rerank_payload is not None:
                rerank_items = list(zip(rerank_payload["indices"].tolist(), rerank_payload["values"].tolist()))
                rerank_sorted = sorted(rerank_items, key=lambda x: x[1], reverse=True)
                data_scores_payload["rerank_topm"] = int(rerank_payload["candidate_count"])
                data_scores_payload["rerank_scores"] = [
                    {"index": int(i), "score": float(s)} for i, s in rerank_sorted
                ]
            self._write_json(self._data_scores_file(), data_scores_payload)

        self._write_json(
            self._selection_summary_file(),
            {
                "mode": "full_scores_only",
                "score_pool_size": len(score_dataset),
                "param_score_dim": int(phi_param.numel()),
                "data_score_count": int(data_score_values.numel()),
                "score_method": self.cfg.score_method,
                "lambda_new": float(self.cfg.lambda_new),
                "lambda_prior": float(self.cfg.lambda_prior),
                "data_rerank_topm": int(self.cfg.data_rerank_topm),
                "data_rerank_candidate_count": int(rerank_payload["candidate_count"]) if rerank_payload is not None else 0,
                "use_diagonal_second_order": self.cfg.use_diagonal_second_order,
                "save_full_intermediates": self.cfg.save_full_intermediates,
            },
        )

        diagnostics = self._build_selection_diagnostics(
            phi_param=phi_param,
            data_score_indices=data_score_indices,
            data_score_values=data_score_values,
            rerank_payload=rerank_payload,
            v_new=v_new,
            v_prior=v_prior,
            G=G,
            u=u,
            c=c,
            splits=splits,
            train_size=len(train_dataset),
            val_size=len(val_dataset),
            anchor_size=len(anchor_dataset),
            score_pool_size=len(score_dataset),
        )
        self._write_json(self._selection_diagnostics_file(), diagnostics)

        self._log("Stage-2 done. Saved full parameter/data scores for budget-free re-selection.")

    def stage_finetune(self) -> None:
        set_seed(self.cfg.seed)
        self.tokenizer = self._load_tokenizer(self.cfg.model_name_or_path)

        train_dataset = load_text_dataset(self.cfg.train_file, self.cfg)
        selection_bundle = self._load_selection_bundle()

        mode = str(selection_bundle.get("mode", "")).lower()
        selected_data_source = "preselected_bundle"
        # Support both full_scores and full_scores_only bundles.
        if mode in {"full_scores", "full_scores_only"}:
            phi_param = selection_bundle["phi_param"].to(device="cpu")
            data_score_indices = selection_bundle["data_score_indices"].to(device="cpu", dtype=torch.long)
            data_score_values = selection_bundle["data_score_values"].to(device="cpu", dtype=torch.float32)
            data_rerank_indices = selection_bundle.get("data_rerank_indices")
            data_rerank_values = selection_bundle.get("data_rerank_values")
            if data_rerank_indices is not None:
                data_rerank_indices = data_rerank_indices.to(device="cpu", dtype=torch.long)
            if data_rerank_values is not None:
                data_rerank_values = data_rerank_values.to(device="cpu", dtype=torch.float32)

            k = self._resolve_budget(
                total=phi_param.numel(),
                ratio=self.cfg.param_budget_ratio,
                explicit=self.cfg.param_budget,
            )
            b = self._resolve_budget(
                total=data_score_values.numel(),
                ratio=self.cfg.data_budget_ratio,
                explicit=self.cfg.data_budget,
            )

            base_model_for_layout = self._load_model(self.cfg.model_name_or_path, self.device)
            layout = get_trainable_layout(base_model_for_layout)
            param_mask_vector = self._topk_mask_with_quota(
                phi_param=phi_param,
                layout=layout,
                k=k,
                use_abs=self.cfg.topk_use_abs,
            ).to(device="cpu")
            del base_model_for_layout
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (
                data_rerank_indices is not None
                and data_rerank_values is not None
                and data_rerank_values.numel() >= b
            ):
                selected_indices = self._select_indices_from_scores(data_rerank_indices, data_rerank_values, b)
                selected_data_source = "exact_rerank_topm"
            else:
                if (
                    data_rerank_values is not None
                    and data_rerank_values.numel() > 0
                    and data_rerank_values.numel() < b
                ):
                    self._log(
                        f"Exact rerank candidate count {int(data_rerank_values.numel())} is smaller than requested data budget {b}; falling back to full-pool scores."
                    )
                selected_indices = self._select_indices_from_scores(data_score_indices, data_score_values, b)
                selected_data_source = "full_score_ranking"
        else:
            selected_indices = selection_bundle["selected_data_indices"]
            param_mask_vector = selection_bundle["param_mask_vector"].to(device="cpu")
            b = len(selected_indices)
            k = int(param_mask_vector.sum().item())

        self._log(
            f"Stage-3 finetune: selected_data={len(selected_indices)}, selected_params={int(param_mask_vector.sum().item())}"
        )

        final_model, _, layout = self._restricted_finetune(selected_indices, param_mask_vector, train_dataset)
        self._save_selected_data(selected_indices)
        self._save_param_summary(final_model, param_mask_vector, layout=layout)

        final_dir = self._final_model_dir()
        final_dir.mkdir(parents=True, exist_ok=True)
        final_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)

        self._write_json(
            self._run_summary_file(),
            {
                "stage": self.cfg.stage,
                "selected_data_count": len(selected_indices),
                "selected_param_count": int(param_mask_vector.sum().item()),
                "data_budget": int(b),
                "param_budget": int(k),
                "data_budget_ratio": float(self.cfg.data_budget_ratio),
                "param_budget_ratio": float(self.cfg.param_budget_ratio),
                "lambda_new": float(self.cfg.lambda_new),
                "lambda_prior": float(self.cfg.lambda_prior),
                "data_rerank_topm": int(self.cfg.data_rerank_topm),
                "topk_use_abs": bool(self.cfg.topk_use_abs),
                "selected_data_source": selected_data_source,
                "learning_rate": float(self.cfg.learning_rate),
                "final_learning_rate": float(self.cfg.final_learning_rate or self.cfg.learning_rate),
                "final_batch_size": int(self.cfg.final_batch_size),
                "final_grad_accum_steps": int(self.cfg.final_grad_accum_steps),
                "final_epochs": int(self.cfg.final_epochs),
                "train_on_prompt": bool(self.cfg.train_on_prompt),
                "prompt_response_separator": self.cfg.prompt_response_separator,
                "selection_source_dir": selection_bundle.get("base_dir"),
                "final_model_dir": str(final_dir),
                "final_model_tag": self.cfg.final_model_tag or self._auto_final_model_tag(),
            },
        )

        self._log(f"Stage-3 done. Final model saved to {final_dir}")

    @staticmethod
    def _layer_id_from_name(name: str) -> int | None:
        m = re.search(r"\.layers\.(\d+)\.", name)
        return int(m.group(1)) if m else None

    @staticmethod
    def _module_group_from_name(name: str) -> str:
        if ".self_attn.q_proj." in name:
            return "q_proj"
        if ".self_attn.k_proj." in name:
            return "k_proj"
        if ".self_attn.v_proj." in name:
            return "v_proj"
        if ".self_attn.o_proj." in name:
            return "o_proj"
        if ".mlp.gate_proj." in name:
            return "mlp_gate"
        if ".mlp.up_proj." in name:
            return "mlp_up"
        if ".mlp.down_proj." in name:
            return "mlp_down"
        return "other"

    @staticmethod
    def _pick_topk_indices(scores: torch.Tensor, k: int, use_abs: bool) -> torch.Tensor:
        if scores.numel() == 0 or k <= 0:
            return torch.empty(0, dtype=torch.long)
        work = scores.abs() if use_abs else scores
        k = min(int(k), work.numel())
        return torch.topk(work, k=k, largest=True).indices

    def _topk_mask_with_quota(
        self,
        phi_param: torch.Tensor,
        layout: List[ParameterSlice],
        k: int,
        use_abs: bool,
    ) -> torch.Tensor:
        # fallback
        if (not self.cfg.quota_enable) or (k <= 0):
            return topk_mask(phi_param, k=k, use_abs=use_abs)

        total = int(phi_param.numel())
        k = max(1, min(int(k), total))
        mask = torch.zeros(total, dtype=torch.bool)

        # Precompute spans per layer and per module group.
        layer_to_spans: Dict[int, List[Tuple[int, int]]] = {}
        module_to_spans: Dict[str, List[Tuple[int, int]]] = {}
        for item in layout:
            s, e = int(item.start), int(item.end)
            lid = self._layer_id_from_name(item.name)
            if lid is not None:
                layer_to_spans.setdefault(lid, []).append((s, e))
            mod = self._module_group_from_name(item.name)
            module_to_spans.setdefault(mod, []).append((s, e))

        used = 0

        # A) layer quota
        layer_quota_total = int(k * max(0.0, self.cfg.quota_layer_min_ratio))
        if layer_quota_total > 0 and len(layer_to_spans) > 0:
            per_layer = max(1, layer_quota_total // len(layer_to_spans))
            for lid, spans in sorted(layer_to_spans.items(), key=lambda x: x[0]):
                if used >= k:
                    break
                idx_parts = [torch.arange(s, e, dtype=torch.long) for s, e in spans]
                idx = torch.cat(idx_parts) if idx_parts else torch.empty(0, dtype=torch.long)
                if idx.numel() == 0:
                    continue
                residual = idx[~mask[idx]]
                kk = min(per_layer, residual.numel(), k - used)
                if kk <= 0:
                    continue
                chosen_local = self._pick_topk_indices(phi_param[residual], kk, use_abs)
                chosen = residual[chosen_local]
                mask[chosen] = True
                used += int(chosen.numel())

        # B) module quota
        module_quota_total = int(k * max(0.0, self.cfg.quota_module_min_ratio))
        if module_quota_total > 0 and len(module_to_spans) > 0 and used < k:
            per_mod = max(1, module_quota_total // len(module_to_spans))
            for mod, spans in module_to_spans.items():
                if used >= k:
                    break
                idx_parts = [torch.arange(s, e, dtype=torch.long) for s, e in spans]
                idx = torch.cat(idx_parts) if idx_parts else torch.empty(0, dtype=torch.long)
                if idx.numel() == 0:
                    continue
                residual = idx[~mask[idx]]
                kk = min(per_mod, residual.numel(), k - used)
                if kk <= 0:
                    continue
                chosen_local = self._pick_topk_indices(phi_param[residual], kk, use_abs)
                chosen = residual[chosen_local]
                mask[chosen] = True
                used += int(chosen.numel())

        # C) remainder global top-k
        rem = k - used
        if rem > 0:
            residual = (~mask).nonzero(as_tuple=False).reshape(-1)
            chosen_local = self._pick_topk_indices(phi_param[residual], rem, use_abs)
            chosen = residual[chosen_local]
            mask[chosen] = True

        return mask

    def run(self) -> None:
        stage = self.cfg.stage.lower()
        if stage not in {"all", "warmup", "select", "finetune"}:
            raise ValueError(f"Unsupported stage: {self.cfg.stage}")

        if stage == "warmup":
            self.stage_warmup()
            return
        if stage == "select":
            self.stage_select()
            return
        if stage == "finetune":
            self.stage_finetune()
            return

        self.stage_warmup()
        self.stage_select()
        self.stage_finetune()
