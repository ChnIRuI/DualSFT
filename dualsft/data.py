from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from .config import DualSFTConfig


class IndexedTextDataset(Dataset):
    def __init__(self, examples: Sequence[Dict[str, Any]], indices: Sequence[int] | None = None):
        self.examples: List[Dict[str, Any]] = []
        for ex in examples:
            text = str(ex["text"])
            prompt_text = ex.get("prompt_text")
            if prompt_text is not None:
                prompt_text = str(prompt_text)
            self.examples.append({"text": text, "prompt_text": prompt_text})

        if indices is None:
            indices = list(range(len(self.examples)))
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        return {
            "text": ex["text"],
            "prompt_text": ex.get("prompt_text"),
            "index": self.indices[idx],
        }

    def subset(self, local_indices: Sequence[int]) -> "IndexedTextDataset":
        sub_texts = [self.examples[i] for i in local_indices]
        sub_orig_indices = [self.indices[i] for i in local_indices]
        return IndexedTextDataset(sub_texts, sub_orig_indices)


class CausalLMCollator:
    def __init__(self, tokenizer, max_length: int, mask_prompt_loss: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt_loss = mask_prompt_loss

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["text"] for f in features]
        prompt_texts = [f.get("prompt_text") for f in features]
        indices = [f["index"] for f in features]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        if self.mask_prompt_loss and any(p is not None for p in prompt_texts):
            prompt_inputs = [p if p is not None else "" for p in prompt_texts]
            prompt_enc = self.tokenizer(
                prompt_inputs,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
            )
            prompt_lens = [len(x) for x in prompt_enc["input_ids"]]
            bos_id = self.tokenizer.bos_token_id

            for i, prompt_text in enumerate(prompt_texts):
                if prompt_text is None:
                    continue

                seq_len = int(enc["attention_mask"][i].sum().item())
                if seq_len <= 0:
                    continue

                bos_offset = 0
                if bos_id is not None and int(enc["input_ids"][i, 0].item()) == int(bos_id):
                    bos_offset = 1

                mask_upto = min(seq_len, int(prompt_lens[i]) + bos_offset)
                if mask_upto > 0:
                    labels[i, :mask_upto] = -100

                # Avoid all-ignored labels after truncation (can yield NaN loss in HF CausalLM).
                if not bool((labels[i, :seq_len] != -100).any().item()):
                    labels[i, seq_len - 1] = enc["input_ids"][i, seq_len - 1]

        enc["labels"] = labels
        enc["indices"] = torch.tensor(indices, dtype=torch.long)
        return enc


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _read_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "data" in data:
        data = data["data"]
    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain a list of records: {path}")
    return data


def load_records(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".jsonl":
        return _read_jsonl(p)
    if p.suffix.lower() == ".json":
        try:
            return _read_json(p)
        except json.JSONDecodeError:
            # Some datasets use `.json` extension but store one JSON object per line.
            return _read_jsonl(p)
    raise ValueError(f"Unsupported file type: {path}")


def record_to_example(record: Dict[str, Any], cfg: DualSFTConfig) -> Dict[str, Any]:
    if cfg.text_field in record:
        return {"text": str(record[cfg.text_field]), "prompt_text": None}
    if cfg.prompt_field in record and cfg.response_field in record:
        prompt = str(record[cfg.prompt_field])
        response = str(record[cfg.response_field])
        prefix = f"{prompt}{cfg.prompt_response_separator}"
        return {
            "text": f"{prefix}{response}",
            "prompt_text": prefix,
        }
    keys = ", ".join(record.keys())
    raise KeyError(
        f"Record missing text fields. expected '{cfg.text_field}' or "
        f"('{cfg.prompt_field}', '{cfg.response_field}'); got keys: {keys}"
    )


def load_text_dataset(path: str, cfg: DualSFTConfig) -> IndexedTextDataset:
    records = load_records(path)
    texts = [record_to_example(r, cfg) for r in records]
    return IndexedTextDataset(texts)


def save_records_jsonl(records: Sequence[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def make_dataloader(
    dataset: Dataset,
    tokenizer,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int = 0,
    prefetch_factor: int = 2,
    mask_prompt_loss: bool = False,
) -> DataLoader:
    collator = CausalLMCollator(
        tokenizer,
        max_length=max_length,
        mask_prompt_loss=mask_prompt_loss,
    )
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
        "num_workers": max(0, int(num_workers)),
    }
    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**kwargs)
