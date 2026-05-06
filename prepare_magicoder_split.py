#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list or {'data': list}")
        return data
    except json.JSONDecodeError:
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows


def dump_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Magicoder dataset into train/val jsonl")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_size", type=int, default=1024)
    args = parser.parse_args()

    src = Path(args.input)
    train_out = Path(args.train_out)
    val_out = Path(args.val_out)

    records = load_records(src)
    n = len(records)
    if n < 2:
        raise ValueError(f"Dataset too small: {n}")

    val_size = max(1, min(int(args.val_size), n - 1))
    indices = list(range(n))
    rng = random.Random(int(args.seed))
    rng.shuffle(indices)

    val_set = set(indices[:val_size])
    train_rows = [records[i] for i in range(n) if i not in val_set]
    val_rows = [records[i] for i in range(n) if i in val_set]

    dump_jsonl(train_rows, train_out)
    dump_jsonl(val_rows, val_out)

    print(f"done: train={len(train_rows)}, val={len(val_rows)}")


if __name__ == "__main__":
    main()
