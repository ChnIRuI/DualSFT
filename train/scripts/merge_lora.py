#!/usr/bin/env python3
from __future__ import annotations

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a base model checkpoint.")
    parser.add_argument("--base-model", required=True, help="Path to the base model checkpoint.")
    parser.add_argument("--lora-adapter", required=True, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output-dir", required=True, help="Directory where the merged model will be written.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward `trust_remote_code=True` when loading the base model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    merged = model.merge_and_unload()
    merged.save_pretrained(args.output_dir, safe_serialization=True)


if __name__ == "__main__":
    main()
