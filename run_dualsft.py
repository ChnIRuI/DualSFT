from __future__ import annotations

import argparse

# Required imports for the CLI entry point.
from dualsft.config import DualSFTConfig
from dualsft.algorithm import DualSFTRunner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("DualSFT runner")

    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--validation_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--stage", type=str, default="all", choices=["all", "warmup", "select", "finetune"])

    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--prompt_field", type=str, default="prompt")
    p.add_argument("--response_field", type=str, default="response")
    p.add_argument("--prompt_response_separator", type=str, default="\n")
    p.add_argument(
        "--train_on_prompt",
        action="store_true",
        help="When enabled, compute loss on both prompt and response tokens. "
             "By default, only response tokens contribute to loss (LlamaFactory-style).",
    )
    p.add_argument("--max_length", type=int, default=1024)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--teacher_device", type=str, default="cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--final_learning_rate", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--final_epochs", type=int, default=3)
    p.add_argument("--warmup_batch_size", type=int, default=2)
    p.add_argument("--score_batch_size", type=int, default=2)
    p.add_argument("--final_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=2)
    p.add_argument("--warmup_grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps for warmup stage.")
    p.add_argument("--final_grad_accum_steps", type=int, default=1,
                   help="Gradient accumulation steps for finetune stage.")

    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--anchor_ratio", type=float, default=0.05)
    p.add_argument("--score_pool_ratio", type=float, default=1.0)

    p.add_argument("--data_budget_ratio", type=float, default=0.10)
    p.add_argument("--param_budget_ratio", type=float, default=0.05)
    p.add_argument("--data_budget", type=int, default=None)
    p.add_argument("--param_budget", type=int, default=None)

    p.add_argument("--lambda_new", type=float, default=1.0)
    p.add_argument("--lambda_prior", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--confidence_mode", type=str, default="max_prob", choices=["uniform", "max_prob", "entropy_inv"])
    p.add_argument("--gradient_normalize", type=str, default="mean", choices=["mean", "sum"])

    p.add_argument("--disable_second_order", action="store_true")
    p.add_argument("--score_method", type=str, default="ghost_linear", choices=["ghost_linear", "exact"])
    p.add_argument("--score_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--ghost_fallback_exact", action="store_true")
    p.add_argument(
        "--data_rerank_topm",
        type=int,
        default=0,
        help="If > 0 and score_method=ghost_linear, exact-score the top-M ghost-ranked samples before final selection.",
    )
    p.add_argument("--topk_use_abs", action="store_true")
    p.add_argument("--save_param_scores", action="store_true")
    p.add_argument("--save_selected_data", action="store_true", default=True)
    p.add_argument("--no_save_selected_data", action="store_false", dest="save_selected_data")
    p.add_argument("--save_data_scores", action="store_true")
    p.add_argument("--save_full_intermediates", action="store_true")

    p.add_argument("--quota_layer_min_ratio", type=float, default=0.0,
                   help="Minimum fraction of global param budget allocated per layer.")
    p.add_argument("--quota_module_min_ratio", type=float, default=0.0,
                   help="Minimum fraction of global param budget allocated per module group (q,k,v,o,mlp).")
    p.add_argument("--quota_enable", action="store_true",
                   help="Enable layer/module quota in parameter selection.")
    p.add_argument("--selection_dir", type=str, default=None,
                   help="Directory containing stage-2 selection artifacts (if different from output_dir).")
    p.add_argument(
        "--final_model_tag",
        type=str,
        default=None,
        help="Optional tag appended to final model directory name for easier experiment tracking.",
    )

    return p


def main() -> None:
    args = build_parser().parse_args()

    cfg = DualSFTConfig(
        model_name_or_path=args.model_name_or_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        stage=args.stage,
        text_field=args.text_field,
        prompt_field=args.prompt_field,
        response_field=args.response_field,
        prompt_response_separator=args.prompt_response_separator,
        train_on_prompt=args.train_on_prompt,
        max_length=args.max_length,
        seed=args.seed,
        device=args.device,
        teacher_device=args.teacher_device,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        learning_rate=args.learning_rate,
        final_learning_rate=args.final_learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        final_epochs=args.final_epochs,
        warmup_batch_size=args.warmup_batch_size,
        score_batch_size=args.score_batch_size,
        final_batch_size=args.final_batch_size,
        eval_batch_size=args.eval_batch_size,
        warmup_grad_accum_steps=max(1, int(args.warmup_grad_accum_steps)),
        final_grad_accum_steps=max(1, int(args.final_grad_accum_steps)),
        warmup_ratio=args.warmup_ratio,
        anchor_ratio=args.anchor_ratio,
        score_pool_ratio=args.score_pool_ratio,
        data_budget_ratio=args.data_budget_ratio,
        param_budget_ratio=args.param_budget_ratio,
        data_budget=args.data_budget,
        param_budget=args.param_budget,
        lambda_new=args.lambda_new,
        lambda_prior=args.lambda_prior,
        tau=args.tau,
        confidence_mode=args.confidence_mode,
        gradient_normalize=args.gradient_normalize,
        use_diagonal_second_order=not args.disable_second_order,
        score_method=args.score_method,
        score_dtype=args.score_dtype,
        ghost_fallback_exact=args.ghost_fallback_exact,
        data_rerank_topm=max(0, int(args.data_rerank_topm)),
        topk_use_abs=args.topk_use_abs,
        save_param_scores=args.save_param_scores,
        save_selected_data=args.save_selected_data,
        save_data_scores=args.save_data_scores,
        save_full_intermediates=args.save_full_intermediates,
        bf16=args.bf16,
        fp16=args.fp16,
        quota_enable=args.quota_enable,
        quota_layer_min_ratio=args.quota_layer_min_ratio,
        quota_module_min_ratio=args.quota_module_min_ratio,
        selection_dir=args.selection_dir,
        final_model_tag=args.final_model_tag,
    )

    runner = DualSFTRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
