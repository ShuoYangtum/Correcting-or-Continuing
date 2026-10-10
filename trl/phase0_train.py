#!/usr/bin/env python3
"""Phase 0 smoke: TRL on-policy distillation on GSM8K.

Student: Qwen2.5-0.5B-Instruct
Teacher: Qwen2.5-1.5B-Instruct
Generation cap: 8k. Reverse KL. Prompts only (never y*).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import DistillationConfig, DistillationTrainer


def load_prompt_dataset(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("prompt"):
                rows.append({"id": rec.get("id"), "prompt": rec["prompt"]})
                continue
            q = (rec.get("question") or rec.get("problem") or "").strip()
            if q:
                rows.append(
                    {
                        "id": rec.get("id"),
                        "prompt": [
                            {
                                "role": "user",
                                "content": q + "\n\nLet's think step by step and put the final numeric answer after ####.",
                            }
                        ],
                    }
                )
    if not rows:
        raise SystemExit(f"No usable prompts in {path}")
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--run-name", default="phase0_gsm8k_qwen25_0p5_from_1p5")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = load_prompt_dataset(args.train_data)
    print(f"Train prompts: {len(train_ds)} from {args.train_data}", flush=True)
    print(f"Student: {args.student}", flush=True)
    print(f"Teacher: {args.teacher}", flush=True)
    print(
        f"Phase0 OPD: reverse_kl={args.beta == 1.0} gen={args.max_completion_length} "
        f"epochs={args.num_train_epochs} max_steps={args.max_steps} "
        f"eff_batch={args.batch_size * args.grad_accum}",
        flush=True,
    )

    config = DistillationConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=6,
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        report_to=["wandb"],
        run_name=args.run_name,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        beta=args.beta,
        use_vllm=False,
        disable_dropout=True,
        trust_remote_code=True,
        log_completions=False,
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        teacher_model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
    )

    trainer = DistillationTrainer(
        model=args.student,
        teacher_model=args.teacher,
        args=config,
        train_dataset=train_ds,
    )
    trainer.train()
    final_dir = Path(args.output_dir) / "student_final"
    trainer.save_model(str(final_dir))
    print(f"OPD training finished -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
