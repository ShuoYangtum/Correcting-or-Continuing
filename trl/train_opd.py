#!/usr/bin/env python3
"""On-policy distillation with HuggingFace TRL DistillationTrainer.

Official verl OPD needs a dedicated teacher GPU pool (student + teacher >= 2
GPUs). This cluster only allocates one H200, so we use TRL's GKD/OPD trainer
(Agarwal et al.; the same algorithm family verl documents) on one GPU.

Student and teacher see the same prompt x only. Completions are on-policy
from the student. Loss is reverse KL (beta=1.0). Never uses y*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from trl import DistillationConfig, DistillationTrainer

BOXED = "Please reason step by step, and put your final answer within \\boxed{}."


def load_prompt_dataset(path: str) -> Dataset:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            problem = (rec.get("problem") or rec.get("question") or "").strip()
            if not problem:
                continue
            content = problem if BOXED[:40] in problem else f"{problem}\n\n{BOXED}"
            rows.append(
                {
                    "id": rec.get("id"),
                    "prompt": [{"role": "user", "content": content}],
                }
            )
    if not rows:
        raise SystemExit(f"No usable prompts in {path}")
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TRL on-policy distillation")
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=800)
    p.add_argument("--max-completion-length", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--beta", type=float, default=1.0, help="1.0 = reverse KL")
    p.add_argument("--run-name", default="trl_opd_1p7b_from_4b_revkl")
    p.add_argument("--no-log-completions", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = load_prompt_dataset(args.train_data)
    print(f"Train prompts: {len(train_ds)} from {args.train_data}", flush=True)
    print(f"Student: {args.student}", flush=True)
    print(f"Teacher: {args.teacher}", flush=True)
    print(
        f"OPD: reverse_kl={args.beta==1.0} gen={args.max_completion_length} "
        f"steps={args.max_steps} eff_batch={args.batch_size * args.grad_accum}",
        flush=True,
    )

    config = DistillationConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_steps=20,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=8,
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
        log_completions=not args.no_log_completions,
        num_completions_to_print=1,
        chat_template_kwargs={"enable_thinking": True},
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
    trainer.save_model(str(Path(args.output_dir) / "student_final"))
    print("OPD training finished", flush=True)


if __name__ == "__main__":
    main()
