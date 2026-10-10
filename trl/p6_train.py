#!/usr/bin/env python3
"""P6: TRL reverse-KL OPD on a non-Qwen pair that fits 1xH200.

Default pair (Llama gated, no HF token on this account):
  student microsoft/Phi-4-mini-instruct
  teacher microsoft/Phi-4-mini-reasoning
Prompts only. Never y*. No enable_thinking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
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
            if rec.get("prompt"):
                rows.append({"id": rec.get("id"), "prompt": rec["prompt"]})
                continue
            q = (rec.get("problem") or rec.get("question") or "").strip()
            if not q:
                continue
            content = q if "\\boxed" in q else f"{q}\n\n{BOXED}"
            rows.append({"id": rec.get("id"), "prompt": [{"role": "user", "content": content}]})
    if not rows:
        raise SystemExit(f"No usable prompts in {path}")
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--student", required=True)
    p.add_argument("--teacher", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--max-completion-length", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--run-name", default="p6_phi4mini_instruct_from_reasoning")
    # TRL stops generation and masks completions on tokenizer.eos_token_id only.
    # Phi-4-mini ends a turn with <|end|> (200020) but its eos_token is
    # <|endoftext|> (199999), so without this rollouts run past the answer to the cap.
    p.add_argument("--eos-token", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_ds = load_prompt_dataset(args.train_data)
    print(f"Train prompts: {len(train_ds)} from {args.train_data}", flush=True)
    print(f"Student: {args.student}", flush=True)
    print(f"Teacher: {args.teacher}", flush=True)
    print(
        f"P6 OPD: reverse_kl={args.beta == 1.0} gen={args.max_completion_length} "
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
        warmup_steps=10,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=4,
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
        trust_remote_code=False,
        log_completions=False,
        model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "device_map": {"": 0},
        },
        teacher_model_init_kwargs={
            "dtype": "bfloat16",
            "attn_implementation": "eager",
            "device_map": {"": 0},
        },
    )

    tok = AutoTokenizer.from_pretrained(args.student, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    if args.eos_token:
        tok.eos_token = args.eos_token
    print(f"eos_token={tok.eos_token!r} ({tok.eos_token_id}) pad={tok.pad_token_id}", flush=True)

    trainer = DistillationTrainer(
        model=args.student,
        teacher_model=args.teacher,
        args=config,
        train_dataset=train_ds,
        processing_class=tok,
    )
    trainer.train()
    final_dir = Path(args.output_dir) / "student_final"
    trainer.save_model(str(final_dir))
    print(f"OPD training finished -> {final_dir}", flush=True)


if __name__ == "__main__":
    main()
