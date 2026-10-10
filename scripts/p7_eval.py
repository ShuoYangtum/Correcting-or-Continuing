#!/usr/bin/env python3
"""P7 step 3: score one arm (frozen base or OPD student) with vLLM.

One model per process: vLLM does not release a loaded engine cleanly inside the
same interpreter, so the caller invokes this once per arm and the paired stats
are computed afterwards on CPU (`docs/_p7_stats.py`).

Prompts and grading are the same helpers every earlier experiment used
(`solver_messages` + `math_verify`, last \\boxed{} only). Stop ids come from the
model's generation_config, which matters for Phi: its turn terminator is
<|end|> (200020) and config.json alone only lists <|endoftext|>.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import _bootstrap  # noqa: F401

from pssd.answers import grade_output
from pssd.data import load_problems
from pssd.templates import solver_messages
from pssd.utils import mean_or_nan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P7 single-arm vLLM eval")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", required=True, choices=["base", "student"])
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--greedy", action="store_true", help="n=1 temperature=0")
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="Fallback tokenizer when a checkpoint dir ships none",
    )
    return p.parse_args()


def resolve_stop_ids(model: str, tok) -> list[int]:
    ids: list[int] = []
    try:
        from transformers import GenerationConfig

        eos = GenerationConfig.from_pretrained(model).eos_token_id
        if isinstance(eos, int):
            ids = [eos]
        elif eos:
            ids = list(eos)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[p7] generation_config unavailable for {model} ({exc})", flush=True)
    if tok.eos_token_id is not None and tok.eos_token_id not in ids:
        ids.append(tok.eos_token_id)
    return ids


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = [p for p in load_problems(args.data, args.max_problems) if p.answer is not None]
    n = 1 if args.greedy else int(args.n_samples)
    print(
        f"[p7] arm={args.arm} model={args.model} n_problems={len(problems)} "
        f"Avg@{n} greedy={args.greedy} max_new={args.max_new_tokens}",
        flush=True,
    )

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok_src = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=False)
    llm = LLM(
        model=args.model,
        tokenizer=tok_src,
        trust_remote_code=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        enforce_eager=False,
    )

    stop_ids = resolve_stop_ids(tok_src, tok)
    print(f"[p7] stop_token_ids={stop_ids}", flush=True)

    if args.greedy:
        params = SamplingParams(
            n=1,
            temperature=0.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids or None,
        )
    else:
        params = SamplingParams(
            n=n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=-1 if args.top_k <= 0 else args.top_k,
            min_p=0.0,
            presence_penalty=0.0,
            max_tokens=args.max_new_tokens,
            stop_token_ids=stop_ids or None,
        )

    prompts = [
        tok.apply_chat_template(
            solver_messages(p.problem, append_boxed_instruction=True),
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in problems
    ]
    outputs = llm.generate(prompts, params, use_tqdm=True)

    per_path = out_dir / f"{args.arm}.jsonl"
    accs = []
    with per_path.open("w", encoding="utf-8") as fh:
        for prob, out in zip(problems, outputs):
            scores = [
                float(grade_output(o.text, prob.answer, grader="math_verify"))
                for o in out.outputs
            ]
            acc = mean_or_nan(scores)
            accs.append(acc)
            fh.write(
                json.dumps(
                    {
                        "id": prob.id,
                        "gold": prob.answer,
                        "acc": acc,
                        "scores": scores,
                        "n_samples": len(scores),
                        "completion_lens": [len(o.token_ids) for o in out.outputs],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "event": "p7_arm_summary",
        "arm": args.arm,
        "model": args.model,
        "data": args.data,
        "n_problems": len(problems),
        "n_samples": n,
        "greedy": args.greedy,
        "temperature": 0.0 if args.greedy else args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "grader": "math_verify",
        "acc": mean_or_nan(accs),
    }
    (out_dir / f"{args.arm}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
