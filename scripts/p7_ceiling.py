#!/usr/bin/env python3
"""P7 step 1: elicitation ceiling (eta) for Phi-4-mini-instruct on a MATH band.

Same question as E0.5 on Qwen3-1.7B (`scripts/run_e05_difficulty.py`): on problems
the frozen student does NOT reliably solve, does showing the REAL worked solution
raise accuracy? If yes there is headroom a distillation method could target, and a
negative OPD result on that subset is informative rather than vacuous.

Methodology is reused deliberately: same `solver_messages` /
`privileged_solver_messages` prompts, same `math_verify` grader, same Avg@N,
same base-pass-rate buckets, same "privileged only for base < threshold" trick.
Two things differ from `run_e05_difficulty.py` and both are forced by Phi:

  * It drives vLLM directly instead of ICPSEngine. ICPSEngine's vLLM path pins
    `stop=["<|im_end|>"]`, which is Qwen's turn terminator; Phi-4-mini ends turns
    with `<|end|>` (id 200020), so that path would never stop early.
  * No `enable_thinking`: Phi has no thinking mode.

y* is diagnostic only here. It is never used by any training objective.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import _bootstrap  # noqa: F401

from pssd.answers import grade_output
from pssd.data import Problem, load_problems
from pssd.templates import privileged_solver_messages, solver_messages
from pssd.utils import mean_or_nan

BUCKETS = [
    "0.00 (base always wrong)",
    "(0,0.5)",
    "[0.5,1.0)",
    "1.00 (base always right)",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="P7 eta ceiling for Phi-4-mini")
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument(
        "--priv-threshold",
        type=float,
        default=1.0,
        help="Run privileged only for problems with base acc < this",
    )
    # Hard-subset carve. "Hard" = frozen student is unreliable AND the reference
    # solution demonstrably helps, i.e. there is headroom to target.
    p.add_argument("--hard-base-max", type=float, default=0.5)
    p.add_argument("--hard-min-gap", type=float, default=0.0)
    p.add_argument("--holdout-every", type=int, default=8)
    return p.parse_args()


def bucket_of(acc: float) -> str:
    if acc <= 0.0:
        return BUCKETS[0]
    if acc < 0.5:
        return BUCKETS[1]
    if acc < 1.0:
        return BUCKETS[2]
    return BUCKETS[3]


class Sampler:
    """vLLM wrapper that renders the pssd prompts and respects Phi's stop ids."""

    def __init__(self, args: argparse.Namespace) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        self.llm = LLM(
            model=args.model,
            trust_remote_code=False,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            enforce_eager=False,
        )
        self.stop_token_ids = self._resolve_stop_ids()
        print(f"[p7] stop_token_ids={self.stop_token_ids}", flush=True)

    def _resolve_stop_ids(self) -> List[int]:
        ids: List[int] = []
        try:
            from transformers import GenerationConfig

            gen = GenerationConfig.from_pretrained(self.args.model)
            eos = gen.eos_token_id
            if isinstance(eos, int):
                ids = [eos]
            elif eos:
                ids = list(eos)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"[p7] generation_config unavailable ({exc})", flush=True)
        if self.tok.eos_token_id is not None and self.tok.eos_token_id not in ids:
            ids.append(self.tok.eos_token_id)
        return ids

    def _render(self, messages) -> str:
        return self.tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _params(self, n: int):
        from vllm import SamplingParams

        return SamplingParams(
            n=n,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=-1 if self.args.top_k <= 0 else self.args.top_k,
            min_p=0.0,
            presence_penalty=0.0,
            max_tokens=self.args.max_new_tokens,
            stop_token_ids=self.stop_token_ids or None,
        )

    def _accuracy(self, prompts: Sequence[str], golds: Sequence[str], n: int) -> List[float]:
        if not prompts:
            return []
        outputs = self.llm.generate(list(prompts), self._params(n), use_tqdm=True)
        accs: List[float] = []
        for out, gold in zip(outputs, golds):
            scores = [
                float(grade_output(o.text, gold, grader="math_verify")) for o in out.outputs
            ]
            accs.append(mean_or_nan(scores))
        return accs

    def base(self, problems: Sequence[Problem], n: int) -> List[float]:
        prompts = [
            self._render(solver_messages(p.problem, append_boxed_instruction=True))
            for p in problems
        ]
        return self._accuracy(prompts, [p.answer for p in problems], n)

    def privileged(self, problems: Sequence[Problem], n: int) -> List[float]:
        prompts = [
            self._render(privileged_solver_messages(p.problem, p.reference_solution))
            for p in problems
        ]
        return self._accuracy(prompts, [p.answer for p in problems], n)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    problems = [p for p in load_problems(args.data, args.max_problems) if p.answer is not None]
    n = int(args.n_samples)
    print(f"[p7] {len(problems)} problems | Avg@{n} | max_new={args.max_new_tokens}", flush=True)

    sampler = Sampler(args)

    print("[p7] === base arm ===", flush=True)
    acc_base = sampler.base(problems, n)

    hard_idx = [i for i, a in enumerate(acc_base) if a < args.priv_threshold]
    print(
        f"[p7] base mean={mean_or_nan(acc_base):.4f} | "
        f"{len(hard_idx)}/{len(problems)} below {args.priv_threshold} -> privileged",
        flush=True,
    )

    acc_priv = list(acc_base)  # saturated problems have gap 0 by construction
    if hard_idx:
        print("[p7] === privileged arm ===", flush=True)
        sub = sampler.privileged([problems[i] for i in hard_idx], n)
        for k, i in enumerate(hard_idx):
            acc_priv[i] = sub[k]

    per_path = out_dir / "p7_ceiling_per_problem.jsonl"
    with per_path.open("w", encoding="utf-8") as fh:
        for i, prob in enumerate(problems):
            fh.write(
                json.dumps(
                    {
                        "id": prob.id,
                        "acc_base": acc_base[i],
                        "acc_privileged": acc_priv[i],
                        "gap": acc_priv[i] - acc_base[i],
                        "privileged_run": i in set(hard_idx),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    buckets: Dict[str, Dict[str, Any]] = {}
    for i in range(len(problems)):
        d = buckets.setdefault(bucket_of(acc_base[i]), {"n": 0, "base": [], "priv": [], "gap": []})
        d["n"] += 1
        d["base"].append(acc_base[i])
        d["priv"].append(acc_priv[i])
        d["gap"].append(acc_priv[i] - acc_base[i])
    bucket_report = {
        b: {
            "n": d["n"],
            "base_mean": mean_or_nan(d["base"]),
            "priv_mean": mean_or_nan(d["priv"]),
            "gap_pp": mean_or_nan(d["gap"]) * 100.0,
        }
        for b, d in buckets.items()
    }

    nonsat = [i for i in range(len(problems)) if acc_base[i] < 1.0]
    ceiling_nonsat_pp = (
        mean_or_nan([acc_priv[i] - acc_base[i] for i in nonsat]) * 100.0 if nonsat else float("nan")
    )

    # ---- carve the hard subset and split it train/eval ----
    hard = [
        i
        for i in range(len(problems))
        if acc_base[i] <= args.hard_base_max
        and (acc_priv[i] - acc_base[i]) > args.hard_min_gap
    ]
    eval_sel = hard[:: args.holdout_every]
    train_sel = [i for i in hard if i not in set(eval_sel)]

    def dump(path: Path, idx: Sequence[int]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for i in idx:
                p = problems[i]
                fh.write(
                    json.dumps(
                        {
                            "id": p.id,
                            "problem": p.problem,
                            "reference_solution": p.reference_solution,
                            "answer": p.answer,
                            "acc_base": acc_base[i],
                            "acc_privileged": acc_priv[i],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    data_dir = Path(os.environ.get("PSSD_DATA_DIR", str(out_dir)))
    train_path = data_dir / "p7_phi_hard_train.jsonl"
    eval_path = data_dir / "p7_phi_hard_eval.jsonl"
    dump(train_path, train_sel)
    dump(eval_path, eval_sel)

    hard_gap_pp = (
        mean_or_nan([acc_priv[i] - acc_base[i] for i in hard]) * 100.0 if hard else float("nan")
    )
    summary = {
        "event": "p7_ceiling_summary",
        "model": args.model,
        "data": args.data,
        "n_problems": len(problems),
        "n_samples": n,
        "grader": "math_verify",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "priv_threshold": args.priv_threshold,
        "avg_acc_base": mean_or_nan(acc_base),
        "avg_acc_privileged": mean_or_nan(acc_priv),
        "n_nonsaturated": len(nonsat),
        "ceiling_gap_nonsaturated_pp": ceiling_nonsat_pp,
        "buckets": bucket_report,
        "hard_subset": {
            "rule": f"acc_base <= {args.hard_base_max} and gap > {args.hard_min_gap}",
            "n_hard": len(hard),
            "hard_base_mean": mean_or_nan([acc_base[i] for i in hard]) if hard else float("nan"),
            "hard_priv_mean": mean_or_nan([acc_priv[i] for i in hard]) if hard else float("nan"),
            "hard_gap_pp": hard_gap_pp,
            "n_train": len(train_sel),
            "n_eval": len(eval_sel),
            "train_path": str(train_path),
            "eval_path": str(eval_path),
        },
    }
    (out_dir / "p7_ceiling_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 70, flush=True)
    print(f"P7 eta ceiling | {args.model} | n={len(problems)} Avg@{n}", flush=True)
    print(f"  base mean              : {summary['avg_acc_base']:.4f}", flush=True)
    print(f"  non-saturated problems : {len(nonsat)}", flush=True)
    print(f"  ceiling gap (base<1)   : {ceiling_nonsat_pp:+.2f}pp", flush=True)
    for b in BUCKETS:
        if b in bucket_report:
            r = bucket_report[b]
            print(
                f"  bucket {b:<26} n={r['n']:<4d} base={r['base_mean']:.3f} "
                f"priv={r['priv_mean']:.3f} gap={r['gap_pp']:+.2f}pp",
                flush=True,
            )
    hs = summary["hard_subset"]
    print(
        f"  HARD subset: n={hs['n_hard']} (train {hs['n_train']} / eval {hs['n_eval']}) "
        f"base={hs['hard_base_mean']:.3f} priv={hs['hard_priv_mean']:.3f} gap={hs['hard_gap_pp']:+.2f}pp",
        flush=True,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
