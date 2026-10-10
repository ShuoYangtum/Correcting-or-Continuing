"""E0.5: difficulty-stratified elicitation ceiling.

E0 showed the training distribution is saturated for the base model
(base~90% on a random MATH-olympiad band; privileged only +3.5pp). The decisive
open question is: *on problems the base does NOT reliably solve, does showing the
REAL worked solution (privileged) substantially raise accuracy?* If yes, there is
headroom a prompt-space method can target; if no, prompt-space elicitation is
capped on this model and we must change the battlefield.

This script:
  1. loads a candidate band (opsd_math_30k slice with real reference solutions),
  2. scores base Avg@N for every problem (vLLM),
  3. scores privileged Avg@N ONLY for problems with base < 1.0 (saves compute on
     already-saturated easy problems; their gap is ~0 by construction),
  4. reports privileged - base bucketed by base pass rate.

Base/privileged both use the FROZEN model; y* is diagnostic-only (never in KL).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import _bootstrap  # noqa: F401
import yaml

from pssd.config import build_config
from pssd.data import load_problems
from pssd.icps import ICPSEngine, ICPSSettings
from pssd.utils import ensure_dir, get_logger, mean_or_nan, set_seed

logger = get_logger("pssd.e05")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E0.5 difficulty-stratified ceiling")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument(
        "--priv-threshold",
        type=float,
        default=1.0,
        help="Run privileged only for problems with base acc < this (default 1.0)",
    )
    return p.parse_args()


def _load_icps_section(raw: Dict[str, Any]) -> ICPSSettings:
    section = dict(raw.get("icps") or {})
    section.pop("train_path", None)
    known = {f.name for f in fields(ICPSSettings)}
    filtered = {k: v for k, v in section.items() if k in known}
    return ICPSSettings(**filtered)


def _bucket(acc: float) -> str:
    if acc <= 0.0:
        return "0.00 (base always wrong)"
    if acc < 0.5:
        return "(0,0.5)"
    if acc < 1.0:
        return "[0.5,1.0)"
    return "1.00 (base always right)"


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    icps = _load_icps_section(raw)
    raw.pop("icps", None)
    cfg = build_config(raw)
    set_seed(cfg.train.seed)

    data_path = args.data or cfg.data.eval_path or cfg.data.train_path
    out_dir = ensure_dir(args.output_dir or cfg.train.output_dir)
    max_problems = args.max_problems if args.max_problems is not None else cfg.data.max_problems
    problems = [p for p in load_problems(data_path, max_problems) if p.answer is not None]
    n = int(args.n_samples)

    texts = [p.problem for p in problems]
    golds = [p.answer for p in problems]
    refs = [p.reference_solution for p in problems]

    logger.info(
        "E0.5 on %d problems | Avg@%d | grader=%s | 16k | TM=%s",
        len(problems), n, cfg.reward.grader,
        "on" if cfg.model.enable_thinking else "off",
    )

    engine = ICPSEngine(cfg, icps)
    if not icps.use_vllm_base:
        raise SystemExit("E0.5 requires icps.use_vllm_base=true")
    icps.vllm_max_model_len = min(
        max(icps.vllm_max_model_len, int(cfg.rollout.max_new_tokens) + 2048), 40960
    )
    engine.attach_vllm_base()

    logger.info("Scoring base Avg@%d on %d problems", n, len(problems))
    acc_base: List[float] = engine.accuracy_base_batch(texts, golds, n)

    hard_idx = [i for i, a in enumerate(acc_base) if a < args.priv_threshold]
    logger.info(
        "Base mean=%.4f | %d/%d problems below threshold %.2f -> privileged",
        mean_or_nan(acc_base), len(hard_idx), len(problems), args.priv_threshold,
    )

    acc_priv: List[float] = list(acc_base)  # easy (>=thr) default to base (gap 0)
    if hard_idx:
        sub_priv = engine.accuracy_privileged_batch(
            [texts[i] for i in hard_idx],
            [refs[i] for i in hard_idx],
            [golds[i] for i in hard_idx],
            n,
        )
        for k, i in enumerate(hard_idx):
            acc_priv[i] = sub_priv[k]

    if engine._vllm is not None:
        engine.release_vllm_restore_hf()

    # ---- per-problem dump ----
    per_path = Path(out_dir) / "e05_per_problem.jsonl"
    with per_path.open("w", encoding="utf-8") as fh:
        for i, prob in enumerate(problems):
            fh.write(json.dumps({
                "id": prob.id,
                "source": getattr(prob, "source", None),
                "acc_base": acc_base[i],
                "acc_privileged": acc_priv[i],
                "gap": acc_priv[i] - acc_base[i],
                "privileged_run": i in set(hard_idx),
            }, ensure_ascii=False) + "\n")

    # ---- bucketed summary ----
    buckets: Dict[str, Dict[str, Any]] = {}
    for i in range(len(problems)):
        b = _bucket(acc_base[i])
        d = buckets.setdefault(b, {"n": 0, "base": [], "priv": [], "gap": []})
        d["n"] += 1
        d["base"].append(acc_base[i])
        d["priv"].append(acc_priv[i])
        d["gap"].append(acc_priv[i] - acc_base[i])

    bucket_report = {}
    for b, d in buckets.items():
        bucket_report[b] = {
            "n": d["n"],
            "base_mean": mean_or_nan(d["base"]),
            "priv_mean": mean_or_nan(d["priv"]),
            "gap_pp": mean_or_nan(d["gap"]) * 100.0,
        }

    # gap among all non-saturated problems (base<1) = the real elicitation ceiling
    nonsat = [i for i in range(len(problems)) if acc_base[i] < 1.0]
    ceiling_nonsat_pp = (
        mean_or_nan([acc_priv[i] - acc_base[i] for i in nonsat]) * 100.0 if nonsat else float("nan")
    )

    summary = {
        "event": "e05_summary",
        "data": str(data_path),
        "n_problems": len(problems),
        "n_samples": n,
        "grader": cfg.reward.grader,
        "max_new_tokens": cfg.rollout.max_new_tokens,
        "temperature": cfg.rollout.temperature,
        "top_p": cfg.rollout.top_p,
        "top_k": cfg.rollout.top_k,
        "enable_thinking": cfg.model.enable_thinking,
        "priv_threshold": args.priv_threshold,
        "avg_acc_base": mean_or_nan(acc_base),
        "n_nonsaturated": len(nonsat),
        "ceiling_gap_nonsaturated_pp": ceiling_nonsat_pp,
        "buckets": bucket_report,
    }
    summ_path = Path(out_dir) / "e05_summary.jsonl"
    with summ_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

    logger.info("=" * 64)
    logger.info("E0.5 difficulty-stratified ceiling (n=%d, Avg@%d)", len(problems), n)
    logger.info("  base mean               : %.4f", summary["avg_acc_base"])
    logger.info("  non-saturated problems  : %d", len(nonsat))
    logger.info("  ceiling gap (base<1)    : %+.2fpp", ceiling_nonsat_pp)
    for b in ["0.00 (base always wrong)", "(0,0.5)", "[0.5,1.0)", "1.00 (base always right)"]:
        if b in bucket_report:
            r = bucket_report[b]
            logger.info(
                "  bucket %-26s n=%-4d base=%.3f priv=%.3f gap=%+.2fpp",
                b, r["n"], r["base_mean"], r["priv_mean"], r["gap_pp"],
            )
    logger.info("Wrote %s", summ_path)


if __name__ == "__main__":
    main()
