"""Harvest a difficulty-filtered training set for E1.

E0.5 established that elicitable headroom is concentrated on problems the base
model does NOT already solve (privileged lifts +27pp on the non-saturated slice,
vs ~0 on the 85% it already solves). Training on the full, mostly-saturated pool
washes out the signal (this is why OPD showed no gain).

This script scores the frozen INSTRUCT model's base Avg@N on a large candidate
pool (opsd_math_30k slice with REAL reference solutions), then writes only the
"elicitable" problems (lo <= base_acc <= hi, default 0..0.75) as the E1 training
set, preserving reference solutions (needed for the privileged teacher).
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

logger = get_logger("pssd.harvest")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest difficulty-filtered trainset")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--n-samples", type=int, default=4)
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--lo", type=float, default=0.0, help="keep base_acc >= lo")
    p.add_argument("--hi", type=float, default=0.75, help="keep base_acc <= hi")
    p.add_argument("--out", required=True, help="output trainset jsonl")
    p.add_argument("--per-problem", default=None, help="optional per-problem dump")
    return p.parse_args()


def _load_icps_section(raw: Dict[str, Any]) -> ICPSSettings:
    section = dict(raw.get("icps") or {})
    section.pop("train_path", None)
    known = {f.name for f in fields(ICPSSettings)}
    return ICPSSettings(**{k: v for k, v in section.items() if k in known})


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    icps = _load_icps_section(raw)
    raw.pop("icps", None)
    cfg = build_config(raw)
    set_seed(cfg.train.seed)

    data_path = args.data or cfg.data.train_path
    max_problems = args.max_problems if args.max_problems is not None else cfg.data.max_problems
    problems = [p for p in load_problems(data_path, max_problems) if p.answer is not None]
    n = int(args.n_samples)
    logger.info("Harvest: scoring base Avg@%d on %d candidates from %s", n, len(problems), data_path)

    engine = ICPSEngine(cfg, icps)
    if not icps.use_vllm_base:
        raise SystemExit("harvest requires icps.use_vllm_base=true")
    icps.vllm_max_model_len = min(
        max(icps.vllm_max_model_len, int(cfg.rollout.max_new_tokens) + 2048), 40960
    )
    engine.attach_vllm_base()

    texts = [p.problem for p in problems]
    golds = [p.answer for p in problems]
    acc_base: List[float] = engine.accuracy_base_batch(texts, golds, n)
    if engine._vllm is not None:
        engine.release_vllm_restore_hf()

    out_path = Path(args.out)
    ensure_dir(out_path.parent)
    kept = 0
    dist = {"==0": 0, "(0,0.5)": 0, "[0.5,hi]": 0, "sat==1": 0, "above_hi": 0}
    with out_path.open("w", encoding="utf-8") as fh:
        for i, prob in enumerate(problems):
            a = acc_base[i]
            if a >= 1.0:
                dist["sat==1"] += 1
                continue
            if a > args.hi:
                dist["above_hi"] += 1
                continue
            if a < args.lo:
                continue
            if a == 0.0:
                dist["==0"] += 1
            elif a < 0.5:
                dist["(0,0.5)"] += 1
            else:
                dist["[0.5,hi]"] += 1
            fh.write(json.dumps({
                "id": prob.id,
                "problem": prob.problem,
                "reference_solution": prob.reference_solution,
                "answer": prob.answer,
                "base_acc": a,
            }, ensure_ascii=False) + "\n")
            kept += 1

    if args.per_problem:
        pp = Path(args.per_problem)
        ensure_dir(pp.parent)
        with pp.open("w", encoding="utf-8") as fh:
            for i, prob in enumerate(problems):
                fh.write(json.dumps({"id": prob.id, "acc_base": acc_base[i]}, ensure_ascii=False) + "\n")

    logger.info("=" * 64)
    logger.info("Harvest done: base_mean=%.4f over %d candidates", mean_or_nan(acc_base), len(problems))
    logger.info("Kept %d problems with %.2f<=base_acc<=%.2f -> %s", kept, args.lo, args.hi, out_path)
    logger.info("Distribution among kept: %s", {k: v for k, v in dist.items() if k not in ("sat==1", "above_hi")})
    logger.info("Excluded: saturated(==1)=%d  above_hi=%d", dist["sat==1"], dist["above_hi"])


if __name__ == "__main__":
    main()
