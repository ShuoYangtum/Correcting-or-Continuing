"""Build the E0 'MATH-olympiad band' eval file.

The AIME24/25 eval sets carry only the final answer as `reference_solution`
(e.g. "The final answer is 204."), so the privileged anchor there degenerates
into answer-copying. To measure a *meaningful* elicitation ceiling
(Acc(privileged) with a real worked solution vs Acc(base)), we carve a random,
gradeable band out of opsd_math_30k, which DOES carry full reference solutions.

Selection (one reservoir-sampling pass, seed=42):
  - `problem` and `reference_solution` present,
  - reference length in [500, 6000] chars (real CoT, fits prompt budget),
  - problem length <= 1500 chars,
  - answer present, <= 30 chars, gradeable (has a digit or latex frac/sqrt),
  - drop bare multiple-choice letters (A-E).
"""

from __future__ import annotations

import argparse
import json
import random
import re

_MC = re.compile(r"^[A-E]$")
_GRADEABLE = re.compile(r"[0-9]|\\frac|\\sqrt|\\pi")


def gradeable(ans: str) -> bool:
    a = (ans or "").strip()
    if not a or len(a) > 30:
        return False
    if _MC.match(a):
        return False
    return bool(_GRADEABLE.search(a))


def keep(rec: dict) -> bool:
    prob = str(rec.get("problem", "")).strip()
    ref = str(rec.get("reference_solution", "")).strip()
    ans = rec.get("answer")
    if not prob or not ref:
        return False
    if not (500 <= len(ref) <= 6000):
        return False
    if len(prob) > 1500:
        return False
    if ans is None:
        return False
    return gradeable(str(ans))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    reservoir: list = []
    n_seen = 0
    n_kept = 0
    with open(args.src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not keep(rec):
                continue
            n_kept += 1
            item = {
                "id": f"mathband-{n_kept:04d}",
                "problem": str(rec["problem"]).strip(),
                "reference_solution": str(rec["reference_solution"]).strip(),
                "answer": str(rec["answer"]).strip(),
                "source": rec.get("source", "opsd_math_30k"),
            }
            # Reservoir sampling for a representative random sample.
            if len(reservoir) < args.k:
                reservoir.append(item)
            else:
                j = rng.randint(0, n_kept - 1)
                if j < args.k:
                    reservoir[j] = item
            n_seen += 1

    # Reassign stable sequential ids after sampling.
    for i, item in enumerate(reservoir, start=1):
        item["id"] = f"mathband-{i:04d}"

    with open(args.dst, "w", encoding="utf-8") as fh:
        for item in reservoir:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"scanned candidates={n_seen} wrote={len(reservoir)} -> {args.dst}")
    if reservoir:
        srcs: dict = {}
        rlens = []
        for it in reservoir:
            srcs[it["source"]] = srcs.get(it["source"], 0) + 1
            rlens.append(len(it["reference_solution"]))
        print("source_counts=", dict(sorted(srcs.items(), key=lambda x: -x[1])))
        print(f"ref_len min/mean/max = {min(rlens)}/{sum(rlens)//len(rlens)}/{max(rlens)}")


if __name__ == "__main__":
    main()
