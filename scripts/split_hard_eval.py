"""Split the hard slice into train/held-out and filter the distill set.

Prevents train==test contamination: hold out every Nth hard problem for eval,
and keep only distill rows whose PARENT problem is in the train split.

Outputs:
  - eval_hard.jsonl        : held-out hard problems (problem, reference_solution, answer)
  - distill_train.jsonl    : distill rows (correct privileged solutions) for train only
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PRIV = re.compile(r"-priv\d+$")


def parent_id(row_id: str) -> str:
    return _PRIV.sub("", str(row_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hard", required=True, help="opsd_hard_trainset.jsonl")
    ap.add_argument("--distill", required=True, help="opsd_hard_distill.jsonl")
    ap.add_argument("--holdout-every", type=int, default=8)
    ap.add_argument("--eval-out", required=True)
    ap.add_argument("--train-out", required=True)
    args = ap.parse_args()

    hard_rows = []
    with open(args.hard, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                hard_rows.append(json.loads(line))

    eval_ids = set()
    with open(args.eval_out, "w", encoding="utf-8") as fh:
        for i, row in enumerate(hard_rows):
            if i % args.holdout_every == 0:
                eval_ids.add(str(row["id"]))
                fh.write(json.dumps({
                    "id": row["id"],
                    "problem": row["problem"],
                    "reference_solution": row["reference_solution"],
                    "answer": row["answer"],
                }, ensure_ascii=False) + "\n")

    n_train = 0
    n_drop = 0
    with open(args.distill, "r", encoding="utf-8") as fin, \
         open(args.train_out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if parent_id(row["id"]) in eval_ids:
                n_drop += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_train += 1

    print(f"held-out eval problems = {len(eval_ids)} -> {args.eval_out}")
    print(f"distill train rows = {n_train} (dropped {n_drop} that leak into eval) -> {args.train_out}")


if __name__ == "__main__":
    main()
