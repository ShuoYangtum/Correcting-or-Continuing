#!/usr/bin/env python3
"""Greedy boxed + math_verify eval for P6 (MATH-500 / AIME)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pssd.answers import _extract_boxed, grade_output

BOXED = "Please reason step by step, and put your final answer within \\boxed{}."


def load_rows(path: str, max_problems: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            q = (rec.get("problem") or rec.get("question") or "").strip()
            gold = rec.get("answer")
            if gold is None:
                gold = rec.get("gold")
            if not q:
                continue
            rows.append({"id": rec.get("id", f"row-{len(rows)}"), "problem": q, "answer": str(gold or "")})
            if max_problems and len(rows) >= max_problems:
                break
    return rows


def generate_one(model, tok, question: str, max_new: int, device) -> str:
    q = question.replace("Return your final response within \\boxed{}.", "").strip()
    user = q if "\\boxed" in q else f"{q}\n\n{BOXED}"
    msgs = [{"role": "user", "content": user}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    gen = out[0, ids["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True)


def eval_model(name: str, model_path: str, rows: list[dict], max_new: int, out_path: Path) -> dict:
    print(f"Eval {name}: {model_path} n={len(rows)} max_new={max_new}", flush=True)
    # Native transformers 4.57 Phi3: remote modeling_phi3.py imports LossKwargs
    # which this env does not have (job 21628 ImportError).
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="eager"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=False
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n_ok = 0
    writer = out_path.open("w", encoding="utf-8")
    for i, rec in enumerate(rows, start=1):
        text = generate_one(model, tok, rec["problem"], max_new, device)
        pred = _extract_boxed(text) or ""
        ok = float(grade_output(text, rec["answer"], grader="math_verify"))
        n_ok += ok
        writer.write(
            json.dumps(
                {
                    "id": rec["id"],
                    "gold": rec["answer"],
                    "pred": pred,
                    "correct": ok,
                    "completion": text[:4000],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        writer.flush()
        if i == 1 or i % 10 == 0 or i == len(rows):
            print(
                f"  {name} {i}/{len(rows)} running_acc={n_ok/i:.4f} last={rec['id']} pred={pred!r} ok={bool(ok)}",
                flush=True,
            )
    writer.close()
    acc = n_ok / max(len(rows), 1)
    summary = {"arm": name, "model": model_path, "n": len(rows), "n_correct": int(n_ok), "acc": acc}
    print(f"  {name} DONE acc={acc:.4f} ({int(n_ok)}/{len(rows)})", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="")
    p.add_argument("--student", default="")
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--gate-min", type=float, default=0.10)
    p.add_argument("--gate-max", type=float, default=0.85)
    p.add_argument("--apply-gate", action="store_true")
    args = p.parse_args()

    rows = load_rows(args.data, args.max_problems)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    if args.base:
        summaries.append(eval_model("base", args.base, rows, args.max_new_tokens, out / "base.jsonl"))
    if args.student:
        summaries.append(eval_model("student", args.student, rows, args.max_new_tokens, out / "student.jsonl"))
    rec = {"event": "summary", "n": len(rows), "data": args.data, "arms": summaries}
    if len(summaries) == 2:
        rec["acc_base"] = summaries[0]["acc"]
        rec["acc_student"] = summaries[1]["acc"]
        rec["delta_pp"] = (summaries[1]["acc"] - summaries[0]["acc"]) * 100.0
    elif len(summaries) == 1:
        rec["acc"] = summaries[0]["acc"]
        rec["n_correct"] = summaries[0]["n_correct"]
    (out / "summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2), flush=True)

    if args.apply_gate:
        acc = summaries[0]["acc"]
        ok = args.gate_min <= acc <= args.gate_max
        gate = {"acc": acc, "gate_min": args.gate_min, "gate_max": args.gate_max, "pass": ok}
        (out / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        print("GATE", gate, flush=True)
        if not ok:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
