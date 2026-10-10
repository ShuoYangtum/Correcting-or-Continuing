#!/usr/bin/env python3
"""Greedy GSM8K eval for Phase 0: base student vs distilled checkpoint."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

INSTR = "Let's think step by step and put the final numeric answer after ####."
NUM_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")


def extract_pred(text: str) -> str:
    m = re.findall(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if m:
        return m[-1].replace(",", "")
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        nums = NUM_RE.findall(boxed[-1].replace(",", ""))
        if nums:
            return nums[-1].replace(",", "")
    nums = NUM_RE.findall(text.replace(",", ""))
    return nums[-1] if nums else ""


def norm(x: str) -> str:
    x = (x or "").strip().replace(",", "")
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return x


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_one(model, tok, question: str, max_new: int, device) -> str:
    user = question if "####" in question else f"{question}\n\n{INSTR}"
    msgs = [{"role": "user", "content": user}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except TypeError:
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
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="sdpa"
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    n_ok = 0
    writer = out_path.open("w", encoding="utf-8")
    for i, rec in enumerate(rows, start=1):
        q = rec.get("question") or rec.get("problem") or ""
        gold = str(rec.get("answer") or "")
        text = generate_one(model, tok, q, max_new, device)
        pred = extract_pred(text)
        ok = float(norm(pred) == norm(gold))
        n_ok += ok
        writer.write(
            json.dumps(
                {
                    "id": rec.get("id", f"row-{i}"),
                    "gold": gold,
                    "pred": pred,
                    "correct": ok,
                    "completion": text[:4000],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        writer.flush()
        if i == 1 or i % 20 == 0 or i == len(rows):
            print(
                f"  {name} {i}/{len(rows)} running_acc={n_ok/i:.4f} last={rec.get('id')} pred={pred!r} gold={gold!r} ok={bool(ok)}",
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
    p.add_argument("--base", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8192)
    p.add_argument("--max-problems", type=int, default=0)
    args = p.parse_args()

    rows = load_rows(args.data)
    if args.max_problems and args.max_problems > 0:
        rows = rows[: args.max_problems]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    summaries.append(
        eval_model("base", args.base, rows, args.max_new_tokens, out / "gsm8k_base.jsonl")
    )
    summaries.append(
        eval_model("student", args.student, rows, args.max_new_tokens, out / "gsm8k_student.jsonl")
    )
    base_acc = summaries[0]["acc"]
    stu_acc = summaries[1]["acc"]
    rec = {
        "event": "summary",
        "acc_base": base_acc,
        "acc_student": stu_acc,
        "delta_pp": (stu_acc - base_acc) * 100.0,
        "n": len(rows),
        "arms": summaries,
    }
    (out / "gsm8k_summary.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("=" * 60, flush=True)
    print(f"GSM8K n={len(rows)}", flush=True)
    print(f"  base    : {base_acc:.4f}", flush=True)
    print(f"  student : {stu_acc:.4f}  (Δ={rec['delta_pp']:+.2f}pp)", flush=True)


if __name__ == "__main__":
    main()
