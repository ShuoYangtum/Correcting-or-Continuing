"""Final-answer extraction and matching for verifiable (math-style) tasks.

Two graders are supported:

- ``regex`` (default): lightweight boxed / #### / answer-is / last-number matching.
  Used for training pilots so the pipeline stays dependency-light and fast.
- ``math_verify``: OPSD-aligned grading (``siyan-zhao/OPSD`` ``eval/evaluate_math.py``):
  extract the last ``\\boxed{...}``, then ``math_verify.parse`` + ``verify``,
  with a normalized string fallback.
"""

from __future__ import annotations

import re
from typing import Optional

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")
_HASH_ANSWER_RE = re.compile(r"####\s*(.+)")
_ANSWER_IS_RE = re.compile(r"(?:final answer|answer)\s*(?:is|:)\s*(.+)", re.IGNORECASE)

_VALID_GRADERS = frozenset({"regex", "math_verify"})


def _extract_boxed(text: str) -> Optional[str]:
    """Extract the content of the LAST \\boxed{...} with brace matching.

    Matches OPSD ``extract_boxed_answer`` (search ``\\boxed``, then brace match).
    """
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = idx
    num_left_braces = 0
    right_brace_idx = None
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        elif text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                right_brace_idx = i
                break
        i += 1
    if right_brace_idx is None:
        return None
    boxed_str = text[idx : right_brace_idx + 1]
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[len("\\boxed{") : -1].strip()
    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the final answer string from a model output or reference."""
    if not text:
        return None
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    hash_matches = _HASH_ANSWER_RE.findall(text)
    if hash_matches:
        return hash_matches[-1].strip()
    answer_matches = _ANSWER_IS_RE.findall(text)
    if answer_matches:
        candidate = answer_matches[-1].strip()
        nums = _NUMBER_RE.findall(candidate)
        if nums:
            return nums[0]
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        if lines and any(ch.isalnum() for ch in lines[0]):
            return lines[0].rstrip(".")
        # Matched "answer is/:" but captured nothing useful — fall through.
    nums = _NUMBER_RE.findall(text)
    return nums[-1] if nums else None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string: strip wrappers, commas, trailing periods."""
    a = answer.strip().strip("$").strip()
    a = a.replace(",", "").rstrip(".")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\s+", " ", a)
    return a


def _to_float(value: str) -> Optional[float]:
    try:
        if "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(predicted: Optional[str], gold: Optional[str], tol: float = 1e-6) -> bool:
    """Compare answers numerically when possible, otherwise as normalized strings."""
    if predicted is None or gold is None:
        return False
    p, g = normalize_answer(predicted), normalize_answer(gold)
    if p == g:
        return True
    pf, gf = _to_float(p), _to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) <= tol * max(1.0, abs(gf))
    return False


def grade_answer_math_verify(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """OPSD ``grade_answer``: math_verify on extracted answers + string fallback."""
    if predicted is None or ground_truth is None:
        return False
    try:
        from math_verify import parse, verify

        pred = predicted if "$" in predicted else f"${predicted}$"
        gt = ground_truth if "$" in str(ground_truth) else f"${ground_truth}$"
        pred_parsed = parse(pred, fallback_mode="no_fallback")
        gt_parsed = parse(gt, fallback_mode="no_fallback")
        return bool(verify(gt_parsed, pred_parsed, timeout_seconds=5))
    except Exception:
        pred_norm = str(predicted).replace("$", "").replace(" ", "").lower().strip()
        gt_norm = str(ground_truth).replace("$", "").replace(" ", "").lower().strip()
        return pred_norm == gt_norm


def grade_output(
    model_output: str,
    gold_answer: Optional[str],
    grader: str = "regex",
) -> bool:
    """Grade a full model output against the gold final answer.

    ``grader='math_verify'`` follows OPSD: only the last ``\\boxed{...}`` counts,
    then math-verify equivalence (string fallback on parse/verify failure).
    """
    if grader not in _VALID_GRADERS:
        raise ValueError(f"Unknown grader {grader!r}; expected one of {sorted(_VALID_GRADERS)}")
    if grader == "math_verify":
        return grade_answer_math_verify(_extract_boxed(model_output), gold_answer)
    return answers_match(extract_final_answer(model_output), gold_answer)
