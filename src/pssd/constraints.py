"""Anti-leakage / anti-omission penalties and diagnostics (doc Sec. 3.4).

1. Structural guard: templates.assert_no_leakage (raises, not a penalty).
2. Length budget penalty: max(0, |r|/|x| - budget).
3. Added-content discriminator: frozen base judges clear injection only
   (lenient prompt; greedy, temperature 0).
4. Number coverage: penalize / invalidate rewrites that drop quantities from x
   (including labeled values inside [asy] figures).
5. Diagnostics (reported): new n-gram rate of r vs x, n-gram overlap with y*.
"""

from __future__ import annotations

import re
from typing import Set, Tuple

from pssd.modeling import PSSDEngine
from pssd.templates import discriminator_messages

_WORD_RE = re.compile(r"[A-Za-z0-9]+|[一-鿿]")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_ASY_RE = re.compile(r"\[asy\].*?\[/asy\]", flags=re.IGNORECASE | re.DOTALL)
# Asymptote label("...") / label(scale(...)*"...")
_ASY_LABEL_RE = re.compile(
    r'label\s*\(\s*(?:scale\s*\([^)]*\)\s*\*\s*)?(?:rotate\s*\([^)]*\)\s*\*\s*)?"([^"]*)"',
    flags=re.IGNORECASE,
)
_ASY_EQ_NUM_RE = re.compile(r"=\s*(\d+(?:\.\d+)?)")


def _tokens(text: str) -> list:
    return _WORD_RE.findall(text.lower())


def ngram_set(text: str, n: int = 4) -> Set[tuple]:
    toks = _tokens(text)
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def new_ngram_rate(original: str, rewrite: str, n: int = 4) -> float:
    """Fraction of rewrite n-grams absent from the original (diagnostic)."""
    rw = ngram_set(rewrite, n)
    if not rw:
        return 0.0
    orig = ngram_set(original, n)
    return len(rw - orig) / len(rw)


def reference_overlap(rewrite: str, reference_solution: str, n: int = 4) -> float:
    """Fraction of rewrite n-grams that also occur in y* (leak diagnostic)."""
    rw = ngram_set(rewrite, n)
    if not rw:
        return 0.0
    ref = ngram_set(reference_solution, n)
    return len(rw & ref) / len(rw)


def length_penalty(length_ratio: float, budget: float, weight: float) -> float:
    """Penalty = weight * max(0, |r|/|x| - budget)."""
    return weight * max(0.0, length_ratio - budget)


def _norm_num(token: str) -> str:
    """Normalize numeric spellings so 3 and 3.0 match."""
    try:
        value = float(token)
    except ValueError:
        return token
    if value == int(value):
        return str(int(value))
    return str(value)


def extract_critical_numbers(text: str) -> Set[str]:
    """Numbers that must be preserved: body text + figure labels / r=3 style eqs.

    Drawing coordinates inside Asymptote that are not in labels are ignored so
    we do not force the rewriter to copy raw plot geometry.
    """
    text = text or ""
    outside = _ASY_RE.sub(" ", text)
    nums = {_norm_num(n) for n in _NUM_RE.findall(outside)}
    for block in _ASY_RE.findall(text):
        for match in _ASY_LABEL_RE.finditer(block):
            nums.update(_norm_num(n) for n in _NUM_RE.findall(match.group(1)))
        for match in _ASY_EQ_NUM_RE.finditer(block):
            nums.add(_norm_num(match.group(1)))
    return nums


def number_coverage(original: str, rewrite: str) -> Tuple[float, float, Set[str]]:
    """Return (coverage, missing_rate, missing_normalized_numbers)."""
    need = extract_critical_numbers(original)
    if not need:
        return 1.0, 0.0, set()
    have = {_norm_num(n) for n in _NUM_RE.findall(rewrite or "")}
    missing = need - have
    missing_rate = len(missing) / len(need)
    return 1.0 - missing_rate, missing_rate, missing


def coverage_penalty(missing_rate: float, weight: float) -> float:
    """Penalty for dropping quantities from the original problem."""
    return weight * max(0.0, missing_rate)


class AddedContentDiscriminator:
    """Frozen-base judge for clear injection (lenient prompt; greedy decode)."""

    def __init__(self, engine: PSSDEngine) -> None:
        self._engine = engine

    def __call__(self, original: str, rewrite: str) -> bool:
        prompt_ids = self._engine.chat_ids(discriminator_messages(original, rewrite))
        out_ids = self._engine.generate(
            prompt_ids, n=1, max_new_tokens=8, do_sample=False, use_adapter=False
        )[0]
        verdict = self._engine.decode(out_ids).strip().lower()
        # Require an explicit leading "yes"; anything else (incl. unsure) is No.
        return bool(re.match(r"^\s*yes\b", verdict))
