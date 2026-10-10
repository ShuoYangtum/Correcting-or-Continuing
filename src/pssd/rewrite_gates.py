"""Inference-time rewrite validation, retry, and fallback to the original problem.

Training already invalidates quantity-dropping rewrites via ``number_coverage``
when scoring. Deployment / eval previously only checked solution leaks; this
module applies the failure modes observed on AIME24 (dropped numbers, duplicated
Hint blocks, over-compressed stems, assertive false hints, length blow-up).

Length is split: ``max_length_ratio`` caps the stem only; Hint has its own
``max_hint_ratio`` / ``max_hint_chars`` budget so short MATH items are not
rejected just for appending a few strategy bullets.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from pssd.config import PSSDConfig
from pssd.constraints import number_coverage
from pssd.templates import (
    clean_rewrite_text,
    extract_rewrite_body,
    looks_like_solution_leak,
)

_HINT_HEADER_RE = re.compile(r"(?im)^\s*Hint\s*:")
_HINT_BULLET_RE = re.compile(r"(?m)^\s*-\s+\S")
_ASSERTIVE_HINT_RE = re.compile(
    r"(?i)\bmust be\b|\bnecessarily\b|\bthe only\b|\bhas to be\b"
)


@dataclass
class RewriteDecision:
    """Result of validated rewrite selection."""

    text: str
    accepted: bool
    attempts: int
    reasons: List[str] = field(default_factory=list)
    fell_back: bool = False


def rewrite_stem(text: str) -> str:
    """Problem stem before the first ``Hint:`` block (or full text if none)."""
    parts = _HINT_HEADER_RE.split(text or "", maxsplit=1)
    return (parts[0] if parts else text or "").strip()


def rewrite_hint_body(text: str) -> str:
    """Text after the first ``Hint:`` header, or empty if none."""
    parts = _HINT_HEADER_RE.split(text or "", maxsplit=1)
    if len(parts) < 2:
        return ""
    return (parts[1] or "").strip()


def _hint_char_budget(orig_n: int, rw_cfg) -> Optional[int]:
    """Allowed hint-body characters, or None if hint length is unrestricted."""
    ratio_budget = int(rw_cfg.max_hint_ratio * orig_n) if rw_cfg.max_hint_ratio > 0 else 0
    floor = int(rw_cfg.max_hint_chars)
    budget = max(floor, ratio_budget)
    return budget if budget > 0 else None


def validate_rewrite(
    original: str,
    rewrite: str,
    cfg: PSSDConfig,
) -> List[str]:
    """Return violation reason codes; empty list means accept."""
    reasons: List[str] = []
    rw = (rewrite or "").strip()
    orig = original or ""
    if not rw:
        return ["empty"]

    rw_cfg = cfg.rewrite
    if rw_cfg.invalidate_solution_leak and looks_like_solution_leak(rw):
        reasons.append("solution_leak")

    if cfg.reward.invalidate_missing_numbers:
        _cov, missing_rate, missing = number_coverage(orig, rw)
        if missing_rate > 0.0:
            reasons.append(
                "missing_numbers:" + ",".join(sorted(missing)[:8])
            )

    hint_blocks = len(_HINT_HEADER_RE.findall(rw))
    if hint_blocks > rw_cfg.max_hint_blocks:
        reasons.append(f"too_many_hint_blocks:{hint_blocks}")

    bullets = len(_HINT_BULLET_RE.findall(rw))
    if bullets > rw_cfg.max_hint_bullets:
        reasons.append(f"too_many_hint_bullets:{bullets}")

    stem = rewrite_stem(rw)
    orig_n = max(1, len(orig.strip()))
    if orig.strip() and rw_cfg.min_stem_ratio > 0:
        ratio = len(stem) / orig_n
        if ratio < rw_cfg.min_stem_ratio:
            reasons.append(f"stem_too_short:{ratio:.2f}")

    if rw_cfg.max_length_ratio > 0 and orig.strip():
        sr = len(stem) / orig_n
        if sr > rw_cfg.max_length_ratio:
            reasons.append(f"stem_too_long:{sr:.2f}")

    hint = rewrite_hint_body(rw)
    hint_budget = _hint_char_budget(orig_n, rw_cfg)
    if hint and hint_budget is not None and len(hint) > hint_budget:
        reasons.append(f"hint_too_long:{len(hint) / orig_n:.2f}")

    if rw_cfg.ban_assertive_hints and _ASSERTIVE_HINT_RE.search(rw):
        # Only flag assertive language inside / after Hint sections.
        hint_part = rw[len(stem) :]
        if hint_part and _ASSERTIVE_HINT_RE.search(hint_part):
            reasons.append("assertive_hint")

    return reasons


def clean_rewrite_candidate(raw: str, *, two_stage: bool) -> str:
    """Normalize a raw rewriter generation into solver-facing text."""
    full = clean_rewrite_text(raw or "")
    body, _ = extract_rewrite_body(full, require_tags=bool(two_stage))
    if two_stage:
        return (body or "").strip()
    # Non-two-stage: prefer tagged body when present, else cleaned full text.
    if body and body.strip():
        return body.strip()
    return full.strip()


def reason_family(reason: str) -> str:
    """Collapse ``missing_numbers:9,4`` → ``missing_numbers`` for histograms."""
    return (reason or "").split(":", 1)[0] or "unknown"


@dataclass
class RewriteFilterResult:
    """Accepted rewrites plus reject diagnostics for logging."""

    accepted: List[str]
    n_raw: int
    n_empty_or_unparsed: int
    n_rejected: int
    reject_reason_counts: Dict[str, int] = field(default_factory=dict)
    # Coarse families for quick inspection (no value suffixes).
    reject_family_counts: Dict[str, int] = field(default_factory=dict)
    # One example rewrite snippet per family (truncated).
    reject_examples: Dict[str, str] = field(default_factory=dict)

    def to_log_dict(self) -> Dict:
        return {
            "n_raw": self.n_raw,
            "n_accepted": len(self.accepted),
            "n_empty_or_unparsed": self.n_empty_or_unparsed,
            "n_rejected": self.n_rejected,
            "reject_reason_counts": dict(self.reject_reason_counts),
            "reject_family_counts": dict(self.reject_family_counts),
            "reject_examples": dict(self.reject_examples),
        }


def filter_rewrite_texts(
    original: str,
    texts: Sequence[str],
    cfg: PSSDConfig,
    *,
    two_stage: bool = False,
    example_chars: int = 180,
) -> RewriteFilterResult:
    """Filter raw rewriter generations; record why each candidate was dropped."""
    accepted: List[str] = []
    reason_counts: Counter = Counter()
    family_counts: Counter = Counter()
    examples: Dict[str, str] = {}
    n_empty = 0

    for raw in texts:
        candidate = clean_rewrite_candidate(raw, two_stage=two_stage)
        if not candidate:
            n_empty += 1
            reason_counts["empty_or_unparsed"] += 1
            family_counts["empty_or_unparsed"] += 1
            continue
        # validate_rewrite already covers solution_leak when configured.
        reasons = validate_rewrite(original, candidate, cfg)
        if not cfg.rewrite.inference_validate:
            # Legacy path: only hard leak check when validation master switch is off.
            reasons = []
            if cfg.rewrite.invalidate_solution_leak and looks_like_solution_leak(candidate):
                reasons = ["solution_leak"]
        if reasons:
            for reason in reasons:
                reason_counts[reason] += 1
                fam = reason_family(reason)
                family_counts[fam] += 1
                if fam not in examples:
                    examples[fam] = candidate[:example_chars].replace("\n", " | ")
            continue
        accepted.append(candidate)

    return RewriteFilterResult(
        accepted=accepted,
        n_raw=len(texts),
        n_empty_or_unparsed=n_empty,
        n_rejected=len(texts) - len(accepted),
        reject_reason_counts=dict(reason_counts),
        reject_family_counts=dict(family_counts),
        reject_examples=examples,
    )


def select_rewrite_with_retries(
    original: str,
    cfg: PSSDConfig,
    generate_raw: Callable[[int], str],
) -> RewriteDecision:
    """Generate / validate / retry; fall back to ``original`` if all attempts fail.

    ``generate_raw(attempt_idx)`` should return a raw model string. Attempt 0 is
    typically greedy; later attempts should sample so retries are not identical.
    """
    rw_cfg = cfg.rewrite
    max_attempts = 1 + max(0, int(rw_cfg.max_rewrite_retries))
    last_reasons: List[str] = ["empty"]

    if not rw_cfg.inference_validate:
        raw = generate_raw(0)
        body = clean_rewrite_candidate(raw, two_stage=rw_cfg.two_stage)
        if not body:
            return RewriteDecision(original, False, 1, ["empty"], fell_back=True)
        if rw_cfg.invalidate_solution_leak and looks_like_solution_leak(body):
            return RewriteDecision(original, False, 1, ["solution_leak"], fell_back=True)
        return RewriteDecision(body, True, 1, [])

    for attempt in range(max_attempts):
        raw = generate_raw(attempt)
        body = clean_rewrite_candidate(raw, two_stage=rw_cfg.two_stage)
        reasons = validate_rewrite(original, body, cfg)
        if not reasons:
            return RewriteDecision(body, True, attempt + 1, [])
        last_reasons = reasons

    return RewriteDecision(
        original, False, max_attempts, last_reasons, fell_back=True
    )


def rewrite_problems_with_retries(
    originals: Sequence[str],
    cfg: PSSDConfig,
    generate_batch: Callable[[List[int], bool], List[str]],
) -> List[RewriteDecision]:
    """Validate/retry a batch of problems.

    ``generate_batch(indices, greedy)`` must return ``len(indices)`` raw strings
    for the given problem indices into ``originals``.
    """
    n = len(originals)
    decisions: List[Optional[RewriteDecision]] = [None] * n
    pending = list(range(n))
    max_attempts = 1 + max(0, int(cfg.rewrite.max_rewrite_retries))
    last_reasons: List[List[str]] = [[] for _ in range(n)]

    if not cfg.rewrite.inference_validate:
        raws = generate_batch(pending, True)
        for idx, raw in zip(pending, raws):
            body = clean_rewrite_candidate(raw, two_stage=cfg.rewrite.two_stage)
            if not body or (
                cfg.rewrite.invalidate_solution_leak and looks_like_solution_leak(body)
            ):
                reason = ["empty"] if not body else ["solution_leak"]
                decisions[idx] = RewriteDecision(
                    originals[idx], False, 1, reason, fell_back=True
                )
            else:
                decisions[idx] = RewriteDecision(body, True, 1, [])
        return [d for d in decisions if d is not None]  # type: ignore[misc]

    for attempt in range(max_attempts):
        if not pending:
            break
        greedy = attempt == 0
        raws = generate_batch(pending, greedy)
        still: List[int] = []
        for idx, raw in zip(pending, raws):
            body = clean_rewrite_candidate(raw, two_stage=cfg.rewrite.two_stage)
            reasons = validate_rewrite(originals[idx], body, cfg)
            if not reasons:
                decisions[idx] = RewriteDecision(body, True, attempt + 1, [])
            else:
                last_reasons[idx] = reasons
                still.append(idx)
        pending = still

    for idx in pending:
        decisions[idx] = RewriteDecision(
            originals[idx],
            False,
            max_attempts,
            last_reasons[idx] or ["rejected"],
            fell_back=True,
        )
    return [decisions[i] for i in range(n)]  # type: ignore[misc]
