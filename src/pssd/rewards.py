"""Reward construction and group-relative advantages (doc Sec. 3.3).

The dense reward for a rollout y under rewrite r is the length-normalized,
pointwise-clipped OPD teacher/solver log-ratio (teacher never sees y*):

    score(y) = mean_n clip( log p_T(y_n | r, y_<n) - log p_S(y_n | r, y_<n), -c, c )

which is a sampled-token estimate of -KL(p_S(.|r) || p_T(.|r)) up to clipping.
p_T is the larger 8B teacher; p_S is the frozen 1.7B solver; both condition
on the same solver-facing prompt. Pointwise clipping prevents heavy-tailed
tokens from dominating. Per-rewrite reward averages over rollouts and
subtracts semantic-invariance penalties.

When ``relative_to_original`` is enabled, the KL term becomes the *gain*
relative to the original prompt x:

    R_KL(r) = score(r) - score(x)  ≈  D(x || p_T) - D(r || p_T)

so zero means "no better than the original wording".  Within a single
problem this does not change argmax_r score(r), but it makes cross-problem
pooling (Spearman) and "prefer original if gain ≤ 0" decisions meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np
import torch


def clipped_token_reward(lp_teacher: torch.Tensor, lp_solver: torch.Tensor, clip_c: float) -> float:
    """Mean of clip(log p_T - log p_S, -c, c) over rollout tokens (length-normalized)."""
    if lp_teacher.numel() == 0 or lp_solver.numel() == 0:
        return float("nan")
    if lp_teacher.numel() == lp_solver.numel():
        diff = torch.clamp(lp_teacher - lp_solver, min=-clip_c, max=clip_c)
        return float(diff.mean().item())
    # Cross-tokenizer OPD: clip the difference of length-normalized logprobs.
    diff = lp_teacher.mean() - lp_solver.mean()
    return float(diff.clamp(min=-clip_c, max=clip_c).item())


def mean_token_logprob(lp: torch.Tensor) -> float:
    """Length-normalized sequence log-prob: mean_n log p(y_n | prefix, y_<n)."""
    if lp.numel() == 0:
        return float("nan")
    return float(lp.mean().item())


def relative_kl_reward(score_r: float, score_x: float) -> float:
    """Local η-style gain: how much closer to the teacher r is vs original x."""
    if score_r != score_r or score_x != score_x:  # NaN
        return float("nan")
    return float(score_r - score_x)


@dataclass
class RewriteScore:
    """Full scoring breakdown for one rewrite (kept for logging/diagnostics)."""
    text: str                                  # full rewriter generation (may be two-stage tagged)
    solver_text: str = ""                      # body shown to frozen solver; empty → use text
    token_ids: List[int] = field(default_factory=list)
    kl_score: float = float("nan")       # absolute or relative (if baseline subtracted)
    kl_score_abs: float = float("nan")   # always absolute score(r), for diagnostics
    baseline_kl: float = float("nan")    # score(x), if relative mode
    acc_score: float = float("nan")      # mean grading Acc over scoring rollouts
    teacher_lik: float = float("nan")    # mean length-normalized log p_T(y)
    dense_reward: str = "kl"             # "kl" | "acc" | "acc_lik"
    acc_weight: float = 1.0
    lik_weight: float = 0.0
    length_ratio: float = float("nan")   # |r| / |x| in tokens
    length_penalty: float = 0.0
    coverage: float = 1.0                # fraction of critical numbers retained
    missing_number_rate: float = 0.0
    coverage_penalty: float = 0.0
    discriminator_flagged: bool = False
    discriminator_penalty: float = 0.0
    valid: bool = True                   # False for empty/degenerate rewrites

    def prompt_for_solver(self) -> str:
        """Text the frozen solver / coverage / length gates should see."""
        return self.solver_text if self.solver_text.strip() else self.text

    @property
    def dense_score(self) -> float:
        if self.dense_reward == "acc_lik":
            if self.acc_score != self.acc_score or self.teacher_lik != self.teacher_lik:
                return float("nan")
            return self.acc_weight * self.acc_score + self.lik_weight * self.teacher_lik
        if self.dense_reward == "acc":
            return self.acc_score
        return self.kl_score

    @property
    def reward(self) -> float:
        base = self.dense_score
        if base != base:  # NaN
            return float("nan")
        return (
            base
            - self.length_penalty
            - self.coverage_penalty
            - self.discriminator_penalty
        )


def group_advantages(rewards: Sequence[float]) -> np.ndarray:
    """GRPO-style group-relative advantages: (r - mean) / (std + eps)."""
    r = np.asarray(rewards, dtype=float)
    return (r - r.mean()) / (r.std() + 1e-8)


def group_is_informative(rewards: Sequence[float], std_eps: float) -> bool:
    """DAPO-style dynamic filtering: skip groups whose reward std is ~0."""
    r = np.asarray(rewards, dtype=float)
    return bool(np.all(np.isfinite(r)) and r.std() >= std_eps)
