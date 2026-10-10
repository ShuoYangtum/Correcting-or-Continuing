"""Elicitation fraction eta (doc Sec. 2.4): how much of a stronger
teacher's advantage can be recovered by pure prompt rewriting?

- eta_acc = (Acc(PSSD) - Acc(base)) / (Acc(privileged) - Acc(base))
  where Acc(privileged) is the frozen *solver*'s accuracy when y* is visible
  (eval-only upper anchor; y* is never used for OPD KL).

- eta_KL(x) = 1 - D(pi_phi(.|x) || p_T) / D(p_S(.|x) || p_T)
  with p_T the 8B OPD teacher (same prompt as the solver, no y*). Each
  divergence uses the clipped sampled-token estimator of the training reward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from pssd.answers import grade_output
from pssd.data import Problem
from pssd.modeling import PSSDEngine
from pssd.rewards import clipped_token_reward
from pssd.rewrite_gates import select_rewrite_with_retries
from pssd.templates import (
    privileged_solver_messages,
    rewriter_messages,
    solver_messages,
    strip_think_blocks,
)
from pssd.utils import get_logger, mean_or_nan

logger = get_logger(__name__)


@dataclass
class ConditionResult:
    name: str
    accuracy: float
    per_problem: List[float]
    # Optional solver-facing rewrites (filled by vLLM PSSD eval for dump/resume).
    rewrites: Optional[Dict[str, str]] = None


def _rewrite_for(engine: PSSDEngine, problem: Problem) -> str:
    """Deployment rewrite with inference gates: validate → retry → else original.

    Attempt 0 is greedy; later attempts sample so retries are not identical.
    """
    rw = engine.cfg.rewrite
    prompt_ids = engine.chat_ids(
        rewriter_messages(
            problem.problem,
            two_stage=rw.two_stage,
            style=rw.style,
            few_shot=rw.few_shot,
        ),
        enable_thinking=rw.enable_thinking,
        rewriter=True,
    )

    def generate_raw(attempt: int) -> str:
        if attempt == 0:
            ids = engine.generate(
                prompt_ids,
                n=1,
                max_new_tokens=rw.max_new_tokens,
                do_sample=False,
                use_rewriter=True,
            )[0]
        else:
            ids = engine.generate(
                prompt_ids,
                n=1,
                max_new_tokens=rw.max_new_tokens,
                temperature=rw.temperature,
                top_p=rw.top_p,
                do_sample=True,
                use_rewriter=True,
            )[0]
        return engine.decode(ids, rewriter=True)

    decision = select_rewrite_with_retries(problem.problem, engine.cfg, generate_raw)
    if decision.fell_back:
        logger.info(
            "Rewrite fallback for %s after %d attempt(s): %s",
            problem.id, decision.attempts, ",".join(decision.reasons),
        )
    return decision.text


def solver_prompt_builder(mode: str, engine: PSSDEngine) -> Callable[[Problem], List[int]]:
    """Return a function mapping a problem to solver prompt ids for a condition.

    - 'base'       : solver sees the original problem x.
    - 'pssd'       : solver sees the trained rewriter's greedy rewrite r.
    - 'privileged' : solver sees x plus the reference solution in context
                     (chat-wrapped teacher wording; upper anchor for eta_acc).
    """
    boxed = engine.cfg.rollout.append_boxed_instruction
    if mode == "base":
        return lambda p: engine.chat_ids(
            solver_messages(p.problem, append_boxed_instruction=boxed)
        )
    if mode == "pssd":
        return lambda p: engine.chat_ids(
            solver_messages(_rewrite_for(engine, p), append_boxed_instruction=boxed)
        )
    if mode == "privileged":
        return lambda p: engine.chat_ids(
            privileged_solver_messages(p.problem, p.reference_solution)
        )
    raise ValueError(f"Unknown condition: {mode!r}")


def _load_mode_progress(path: Optional[Path]) -> Dict[str, float]:
    if path is None or not path.exists():
        return {}
    done: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = row.get("id")
            acc = row.get("acc")
            if pid is not None and acc is not None:
                done[str(pid)] = float(acc)
    return done


def evaluate_condition(
    engine: PSSDEngine,
    problems: List[Problem],
    mode: str,
    n_samples: int,
    progress_path: Optional[Path] = None,
) -> ConditionResult:
    """Avg@n accuracy of the FROZEN solver under one prompting condition.

    If ``progress_path`` is set, completed problem scores are loaded and skipped,
    and each new score is appended immediately (for 48h wall-clock chaining).
    """
    build = solver_prompt_builder(mode, engine)
    grader = engine.cfg.reward.grader
    resumed = _load_mode_progress(progress_path)
    if resumed:
        logger.info("[%s] resume: %d/%d problems already scored", mode, len(resumed), len(problems))
    progress_fh = progress_path.open("a", encoding="utf-8") if progress_path is not None else None
    per_problem: List[float] = []
    try:
        for i, problem in enumerate(problems, start=1):
            if problem.answer is None:
                logger.warning("Problem %s has no gold answer; skipped in accuracy eval", problem.id)
                continue
            if problem.id in resumed:
                per_problem.append(resumed[problem.id])
                logger.info(
                    "[%s] problem %d/%d (%s): avg@%d = %.3f (cached)",
                    mode, i, len(problems), problem.id, n_samples, resumed[problem.id],
                )
                continue
            prompt_ids = build(problem)
            rollouts = engine.sample_rollouts(prompt_ids, n_samples)
            correct = []
            for r in rollouts:
                text = engine.decode(r)
                # OPSD grades the raw generation (boxed may sit after </think>).
                # Regex grader keeps the older strip_think path for pilot Acc.
                if grader != "math_verify":
                    text = strip_think_blocks(text)
                correct.append(grade_output(text, problem.answer, grader=grader))
            acc = sum(correct) / max(1, len(correct))
            per_problem.append(acc)
            if progress_fh is not None:
                progress_fh.write(json.dumps({"id": problem.id, "acc": acc}) + "\n")
                progress_fh.flush()
            logger.info("[%s] problem %d/%d (%s): avg@%d = %.3f",
                        mode, i, len(problems), problem.id, n_samples, acc)
    finally:
        if progress_fh is not None:
            progress_fh.close()
    return ConditionResult(mode, mean_or_nan(per_problem), per_problem)


def eta_acc(acc_base: float, acc_pssd: float, acc_privileged: float) -> float:
    """Accuracy-based elicitation fraction; NaN when the privileged gap vanishes."""
    gap = acc_privileged - acc_base
    if abs(gap) < 1e-9:
        return float("nan")
    return (acc_pssd - acc_base) / gap


def eta_kl(
    engine: PSSDEngine,
    problems: List[Problem],
    n_samples: int,
    clip_c: float,
    scoring_batch_size: int = 4,
) -> dict:
    """Sampled, clipped estimate of eta_KL averaged over problems.

    For each problem, D(policy || p_T) is estimated as the mean over the
    policy's own rollouts of mean_n clip(log p_policy - log p_T, -c, c);
    both policies share the frozen solver, differing only in the prompt.
    """
    per_problem_eta: List[float] = []
    d_base_all: List[float] = []
    d_pssd_all: List[float] = []
    boxed = engine.cfg.rollout.append_boxed_instruction
    for problem in problems:
        divergences = {}
        for mode in ("base", "pssd"):
            prompt_text = problem.problem if mode == "base" else _rewrite_for(engine, problem)
            prompt_ids = engine.chat_ids(
                solver_messages(prompt_text, append_boxed_instruction=boxed)
            )
            rollouts = [r for r in engine.sample_rollouts(prompt_ids, n_samples) if r]
            if not rollouts:
                divergences[mode] = float("nan")
                continue
            lp_solver = engine.score_target_logprobs(
                [(prompt_ids, r) for r in rollouts],
                role="solver",
                batch_size=scoring_batch_size,
            )
            teacher_prefix_ids = engine.chat_ids(
                solver_messages(prompt_text, append_boxed_instruction=boxed),
                teacher=True,
            )
            lp_teacher = engine.score_target_logprobs(
                [(teacher_prefix_ids, engine.continuation_for_teacher(r)) for r in rollouts],
                role="teacher",
                batch_size=scoring_batch_size,
            )
            samples = [
                -clipped_token_reward(lt, ls, clip_c)
                for lt, ls in zip(lp_teacher, lp_solver)
            ]
            divergences[mode] = mean_or_nan([s for s in samples if s == s])
        d_base, d_pssd = divergences.get("base"), divergences.get("pssd")
        if d_base is not None and d_pssd is not None and d_base == d_base and abs(d_base) > 1e-9:
            per_problem_eta.append(1.0 - d_pssd / d_base)
            d_base_all.append(d_base)
            d_pssd_all.append(d_pssd)
    return {
        "eta_kl": mean_or_nan(per_problem_eta),
        "d_base": mean_or_nan(d_base_all),
        "d_pssd": mean_or_nan(d_pssd_all),
        "n_problems": len(per_problem_eta),
    }
