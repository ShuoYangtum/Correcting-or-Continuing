"""Dual-loop PSSD trainer (doc Sec. 3.2).

Per problem x (y* is never shown to the rewriter or the OPD teacher):
  1. Sample K rewrites from the full-parameter 8B rewriter (context is x only).
  2. Sample G rollouts per rewrite from the FROZEN 1.7B solver.
  3. Score every rewrite with 8B-teacher OPD KL minus semantic-invariance penalties.
  4. Outer loop (optional): group-relative REINFORCE on rewrite tokens.
  5. Inner loop: amortize the reward-selected winner into the 8B rewriter
     via supervised NLL (imitation). Optional prefer-original gate skips
     distillation when no rewrite beats x.

Only the 8B rewriter is updated; the 1.7B solver and 8B OPD teacher stay frozen.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
from transformers import get_cosine_schedule_with_warmup

from pssd.scoring import (
    apply_penalties,
    grade_rollout,
    score_prompt_kl,
    score_rewrites,
)
from pssd.config import PSSDConfig, config_to_dict
from pssd.constraints import (
    AddedContentDiscriminator,
    new_ngram_rate,
    reference_overlap,
)
from pssd.rewrite_gates import validate_rewrite
from pssd.data import Problem, ProblemSampler, load_problems
from pssd.modeling import PSSDEngine
from pssd.rewards import (
    RewriteScore,
    group_advantages,
    group_is_informative,
)
from pssd.templates import (
    LeakageError,
    assert_no_leakage,
    extract_rewrite_body,
    inner_teacher_messages,
    looks_like_solution_leak,
    rewriter_messages,
    strip_opsd_answer_preamble,
)
from pssd.utils import JsonlWriter, ensure_dir, get_logger, mean_or_nan, set_seed

logger = get_logger(__name__)


def _init_wandb(cfg: PSSDConfig) -> Any:
    """Initialize wandb when enabled; return the module or None."""
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled but the wandb package is not installed"
        ) from exc
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=config_to_dict(cfg),
        dir=str(Path(cfg.train.output_dir) / "wandb"),
    )
    return wandb


@dataclass
class ProblemStats:
    """Per-problem statistics collected for logging."""
    reward_mean: float = float("nan")
    reward_std: float = float("nan")
    kl_mean: float = float("nan")
    acc_mean: float = float("nan")
    teacher_lik_mean: float = float("nan")
    length_ratio_mean: float = float("nan")
    new_ngram_rate: float = float("nan")
    reference_overlap: float = float("nan")
    number_coverage: float = float("nan")
    missing_number_rate: float = float("nan")
    discriminator_rate: float = float("nan")
    loss_outer: float = float("nan")
    loss_inner: float = float("nan")
    skipped: bool = False
    skip_reason: str = ""


@dataclass
class StepStats:
    problems: List[ProblemStats] = field(default_factory=list)

    def summary(self) -> dict:
        active = [p for p in self.problems if not p.skipped]
        skip_reasons = dict(Counter(p.skip_reason or "unknown" for p in self.problems if p.skipped))
        return {
            "n_problems": len(self.problems),
            "n_skipped": sum(p.skipped for p in self.problems),
            "skip_reasons": skip_reasons,
            "reward_mean": mean_or_nan([p.reward_mean for p in active]),
            "reward_std": mean_or_nan([p.reward_std for p in active]),
            "kl_mean": mean_or_nan([p.kl_mean for p in active]),
            "acc_mean": mean_or_nan([p.acc_mean for p in active]),
            "teacher_lik_mean": mean_or_nan([p.teacher_lik_mean for p in active]),
            "length_ratio": mean_or_nan([p.length_ratio_mean for p in active]),
            "new_ngram_rate": mean_or_nan([p.new_ngram_rate for p in active]),
            "reference_overlap": mean_or_nan([p.reference_overlap for p in active]),
            "number_coverage": mean_or_nan([p.number_coverage for p in active]),
            "missing_number_rate": mean_or_nan([p.missing_number_rate for p in active]),
            "discriminator_rate": mean_or_nan([p.discriminator_rate for p in active]),
            "loss_outer": mean_or_nan([p.loss_outer for p in active if p.loss_outer == p.loss_outer]),
            "loss_inner": mean_or_nan([p.loss_inner for p in active if p.loss_inner == p.loss_inner]),
        }


class PSSDTrainer:
    """Orchestrates the dual-loop training over full 8B rewriter parameters."""

    def __init__(self, cfg: PSSDConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        set_seed(cfg.train.seed)

        self.engine = PSSDEngine(cfg, load_rewriter=True)
        self.problems = load_problems(cfg.data.train_path, cfg.data.max_problems)
        self.sampler = ProblemSampler(self.problems, seed=cfg.train.seed)
        logger.info("Loaded %d training problems from %s", len(self.problems), cfg.data.train_path)
        logger.info(
            "Rewriter=%s (full FT) | solver=%s | OPD teacher=%s (no y*) | optimizer=%s",
            cfg.model.rewriter_name_or_path,
            cfg.model.name_or_path,
            cfg.model.teacher_name_or_path,
            cfg.train.optimizer,
        )

        params = self.engine.trainable_parameters()
        if not params:
            raise RuntimeError("Rewriter has no trainable parameters")
        self.optimizer = self._make_optimizer(params)
        # Cosine schedule is for the main loop; skip when train.steps=0 (SFT-only).
        self.scheduler = None
        if cfg.train.steps >= 1:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=min(cfg.train.warmup_steps, cfg.train.steps),
                num_training_steps=cfg.train.steps,
            )
        self.discriminator: Optional[AddedContentDiscriminator] = (
            AddedContentDiscriminator(self.engine) if cfg.reward.use_discriminator else None
        )

        self.output_dir = ensure_dir(cfg.train.output_dir)
        self.metrics = JsonlWriter(self.output_dir / "metrics.jsonl")
        self.metrics.write({"event": "config", **config_to_dict(cfg)})
        self.wandb = _init_wandb(cfg)

    def _make_optimizer(self, params: List[torch.nn.Parameter]) -> torch.optim.Optimizer:
        cfg = self.cfg
        if cfg.train.optimizer == "adamw8bit":
            try:
                import bitsandbytes as bnb
            except ImportError as exc:
                raise ImportError(
                    "train.optimizer=adamw8bit requires bitsandbytes "
                    "(needed to full-FT Qwen3-8B on one 80GB GPU)"
                ) from exc
            logger.info("Using bitsandbytes AdamW8bit for 8B full-parameter FT")
            return bnb.optim.AdamW8bit(
                params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
            )
        return torch.optim.AdamW(
            params, lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
        )

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #

    def _grade_rollout(self, rollout_ids: List[int], gold: str) -> float:
        return grade_rollout(self.engine, self.cfg, rollout_ids, gold)

    def _score_prompt_kl(self, prompt_text: str, g: int) -> float:
        return score_prompt_kl(self.engine, self.cfg, prompt_text, g)

    def _apply_penalties(self, problem: Problem, score: RewriteScore, x_tokens: int) -> None:
        apply_penalties(
            self.engine, self.cfg, problem, score, x_tokens, self.discriminator
        )

    def _score_rewrites(self, problem: Problem, scores: List[RewriteScore]) -> None:
        score_rewrites(
            self.engine, self.cfg, problem, scores, discriminator=self.discriminator
        )

    # ------------------------------------------------------------------ #
    # Losses
    # ------------------------------------------------------------------ #

    def _outer_loss(
        self, rewriter_prompt_ids: List[int], scores: List[RewriteScore]
    ) -> Optional[torch.Tensor]:
        """Group-relative REINFORCE over rewrite tokens (doc Sec. 3.2/3.3)."""
        rewards = [s.reward for s in scores]
        advantages = group_advantages(rewards)
        target_lists = [s.token_ids for s in scores]
        logprobs = self.engine.rewrite_logprobs_with_grad(rewriter_prompt_ids, target_lists)
        terms = []
        for adv, lp in zip(advantages, logprobs):
            if lp.numel() == 0:
                continue
            # Length-normalized per-rewrite log-likelihood weighted by advantage.
            terms.append(-float(adv) * lp.mean())
        if not terms:
            return None
        return torch.stack(terms).mean()

    def _inner_loss_opsd(
        self, problem: Problem, rewriter_prompt_ids: List[int], winner: RewriteScore
    ) -> Optional[torch.Tensor]:
        """Rewrite-space OPSD: clipped full-vocab forward KL on a fresh student rewrite.

        Teacher = frozen base conditioned on (x, r+) demo.
        Student = current rewriter conditioned on x alone.
        """
        _, student_ids = self.engine.sample_rewrites(rewriter_prompt_ids, k=1, use_adapter=True)
        rollout = student_ids[0]
        if len(rollout) == 0:
            return None
        demo_messages = inner_teacher_messages(
            problem.problem,
            winner.text,
            two_stage=self.cfg.rewrite.two_stage,
            style=self.cfg.rewrite.style,
            few_shot=self.cfg.rewrite.few_shot,
        )
        try:
            assert_no_leakage(
                self.engine.chat_text(
                    demo_messages,
                    enable_thinking=self.cfg.rewrite.enable_thinking,
                    rewriter=True,
                ),
                problem.reference_solution,
                allowed_text=problem.problem,
            )
        except LeakageError as exc:
            logger.warning("Inner-loop leakage guard on problem %s: %s", problem.id, exc)
            return None
        demo_prompt_ids = self.engine.chat_ids(
            demo_messages,
            enable_thinking=self.cfg.rewrite.enable_thinking,
            rewriter=True,
        )

        teacher_lp = self.engine.full_vocab_logprobs(
            demo_prompt_ids, rollout, use_adapter=False, requires_grad=False
        )
        student_lp = self.engine.full_vocab_logprobs(
            rewriter_prompt_ids, rollout, use_adapter=True, requires_grad=True
        )
        p_teacher = teacher_lp.exp()
        fkl_per_token = (p_teacher * (teacher_lp - student_lp)).sum(dim=-1)
        fkl_per_token = torch.clamp(fkl_per_token, max=self.cfg.inner.clip_tau)
        return fkl_per_token.mean()

    def _inner_loss_imitation(
        self, rewriter_prompt_ids: List[int], winner: RewriteScore
    ) -> Optional[torch.Tensor]:
        """Supervised NLL on Acc-selected winning rewrite tokens: -mean log q_phi(r+|x)."""
        targets = winner.token_ids
        if not targets:
            # Re-encode cleaned text if ids were stripped / empty.
            targets = self.engine.encode(winner.text, rewriter=True)
        if not targets:
            return None
        logprobs = self.engine.rewrite_logprobs_with_grad(rewriter_prompt_ids, [targets])
        lp = logprobs[0]
        if lp.numel() == 0:
            return None
        return -lp.mean()

    def _inner_loss(
        self, problem: Problem, rewriter_prompt_ids: List[int], winner: RewriteScore
    ) -> Optional[torch.Tensor]:
        if self.cfg.inner.mode == "imitation":
            return self._inner_loss_imitation(rewriter_prompt_ids, winner)
        return self._inner_loss_opsd(problem, rewriter_prompt_ids, winner)

    def _baseline_score(self, problem: Problem) -> RewriteScore:
        """Score the original problem under the same dense reward (for prefer-original)."""
        # Match rewriter input cleaning: drop OPSD \\boxed answer-format preamble.
        cleaned = strip_opsd_answer_preamble(problem.problem)
        baseline = RewriteScore(
            text=cleaned,
            solver_text=cleaned,
            valid=True,
        )
        self._score_rewrites(problem, [baseline])
        return baseline

    def _baseline_reward(self, problem: Problem) -> float:
        baseline = self._baseline_score(problem)
        if not baseline.valid or baseline.reward != baseline.reward:
            return float("nan")
        return float(baseline.reward)

    # ------------------------------------------------------------------ #
    # Per-problem processing
    # ------------------------------------------------------------------ #

    def _process_problem(self, problem: Problem) -> ProblemStats:
        cfg = self.cfg
        stats = ProblemStats()

        two_stage = cfg.rewrite.two_stage
        style = cfg.rewrite.style
        rewriter_msgs = rewriter_messages(
            problem.problem,
            two_stage=two_stage,
            style=style,
            few_shot=cfg.rewrite.few_shot,
        )
        # Structural guarantee: privileged info must never reach the rewriter.
        try:
            assert_no_leakage(
                self.engine.chat_text(
                    rewriter_msgs,
                    enable_thinking=cfg.rewrite.enable_thinking,
                    rewriter=True,
                ),
                problem.reference_solution,
                allowed_text=problem.problem,
            )
        except LeakageError as exc:
            # Should be rare after allowlisting problem-overlap; skip rather than kill the job.
            logger.warning("Leakage guard on problem %s: %s", problem.id, exc)
            stats.skipped, stats.skip_reason = True, "leakage_guard"
            return stats
        rewriter_prompt_ids = self.engine.chat_ids(
            rewriter_msgs, enable_thinking=cfg.rewrite.enable_thinking, rewriter=True
        )

        texts, ids_list = self.engine.sample_rewrites(rewriter_prompt_ids, cfg.rewrite.k)
        orig = strip_opsd_answer_preamble(problem.problem)
        scores: List[RewriteScore] = []
        for t, ids in zip(texts, ids_list):
            body, ok = extract_rewrite_body(t, require_tags=two_stage)
            valid = bool(t.strip()) and ok and bool(body.strip())
            if valid and cfg.rewrite.invalidate_solution_leak and looks_like_solution_leak(body):
                valid = False
            if valid and cfg.rewrite.inference_validate and validate_rewrite(orig, body, cfg):
                valid = False
            scores.append(
                RewriteScore(
                    text=t,
                    solver_text=body if ok else "",
                    token_ids=ids,
                    valid=valid,
                )
            )
        if sum(s.valid for s in scores) < (2 if cfg.outer.enabled else 1):
            stats.skipped, stats.skip_reason = True, "too_few_valid_rewrites"
            return stats
        scores = [s for s in scores if s.valid]

        self._score_rewrites(problem, scores)
        scores = [s for s in scores if s.valid]
        if len(scores) < (2 if cfg.outer.enabled else 1):
            stats.skipped, stats.skip_reason = True, "too_few_scored_rewrites"
            return stats

        rewards = [s.reward for s in scores]
        # Drop any rewrite whose dense score is non-finite before the outer loop.
        scores = [s for s in scores if s.reward == s.reward]
        if len(scores) < (2 if cfg.outer.enabled else 1):
            stats.skipped, stats.skip_reason = True, "too_few_finite_rewards"
            return stats
        rewards = [s.reward for s in scores]
        stats.reward_mean = float(sum(rewards) / len(rewards))
        stats.reward_std = float(torch.tensor(rewards).std(unbiased=False).item())
        stats.kl_mean = mean_or_nan([s.kl_score for s in scores])
        stats.acc_mean = mean_or_nan([s.acc_score for s in scores])
        stats.teacher_lik_mean = mean_or_nan([s.teacher_lik for s in scores])
        stats.length_ratio_mean = mean_or_nan([s.length_ratio for s in scores])
        stats.new_ngram_rate = mean_or_nan(
            [new_ngram_rate(problem.problem, s.prompt_for_solver()) for s in scores]
        )
        stats.reference_overlap = mean_or_nan(
            [
                reference_overlap(s.prompt_for_solver(), problem.reference_solution)
                for s in scores
            ]
        )
        stats.number_coverage = mean_or_nan([s.coverage for s in scores])
        stats.missing_number_rate = mean_or_nan([s.missing_number_rate for s in scores])
        stats.discriminator_rate = mean_or_nan([float(s.discriminator_flagged) for s in scores])

        # DAPO-style filtering: zero-std groups give a zero outer gradient, but the
        # inner loop (best-of-K distillation) can still receive a useful signal.
        outer_ok = (not cfg.outer.enabled) or group_is_informative(
            rewards, cfg.reward.group_std_eps
        )
        if cfg.outer.enabled and not outer_ok:
            stats.skip_reason = "zero_reward_std"

        loss = None
        if cfg.outer.enabled and outer_ok:
            loss_outer = self._outer_loss(rewriter_prompt_ids, scores)
            if loss_outer is not None:
                stats.loss_outer = float(loss_outer.item())
                loss = loss_outer

        if cfg.inner.enabled:
            # Acc imitation / Acc MVP: select and gate on pure Acc (ignore length etc.).
            use_acc_select = cfg.reward.dense_reward in ("acc", "acc_lik")
            if use_acc_select:
                winner = max(
                    scores,
                    key=lambda s: s.acc_score if s.acc_score == s.acc_score else float("-inf"),
                )
            else:
                winner = max(scores, key=lambda s: s.reward)
            skip_inner = False
            if cfg.reward.prefer_original_if_nonpositive:
                baseline = self._baseline_score(problem)
                if use_acc_select:
                    base_acc = baseline.acc_score if baseline.valid else float("nan")
                    win_acc = winner.acc_score
                    no_gain = (
                        base_acc == base_acc
                        and win_acc == win_acc
                        and win_acc <= base_acc
                    )
                else:
                    baseline_reward = (
                        float(baseline.reward)
                        if baseline.valid and baseline.reward == baseline.reward
                        else float("nan")
                    )
                    no_gain = (
                        baseline_reward == baseline_reward
                        and winner.reward <= baseline_reward
                    )
                if no_gain:
                    # Align with Acc MVP: do not distill a rewrite no better than x.
                    skip_inner = True
                    if loss is None:
                        stats.skipped = True
                        stats.skip_reason = "no_gain_vs_original"
                        return stats
                    stats.skip_reason = stats.skip_reason or "inner_skipped_no_gain"
            if not skip_inner:
                loss_inner = self._inner_loss(problem, rewriter_prompt_ids, winner)
                if loss_inner is not None:
                    stats.loss_inner = float(loss_inner.item())
                    weighted = cfg.inner.weight * loss_inner
                    loss = weighted if loss is None else loss + weighted

        if loss is None:
            stats.skipped = True
            stats.skip_reason = stats.skip_reason or "no_loss"
            return stats
        if not torch.isfinite(loss):
            stats.skipped, stats.skip_reason = True, "non_finite_loss"
            logger.warning("Non-finite loss on problem %s; skipping", problem.id)
            return stats

        # Inner (and/or outer) produced a usable loss; clear the advisory skip reason.
        if stats.skip_reason == "zero_reward_std":
            stats.skip_reason = "outer_skipped_zero_std"
        (loss / cfg.train.batch_size).backward()
        return stats

    # ------------------------------------------------------------------ #
    # Acc-winner collection (shared by SFT warmup)
    # ------------------------------------------------------------------ #

    def _collect_acc_winner_pair(
        self, problem: Problem
    ) -> Optional[Tuple[List[int], RewriteScore]]:
        """Return (rewriter_prompt_ids, winner) if dense score beats the original."""
        cfg = self.cfg
        two_stage = cfg.rewrite.two_stage
        rewriter_msgs = rewriter_messages(
            problem.problem,
            two_stage=two_stage,
            style=cfg.rewrite.style,
            few_shot=cfg.rewrite.few_shot,
        )
        try:
            assert_no_leakage(
                self.engine.chat_text(
                    rewriter_msgs, enable_thinking=cfg.rewrite.enable_thinking, rewriter=True
                ),
                problem.reference_solution,
                allowed_text=problem.problem,
            )
        except LeakageError:
            return None
        rewriter_prompt_ids = self.engine.chat_ids(
            rewriter_msgs, enable_thinking=cfg.rewrite.enable_thinking, rewriter=True
        )
        texts, ids_list = self.engine.sample_rewrites(rewriter_prompt_ids, cfg.rewrite.k)
        orig = strip_opsd_answer_preamble(problem.problem)
        scores: List[RewriteScore] = []
        for t, ids in zip(texts, ids_list):
            body, ok = extract_rewrite_body(t, require_tags=two_stage)
            valid = bool(t.strip()) and ok and bool(body.strip())
            if valid and cfg.rewrite.invalidate_solution_leak and looks_like_solution_leak(body):
                valid = False
            if valid and cfg.rewrite.inference_validate and validate_rewrite(orig, body, cfg):
                valid = False
            scores.append(
                RewriteScore(
                    text=t,
                    solver_text=body if ok else "",
                    token_ids=ids,
                    valid=valid,
                )
            )
        scores = [s for s in scores if s.valid]
        if not scores:
            return None
        self._score_rewrites(problem, scores)
        scores = [s for s in scores if s.valid and s.reward == s.reward]
        if not scores:
            return None
        winner = max(scores, key=lambda s: s.reward)
        baseline = self._baseline_score(problem)
        if not baseline.valid or baseline.reward != baseline.reward:
            return None
        if winner.reward <= baseline.reward:
            return None
        return rewriter_prompt_ids, winner

    def _pair_from_rewrite_text(
        self, problem_text: str, rewrite_text: str
    ) -> Tuple[List[int], RewriteScore]:
        cfg = self.cfg
        rewriter_msgs = rewriter_messages(
            problem_text,
            two_stage=cfg.rewrite.two_stage,
            style=cfg.rewrite.style,
            few_shot=cfg.rewrite.few_shot,
        )
        prompt_ids = self.engine.chat_ids(
            rewriter_msgs, enable_thinking=cfg.rewrite.enable_thinking, rewriter=True
        )
        body, ok = extract_rewrite_body(
            rewrite_text, require_tags=cfg.rewrite.two_stage
        )
        text = (body if ok else rewrite_text).strip()
        winner = RewriteScore(
            text=text,
            solver_text=text,
            token_ids=self.engine.encode(text, rewriter=True),
            valid=True,
        )
        return prompt_ids, winner

    def _load_sft_pairs_jsonl(self, path: Path) -> List[Tuple[List[int], RewriteScore]]:
        pairs: List[Tuple[List[int], RewriteScore]] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                problem = str(record.get("problem", "")).strip()
                rewrite = str(record.get("rewrite", "")).strip()
                if not problem or not rewrite:
                    logger.warning("%s:%d missing problem/rewrite; skip", path, lineno)
                    continue
                pairs.append(self._pair_from_rewrite_text(problem, rewrite))
        return pairs

    def _run_sft_warmup(self) -> int:
        """Collect Acc-beating rewrites on the train pool, then imitate them.

        Returns the number of optimizer updates performed (0 if skipped / empty).
        Logs per-update ``sft/loss`` to wandb when enabled.
        """
        cfg = self.cfg
        wcfg = cfg.warmup
        if not wcfg.enabled:
            return 0

        pairs: List[Tuple[List[int], RewriteScore]] = []
        scanned = 0
        if wcfg.pairs_path:
            pair_path = Path(wcfg.pairs_path)
            if not pair_path.exists():
                raise FileNotFoundError(f"warmup.pairs_path not found: {pair_path}")
            pairs = self._load_sft_pairs_jsonl(pair_path)
            logger.info("SFT warmup: loaded %d pairs from %s", len(pairs), pair_path)
        else:
            logger.info(
                "SFT warmup: collect up to %d Acc-winning pairs (scan ≤%d problems, "
                "K=%d G=%d, greedy=%s), then %d epoch(s)",
                wcfg.target_pairs,
                wcfg.max_problems,
                cfg.rewrite.k,
                cfg.rollout.g,
                cfg.rewrite.include_greedy,
                wcfg.sft_epochs,
            )
            order = list(self.problems)
            random.Random(cfg.train.seed + 17).shuffle(order)
            for problem in order:
                if len(pairs) >= wcfg.target_pairs or scanned >= wcfg.max_problems:
                    break
                scanned += 1
                got = self._collect_acc_winner_pair(problem)
                if got is None:
                    continue
                pairs.append(got)
                logger.info(
                    "warmup collected %d/%d (scanned %d) from %s Acc=%.3f",
                    len(pairs),
                    wcfg.target_pairs,
                    scanned,
                    problem.id,
                    got[1].acc_score,
                )

        self.metrics.write(
            {
                "event": "warmup_collect",
                "scanned": scanned,
                "n_pairs": len(pairs),
                "target_pairs": wcfg.target_pairs,
                "pairs_path": wcfg.pairs_path,
            }
        )
        if not pairs:
            logger.warning("SFT warmup found 0 Acc-winning pairs; continuing without warmup")
            if self.wandb is not None:
                self.wandb.log(
                    {"sft/n_pairs": 0, "sft/n_updates": 0, "sft/scanned": scanned},
                    step=0,
                )
            return 0

        n_updates = 0
        for epoch in range(1, wcfg.sft_epochs + 1):
            random.Random(cfg.train.seed + epoch).shuffle(pairs)
            for i in range(0, len(pairs), wcfg.sft_batch_size):
                batch = pairs[i : i + wcfg.sft_batch_size]
                self.optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                n_loss = 0
                for prompt_ids, winner in batch:
                    loss = self._inner_loss_imitation(prompt_ids, winner)
                    if loss is None:
                        continue
                    scaled = loss / len(batch)
                    scaled.backward()
                    loss_sum += float(loss.item())
                    n_loss += 1
                if n_loss == 0:
                    continue
                torch.nn.utils.clip_grad_norm_(
                    self.engine.trainable_parameters(), cfg.train.max_grad_norm
                )
                self.optimizer.step()
                # Keep cosine schedule for the main loop only.
                n_updates += 1
                mean_nll = loss_sum / n_loss
                if n_updates % max(1, cfg.train.log_every) == 0:
                    logger.info(
                        "warmup SFT epoch %d | update %d | mean_nll %.4f | pairs %d",
                        epoch,
                        n_updates,
                        mean_nll,
                        len(pairs),
                    )
                self.metrics.write(
                    {
                        "event": "sft_update",
                        "epoch": epoch,
                        "update": n_updates,
                        "loss": mean_nll,
                        "n_pairs": len(pairs),
                        "lr": cfg.train.learning_rate,
                    }
                )
                if self.wandb is not None:
                    self.wandb.log(
                        {
                            "sft/loss": mean_nll,
                            "sft/epoch": epoch,
                            "sft/n_pairs": len(pairs),
                            "sft/lr": cfg.train.learning_rate,
                            "sft/update": n_updates,
                        },
                        step=n_updates,
                    )

        warm_dir = self.output_dir / "checkpoint-warmup"
        self.engine.save_rewriter(warm_dir)
        logger.info("SFT warmup done: %d pairs, %d updates → %s", len(pairs), n_updates, warm_dir)
        self.metrics.write(
            {
                "event": "warmup_done",
                "n_pairs": len(pairs),
                "n_updates": n_updates,
                "rewriter": str(warm_dir),
            }
        )
        if self.wandb is not None:
            self.wandb.log(
                {
                    "sft/n_pairs_final": len(pairs),
                    "sft/n_updates_final": n_updates,
                    "sft/scanned": scanned,
                },
                step=max(n_updates, 1),
            )
        return n_updates

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def train(self) -> None:
        cfg = self.cfg
        sft_updates = self._run_sft_warmup()
        if cfg.train.steps <= 0:
            logger.info(
                "SFT-only mode (train.steps=0): skipping outer/inner main loop "
                "(%d SFT updates done)",
                sft_updates,
            )
        else:
            logger.info(
                "Starting PSSD training: %d steps x %d problems (K=%d, G=%d, outer=%s, "
                "inner=%s, greedy=%s)",
                cfg.train.steps, cfg.train.batch_size, cfg.rewrite.k, cfg.rollout.g,
                cfg.outer.enabled, cfg.inner.enabled, cfg.rewrite.include_greedy,
            )
        for step in range(1, cfg.train.steps + 1):
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            step_stats = StepStats()

            for problem in self.sampler.sample_batch(cfg.train.batch_size):
                pst = self._process_problem(problem)
                step_stats.problems.append(pst)
                if pst.skipped:
                    logger.info(
                        "step %d problem %s skipped (%s)",
                        step,
                        problem.id,
                        pst.skip_reason or "unknown",
                    )

            summary = step_stats.summary()
            did_update = summary["n_skipped"] < summary["n_problems"]
            if did_update:
                torch.nn.utils.clip_grad_norm_(
                    self.engine.trainable_parameters(), cfg.train.max_grad_norm
                )
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            else:
                logger.warning(
                    "Step %d: every problem was skipped; no update | skip_reasons=%s",
                    step,
                    summary.get("skip_reasons"),
                )

            lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler is not None
                else cfg.train.learning_rate
            )
            summary.update({
                "event": "step",
                "step": step,
                "lr": lr,
                "updated": did_update,
                "seconds": round(time.time() - t0, 2),
            })
            self.metrics.write(summary)
            if self.wandb is not None:
                # Offset main-loop steps so they do not collide with sft/update steps.
                log_payload = {k: v for k, v in summary.items() if k != "event"}
                reasons = log_payload.pop("skip_reasons", {}) or {}
                for reason, count in reasons.items():
                    log_payload[f"skip/{reason}"] = count
                self.wandb.log(log_payload, step=sft_updates + step)
            if step % cfg.train.log_every == 0:
                logger.info(
                    "step %d/%d | reward %.4f (std %.4f) | acc %.3f | lik %.4f | "
                    "loss_out %.4f | loss_in %.4f | len_ratio %.2f | skipped %d/%d | "
                    "skip_reasons=%s | %.1fs",
                    step, cfg.train.steps, summary["reward_mean"], summary["reward_std"],
                    summary["acc_mean"], summary["teacher_lik_mean"],
                    summary["loss_outer"], summary["loss_inner"], summary["length_ratio"],
                    summary["n_skipped"], summary["n_problems"],
                    summary.get("skip_reasons"), summary["seconds"],
                )
            if step % cfg.train.save_every == 0 or step == cfg.train.steps:
                ckpt = self.output_dir / f"checkpoint-{step}"
                self.engine.save_rewriter(ckpt)
                logger.info("Saved rewriter checkpoint: %s", ckpt)

        final_rewriter = self.output_dir / "rewriter"
        self.engine.save_rewriter(final_rewriter)
        self.metrics.write(
            {
                "event": "done",
                "steps": cfg.train.steps,
                "sft_updates": sft_updates,
                "rewriter": str(final_rewriter),
            }
        )
        self.metrics.close()
        if self.wandb is not None:
            self.wandb.finish()
        logger.info("Training complete. Final rewriter: %s", final_rewriter)
