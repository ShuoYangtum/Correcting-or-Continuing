"""ICPS training loops: batched CE and OPSD-style KL → encoder."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from pssd.data import Problem, ProblemSampler
from pssd.icps import ICPSEngine
from pssd.utils import get_logger

logger = get_logger(__name__)


class _RunningMean:
    """Simple exponential moving average for logging."""

    def __init__(self, momentum: float = 0.95) -> None:
        self.momentum = float(momentum)
        self.value: Optional[float] = None

    def update(self, x: float) -> float:
        if x != x:  # NaN
            return self.value if self.value is not None else float("nan")
        if self.value is None:
            self.value = float(x)
        else:
            m = self.momentum
            self.value = m * self.value + (1.0 - m) * float(x)
        return self.value


class _EarlyStop:
    """Stop when EMA has not improved by ``min_delta`` for ``patience`` steps.

    ``patience<=0`` disables. No stop before ``min_steps``. Returns
    ``(is_best, should_stop)`` from ``update``.
    """

    def __init__(
        self,
        *,
        patience: int,
        min_delta: float,
        min_steps: int,
        name: str,
    ) -> None:
        self.patience = max(0, int(patience))
        self.min_delta = float(min_delta)
        self.min_steps = max(0, int(min_steps))
        self.name = name
        self.best = float("inf")
        self.best_step = 0
        self.wait = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def update(self, step: int, ema: float) -> Tuple[bool, bool]:
        if (not self.enabled) or ema != ema:
            return False, False
        improved = ema < (self.best - self.min_delta)
        if improved or self.best_step == 0:
            self.best = float(ema)
            self.best_step = int(step)
            self.wait = 0
            return True, False
        if step < self.min_steps:
            return False, False
        self.wait += 1
        if self.wait >= self.patience:
            logger.info(
                "%s early stop at step %d (best ema=%.4f at step %d, patience=%d)",
                self.name,
                step,
                self.best,
                self.best_step,
                self.patience,
            )
            return False, True
        return False, False


def _chunks(items: Sequence[Problem], size: int) -> List[List[Problem]]:
    size = max(1, int(size))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def _snapshot_encoder(engine: ICPSEngine) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in engine.encoder.state_dict().items()}


def _restore_encoder(engine: ICPSEngine, state: Dict[str, torch.Tensor]) -> None:
    engine.encoder.load_state_dict(state)


def _step_optimizer(
    engine: ICPSEngine,
    opt: torch.optim.Optimizer,
    *,
    step_label: str,
) -> Tuple[bool, float]:
    """Clip grads and step. Returns (ok, grad_norm)."""
    params = engine.trainable_parameters()
    bad = False
    for p in params:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            p.grad = None
            bad = True
    if bad:
        logger.warning("%s: non-finite grad; skip step", step_label)
        opt.zero_grad(set_to_none=True)
        return False, float("nan")
    grad_norm = float(
        torch.nn.utils.clip_grad_norm_(params, engine.icps.max_grad_norm)
    )
    if not math.isfinite(grad_norm):
        logger.warning("%s: non-finite grad_norm; skip step", step_label)
        opt.zero_grad(set_to_none=True)
        return False, float("nan")
    opt.step()
    opt.zero_grad(set_to_none=True)
    if not engine.encoder_is_finite():
        raise RuntimeError(f"{step_label}: trainable params became non-finite after step")
    return True, grad_norm


def train_ce(
    engine: ICPSEngine,
    problems: Sequence[Problem],
    epochs: int,
    *,
    wb: Any = None,
    global_step_start: int = 0,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Batched token-weighted CE. Returns ``(n_optimizer_steps, next_global_step)``."""
    icps = engine.icps
    if epochs <= 0 or not problems:
        return 0, global_step_start

    batch_size = max(1, int(icps.ce_batch_size))
    accum = max(1, int(icps.ce_grad_accum))
    log_every = max(1, int(icps.log_every))
    save_every = max(0, int(icps.save_every))
    ce_steps = icps.ce_steps
    if ce_steps is not None:
        ce_steps = max(1, int(ce_steps))

    # Keep frozen LM in its load dtype (bf16). Casting to fp32 OOMs at 8k.
    opt = torch.optim.AdamW(
        engine.encoder.parameters(),
        lr=icps.learning_rate,
        weight_decay=icps.weight_decay,
        eps=1e-8,
    )
    engine.encoder.train()
    opt.zero_grad(set_to_none=True)
    if hasattr(engine.model, "gradient_checkpointing_enable"):
        engine.model.gradient_checkpointing_enable()
        if hasattr(engine.model, "config"):
            engine.model.config.use_cache = False

    n_ok = 0
    n_skip = 0
    n_examples = 0
    gstep = global_step_start
    loss_ema = _RunningMean()
    early = _EarlyStop(
        patience=int(getattr(icps, "ce_early_stop_patience", 0)),
        min_delta=float(getattr(icps, "ce_early_stop_min_delta", 0.0)),
        min_steps=int(getattr(icps, "ce_early_stop_min_steps", 0)),
        name="CE",
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    micro = 0
    accum_loss = 0.0
    accum_tokens = 0.0
    accum_norm = 0.0
    accum_batches = 0
    stop_ce = False
    stopped_early = False

    batches_per_epoch = math.ceil(len(problems) / batch_size)
    total_batches = epochs * batches_per_epoch
    total_opt = math.ceil(total_batches / accum)
    if ce_steps is not None:
        total_opt = min(total_opt, ce_steps)

    logger.info(
        "CE: %d problems | epochs=%d | batch=%d accum=%d | target_opt_steps=%s | lr=%.2e wd=%.2e",
        len(problems),
        epochs,
        batch_size,
        accum,
        str(ce_steps) if ce_steps is not None else f"~{total_opt}(full)",
        icps.learning_rate,
        icps.weight_decay,
    )
    if early.enabled:
        logger.info(
            "CE early stop: patience=%d min_delta=%.4f min_steps=%d",
            early.patience,
            early.min_delta,
            early.min_steps,
        )
    if wb is not None:
        wb.log(
            {
                "train/phase": 0,
                "train/phase_name_ce": 1,
                "ce/n_problems": len(problems),
                "ce/epochs": epochs,
                "ce/steps": ce_steps if ce_steps is not None else total_opt,
                "ce/batch_size": batch_size,
                "ce/grad_accum": accum,
                "ce/effective_batch": batch_size * accum,
                "ce/lr": icps.learning_rate,
                "ce/weight_decay": icps.weight_decay,
                "ce/prefix_l2_coef": icps.prefix_l2_coef,
                "ce/early_stop_patience": early.patience,
                "ce/early_stop_min_delta": early.min_delta,
                "ce/early_stop_min_steps": early.min_steps,
            },
            step=gstep,
        )

    try:
        batch_idx = 0
        for epoch in range(1, epochs + 1):
            if stop_ce:
                break
            order = list(problems)
            random.Random(engine.cfg.train.seed + epoch).shuffle(order)
            for batch in _chunks(order, batch_size):
                if ce_steps is not None and n_ok >= ce_steps:
                    stop_ce = True
                    break
                batch_idx += 1
                texts = [p.problem for p in batch]
                refs = [p.reference_solution for p in batch]
                try:
                    loss, stats = engine.nll_on_reference_batch(texts, refs)
                except torch.cuda.OutOfMemoryError:
                    logger.warning(
                        "CE batch %d: CUDA OOM; skip and empty cache", batch_idx
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    opt.zero_grad(set_to_none=True)
                    n_skip += 1
                    continue
                loss_val = float(loss.detach().float().item())
                n_tok = float(stats.get("n_tokens", 0.0))

                if (not math.isfinite(loss_val)) or abs(loss_val) > 1e4 or n_tok < 1:
                    logger.warning(
                        "CE batch %d: skip (loss=%s tokens=%s)",
                        batch_idx,
                        loss_val,
                        n_tok,
                    )
                    n_skip += 1
                    continue

                # Scale for gradient accumulation (mean over micro-batches).
                (loss / float(accum)).backward()
                micro += 1
                n_examples += len(batch)
                accum_loss += loss_val
                accum_tokens += n_tok
                accum_norm += float(stats.get("prefix_norm", float("nan")))
                accum_batches += 1

                if micro < accum:
                    continue

                gstep += 1
                ok, grad_norm = _step_optimizer(
                    engine, opt, step_label=f"CE opt {gstep}"
                )
                micro = 0
                if not ok:
                    n_skip += 1
                    accum_loss = 0.0
                    accum_tokens = 0.0
                    accum_norm = 0.0
                    accum_batches = 0
                    continue

                n_ok += 1
                mean_loss = accum_loss / max(1, accum_batches)
                mean_norm = accum_norm / max(1, accum_batches)
                ema = loss_ema.update(mean_loss)
                mean_tokens = accum_tokens / max(1, accum_batches)
                accum_loss = 0.0
                accum_tokens = 0.0
                accum_norm = 0.0
                accum_batches = 0

                is_best, should_stop = early.update(n_ok, float(ema) if ema == ema else float("nan"))
                if is_best:
                    best_state = _snapshot_encoder(engine)
                    if checkpoint_dir is not None:
                        engine.save_encoder(
                            Path(checkpoint_dir) / "icps_encoder_ce_best.pt",
                            extra={
                                "phase": "ce_best",
                                "opt_step": n_ok,
                                "global_step": gstep,
                                "loss_ema": float(ema),
                            },
                        )

                progress_den = float(ce_steps if ce_steps is not None else max(1, total_opt))
                if wb is not None and (
                    n_ok % log_every == 0
                    or n_ok == 1
                    or is_best
                    or should_stop
                    or (ce_steps is not None and n_ok >= ce_steps)
                ):
                    wb.log(
                        {
                            "global_step": gstep,
                            "train/phase": 0,
                            "ce/loss": mean_loss,
                            "ce/loss_ema": ema,
                            "ce/prefix_norm": mean_norm,
                            "ce/tokens_per_batch": mean_tokens,
                            "ce/epoch": epoch,
                            "ce/batch_in_phase": batch_idx,
                            "ce/ok_updates": n_ok,
                            "ce/skip_updates": n_skip,
                            "ce/examples_seen": n_examples,
                            "ce/grad_norm": grad_norm,
                            "ce/progress": n_ok / progress_den,
                            "ce/best_ema": early.best if early.best < float("inf") else ema,
                            "ce/best_step": early.best_step,
                            "ce/patience_wait": early.wait,
                        },
                        step=gstep,
                    )

                if n_ok == 1 or n_ok % log_every == 0 or (ce_steps is not None and n_ok >= ce_steps):
                    logger.info(
                        "CE epoch %d/%d opt=%d/%s loss=%.4f ema=%.4f "
                        "prefix_norm=%.4f tokens/batch=%.1f grad=%.3f",
                        epoch,
                        epochs,
                        n_ok,
                        str(ce_steps) if ce_steps is not None else "full",
                        mean_loss,
                        ema if ema == ema else float("nan"),
                        mean_norm,
                        mean_tokens,
                        grad_norm,
                    )

                if (
                    checkpoint_dir is not None
                    and save_every > 0
                    and n_ok % save_every == 0
                ):
                    engine.save_encoder(
                        Path(checkpoint_dir) / f"icps_encoder_ce_step{n_ok}.pt",
                        extra={"phase": "ce", "opt_step": n_ok, "global_step": gstep},
                    )

                if should_stop:
                    stopped_early = True
                    stop_ce = True
                    break
                if ce_steps is not None and n_ok >= ce_steps:
                    stop_ce = True
                    break

        # Flush a partial accumulation window (do not exceed ce_steps).
        if micro > 0 and (not stopped_early) and (ce_steps is None or n_ok < ce_steps):
            gstep += 1
            # Rescale: grads were divided by full `accum`, but only `micro` micros ran.
            scale = float(accum) / float(micro)
            for p in engine.encoder.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            ok, grad_norm = _step_optimizer(
                engine, opt, step_label=f"CE opt flush {gstep}"
            )
            micro = 0
            if ok:
                n_ok += 1
                mean_loss = accum_loss / max(1, accum_batches)
                mean_norm = accum_norm / max(1, accum_batches)
                ema = loss_ema.update(mean_loss)
                is_best, _ = early.update(
                    n_ok, float(ema) if ema == ema else float("nan")
                )
                if is_best:
                    best_state = _snapshot_encoder(engine)
                    if checkpoint_dir is not None:
                        engine.save_encoder(
                            Path(checkpoint_dir) / "icps_encoder_ce_best.pt",
                            extra={
                                "phase": "ce_best",
                                "opt_step": n_ok,
                                "global_step": gstep,
                                "loss_ema": float(ema),
                            },
                        )
                logger.info(
                    "CE flush opt=%d loss=%.4f ema=%.4f prefix_norm=%.4f grad=%.3f",
                    n_ok,
                    mean_loss,
                    ema if ema == ema else float("nan"),
                    mean_norm,
                    grad_norm,
                )
            accum_loss = 0.0
            accum_tokens = 0.0
            accum_norm = 0.0
            accum_batches = 0
        elif micro > 0:
            opt.zero_grad(set_to_none=True)
            micro = 0
            accum_loss = 0.0
            accum_tokens = 0.0
            accum_norm = 0.0
            accum_batches = 0
    finally:
        if hasattr(engine.model, "gradient_checkpointing_disable"):
            engine.model.gradient_checkpointing_disable()
            if hasattr(engine.model, "config"):
                engine.model.config.use_cache = True
        engine.encoder.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    restored = False
    if best_state is not None:
        _restore_encoder(engine, best_state)
        restored = True
        logger.info(
            "CE: restored best encoder (step=%d ema=%.4f, ran=%d%s)",
            early.best_step,
            early.best,
            n_ok,
            ", early stop" if stopped_early else "",
        )

    if wb is not None:
        wb.log(
            {
                "global_step": gstep,
                "ce/finished": 1,
                "ce/ok_updates_final": n_ok,
                "ce/skip_updates_final": n_skip,
                "ce/examples_seen_final": n_examples,
                "ce/early_stop": int(stopped_early),
                "ce/best_step": early.best_step,
                "ce/best_ema": early.best if early.best < float("inf") else float("nan"),
                "ce/restored_best": int(restored),
            },
            step=gstep,
        )
    return n_ok, gstep


def train_opsd(
    engine: ICPSEngine,
    problems: Sequence[Problem],
    steps: int,
    *,
    wb: Any = None,
    global_step_start: int = 0,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[int, int]:
    """Batched on-policy ``KL(p_T||p_S)`` → encoder. Returns ``(n_ok, next_gstep)``."""
    icps = engine.icps
    phase = "OPD" if getattr(engine, "distill", "") == "opd" else "OPSD"
    if steps <= 0 or not problems:
        return 0, global_step_start
    if not engine.encoder_is_finite():
        logger.warning("Skip OPSD phase: encoder non-finite")
        return 0, global_step_start

    batch_size = max(1, int(icps.opsd_batch_size))
    accum = max(1, int(icps.opsd_grad_accum))
    log_every = max(1, int(icps.log_every))
    save_every = max(0, int(getattr(icps, "opsd_save_every", icps.save_every)))

    opt = torch.optim.AdamW(
        engine.trainable_parameters(),
        lr=icps.learning_rate * icps.opsd_lr_scale,
        weight_decay=icps.weight_decay,
        eps=1e-8,
    )
    sampler = ProblemSampler(list(problems), seed=engine.cfg.train.seed + 99)
    if engine.uses_prefix():
        engine.encoder.train()
    else:
        engine.model.train()
    opt.zero_grad(set_to_none=True)

    n_ok = 0
    n_skip = 0
    n_gate_skip = 0
    n_consecutive_gated = 0
    gstep = global_step_start
    loss_ema = _RunningMean()
    early = _EarlyStop(
        patience=int(getattr(icps, "opsd_early_stop_patience", 0)),
        min_delta=float(getattr(icps, "opsd_early_stop_min_delta", 0.0)),
        min_steps=int(getattr(icps, "opsd_early_stop_min_steps", 0)),
        name=phase,
    )
    gate_epoch = max(len(problems), 32)
    best_state: Optional[Dict[str, torch.Tensor]] = None
    stopped_early = False
    micro = 0
    accum_loss = 0.0
    accum_norm = 0.0
    accum_batches = 0
    total_micro = steps * accum

    logger.info(
        "%s-KL: %d opt steps | batch=%d accum=%d | lr=%.2e | max_new=%d | "
        "save_every=%d | kl_topk=%s | kl_max_tokens=%s | kl_seq_chunk=%s | "
        "kl_direction=%s | advantage_gate=%s",
        phase,
        steps,
        batch_size,
        accum,
        icps.learning_rate * icps.opsd_lr_scale,
        icps.opsd_max_new_tokens,
        save_every,
        str(getattr(icps, "opsd_kl_topk", 0)),
        str(getattr(icps, "opsd_kl_max_tokens", 0)),
        str(getattr(icps, "opsd_kl_seq_chunk", 0)),
        str(getattr(icps, "opsd_kl_direction", "forward")),
        str(getattr(icps, "opsd_advantage_gate", False)),
    )
    if early.enabled:
        logger.info(
            "OPSD early stop: patience=%d min_delta=%.4f min_steps=%d",
            early.patience,
            early.min_delta,
            early.min_steps,
        )
    if wb is not None:
        wb.log(
            {
                "train/phase": 1,
                "train/phase_name_opsd": 1,
                "opsd/n_steps": steps,
                "opsd/batch_size": batch_size,
                "opsd/grad_accum": accum,
                "opsd/effective_batch": batch_size * accum,
                "opsd/lr": icps.learning_rate * icps.opsd_lr_scale,
                "opsd/max_new_tokens": icps.opsd_max_new_tokens,
                "opsd/kl_clip": icps.opsd_kl_clip,
                "opsd/early_stop_patience": early.patience,
                "opsd/early_stop_min_delta": early.min_delta,
                "opsd/early_stop_min_steps": early.min_steps,
            },
            step=gstep,
        )

    micro_idx = 0
    while n_ok < steps:
        batch = sampler.sample_batch(batch_size)
        texts = [p.problem for p in batch]
        refs = [p.reference_solution for p in batch]
        answers = [p.answer for p in batch]
        loss, stats = engine.opsd_kl_loss_batch(
            texts, refs, answers=answers, grad_scale=1.0 / float(accum)
        )
        loss_val = float(loss.detach().float().item())
        n_tok = float(stats.get("n_tokens", 0.0))
        n_distilled = float(stats.get("n_ok", 0.0))
        n_gated = float(stats.get("n_gated", 0.0))
        micro_idx += 1

        if (not math.isfinite(loss_val)) or abs(loss_val) > 1e3 or n_distilled < 1 or n_tok < 1:
            gated_only = n_gated >= 1.0 and n_distilled < 1 and math.isfinite(loss_val)
            if gated_only:
                n_gate_skip += 1
                n_consecutive_gated += 1
                if n_consecutive_gated == 1 or n_consecutive_gated % log_every == 0:
                    logger.info(
                        "%s micro %d: gated skip (already-correct=%s consecutive=%d)",
                        phase,
                        micro_idx,
                        n_gated,
                        n_consecutive_gated,
                    )
                if wb is not None and n_consecutive_gated % log_every == 0:
                    wb.log(
                        {
                            "opsd/gate_frac": float(stats.get("gate_frac", 0.0)),
                            "opsd/onpolicy_acc": float(stats.get("onpolicy_acc", float("nan"))),
                            "opsd/n_gated": n_gated,
                            "opsd/n_gate_skip": n_gate_skip,
                        },
                        step=max(gstep, 1),
                    )
                if n_consecutive_gated >= gate_epoch:
                    logger.info(
                        "%s: student already correct for %d consecutive micros; "
                        "stopping distill early",
                        phase,
                        n_consecutive_gated,
                    )
                    break
                continue
            logger.warning(
                "%s micro %d: skip (kl=%s tokens=%s distilled=%s gated=%s)",
                phase,
                micro_idx,
                loss_val,
                n_tok,
                n_distilled,
                n_gated,
            )
            n_skip += 1
            n_consecutive_gated = 0
            if n_skip > total_micro * 5:
                raise RuntimeError(f"{phase} phase: too many skipped micro-batches")
            continue
        n_consecutive_gated = 0

        if not stats.get("backward_done"):
            (loss / float(accum)).backward()
        micro += 1
        accum_loss += loss_val
        accum_norm += float(stats.get("prefix_norm", float("nan")))
        accum_batches += 1

        if micro < accum:
            continue

        gstep += 1
        ok, grad_norm = _step_optimizer(engine, opt, step_label=f"OPSD opt {gstep}")
        micro = 0
        if not ok:
            n_skip += 1
            accum_loss = 0.0
            accum_norm = 0.0
            accum_batches = 0
            continue

        n_ok += 1
        mean_loss = accum_loss / max(1, accum_batches)
        mean_norm = accum_norm / max(1, accum_batches)
        ema = loss_ema.update(mean_loss)
        accum_loss = 0.0
        accum_norm = 0.0
        accum_batches = 0

        is_best, should_stop = early.update(
            n_ok, float(ema) if ema == ema else float("nan")
        )
        if is_best:
            if engine.uses_prefix():
                best_state = _snapshot_encoder(engine)
                if checkpoint_dir is not None:
                    engine.save_encoder(
                        Path(checkpoint_dir) / "icps_encoder_opsd_best.pt",
                        extra={
                            "phase": "opsd_best",
                            "opt_step": n_ok,
                            "global_step": gstep,
                            "kl_ema": float(ema),
                        },
                    )
            elif checkpoint_dir is not None:
                engine.save_student(
                    Path(checkpoint_dir) / "student_opd_best",
                    extra={
                        "phase": "opsd_best",
                        "opt_step": n_ok,
                        "global_step": gstep,
                        "kl_ema": float(ema),
                    },
                )
                best_state = {"disk": True}

        if wb is not None and (
            n_ok % log_every == 0
            or n_ok == 1
            or is_best
            or should_stop
            or n_ok == steps
        ):
            wb.log(
                {
                    "global_step": gstep,
                    "train/phase": 1,
                    "opsd/kl_loss": mean_loss,
                    "opsd/kl_loss_ema": ema,
                    "opsd/prefix_norm": mean_norm,
                    "opsd/step_in_phase": n_ok,
                    "opsd/ok_updates": n_ok,
                    "opsd/skip_updates": n_skip,
                    "opsd/grad_norm": grad_norm,
                    "opsd/progress": n_ok / max(1, steps),
                    "opsd/best_ema": early.best if early.best < float("inf") else ema,
                    "opsd/best_step": early.best_step,
                    "opsd/patience_wait": early.wait,
                    "opsd/gate_frac": float(stats.get("gate_frac", 0.0)),
                    "opsd/onpolicy_acc": float(stats.get("onpolicy_acc", float("nan"))),
                    "opsd/n_gated": float(stats.get("n_gated", 0.0)),
                    "opsd/n_gate_skip": n_gate_skip,
                },
                step=gstep,
            )

        if n_ok == 1 or n_ok % log_every == 0 or n_ok == steps:
            logger.info(
                "%s step %d/%d kl=%.4f ema=%.4f prefix_norm=%.4f grad=%.3f gate=%.2f",
                phase,
                n_ok,
                steps,
                mean_loss,
                ema if ema == ema else float("nan"),
                mean_norm,
                grad_norm,
                float(stats.get("gate_frac", 0.0)),
            )

        if checkpoint_dir is not None and save_every > 0 and n_ok % save_every == 0:
            if engine.uses_prefix():
                engine.save_encoder(
                    Path(checkpoint_dir) / f"icps_encoder_opsd_step{n_ok}.pt",
                    extra={"phase": "opsd", "opt_step": n_ok, "global_step": gstep},
                )
            else:
                engine.save_student(
                    Path(checkpoint_dir) / f"student_opd_step{n_ok}",
                    extra={"phase": "opsd", "opt_step": n_ok, "global_step": gstep},
                )

        if should_stop:
            stopped_early = True
            break

    restored = False
    if engine.uses_prefix() and best_state is not None:
        _restore_encoder(engine, best_state)
        restored = True
        logger.info(
            "%s: restored best encoder (step=%d ema=%.4f, ran=%d%s)",
            phase,
            early.best_step,
            early.best,
            n_ok,
            ", early stop" if stopped_early else "",
        )
    elif (not engine.uses_prefix()) and checkpoint_dir is not None:
        best_dir = Path(checkpoint_dir) / "student_opd_best"
        if best_dir.exists():
            engine.load_student(best_dir)
            restored = True
            logger.info(
                "%s: restored best student (step=%d ema=%.4f, ran=%d%s)",
                phase,
                early.best_step,
                early.best,
                n_ok,
                ", early stop" if stopped_early else "",
            )

    engine.encoder.eval()
    engine.model.eval()
    if wb is not None:
        wb.log(
            {
                "global_step": gstep,
                "opsd/finished": 1,
                "opsd/ok_updates_final": n_ok,
                "opsd/skip_updates_final": n_skip,
                "opsd/gate_skip_final": n_gate_skip,
                "opsd/early_stop": int(stopped_early),
                "opsd/best_step": early.best_step,
                "opsd/best_ema": early.best if early.best < float("inf") else float("nan"),
                "opsd/restored_best": int(restored),
            },
            step=gstep,
        )
    return n_ok, gstep


def evaluate_icps(
    engine: ICPSEngine,
    problems: Sequence[Problem],
    writer: Any,
    *,
    wb: Any = None,
    global_step: int = 0,
) -> Dict[str, Any]:
    """Eval frozen base vs trained student.

    Prefix mode: base (optional vLLM of the original snapshot) vs ICPS
    soft-prefix vs prefer_zero. Full-FT mode: same base vs the updated
    weights with no prefix; prefer_zero is ignored.
    """
    from pssd.utils import mean_or_nan

    icps = engine.icps
    use_prefix = engine.uses_prefix()
    prefer = bool(icps.prefer_zero_if_nonpositive) and use_prefix
    engine.encoder.eval()
    engine.model.eval()

    if icps.use_vllm_base:
        try:
            engine.attach_vllm_base()
        except Exception as exc:
            logger.warning("vLLM attach failed (%s); scoring base without vLLM", exc)

    texts = [p.problem for p in problems]
    golds = [p.answer for p in problems]

    def _score_base(n: int) -> List[float]:
        if engine._vllm is None and not use_prefix:
            return engine.accuracy_snapshot_batch(
                texts, golds, n, engine.cfg.model.name_or_path
            )
        return engine.accuracy_base_batch(texts, golds, n)

    logger.info("Eval: base score@%d on %d problems", icps.g_score, len(problems))
    score_base = _score_base(icps.g_score)
    if icps.n_solve == icps.g_score:
        acc_base = list(score_base)
    else:
        logger.info("Eval: base Avg@%d", icps.n_solve)
        acc_base = _score_base(icps.n_solve)

    if engine._vllm is not None:
        engine.release_vllm_restore_hf()
        engine.model.eval()

    student_label = "ICPS soft-prefix (HF)" if use_prefix else "full-FT student (HF, no prefix)"
    logger.info("Eval: %s", student_label)
    score_icps: List[float] = []
    acc_icps: List[float] = []
    norms: List[float] = []
    for i, problem in enumerate(problems, start=1):
        if use_prefix:
            with torch.no_grad():
                soft = engine.soft_prefix(problem.problem)
                norms.append(engine.prefix_norm(soft))
        else:
            norms.append(float("nan"))
        si = engine.accuracy(
            problem.problem, problem.answer, icps.g_score, use_icps=use_prefix
        )
        if icps.n_solve == icps.g_score:
            ai = si
        else:
            ai = engine.accuracy(
                problem.problem,
                problem.answer,
                icps.n_solve,
                use_icps=use_prefix,
            )
        score_icps.append(si)
        acc_icps.append(ai)
        if wb is not None and (i == 1 or i % 5 == 0 or i == len(problems)):
            wb.log(
                {
                    "global_step": global_step + i,
                    "eval/running_acc_icps_mean": mean_or_nan(acc_icps),
                    "eval/running_acc_base_mean": mean_or_nan(acc_base[:i]),
                    "eval/problem_idx": i,
                    "eval/prefix_norm": norms[-1],
                },
                step=global_step + i,
            )
        if i == 1 or i % 5 == 0 or i == len(problems):
            logger.info(
                "Student eval %d/%d %s: score=%.3f acc=%.3f%s",
                i,
                len(problems),
                problem.id,
                si,
                ai,
                f" norm={norms[-1]:.3f}" if use_prefix else "",
            )

    arms = {"base": [], "icps": [], "selected": []}
    n_fallback = 0
    for i, problem in enumerate(problems):
        use_icps = True
        if prefer and (
            score_icps[i] != score_icps[i] or score_icps[i] <= score_base[i]
        ):
            use_icps = False
            n_fallback += 1
        sel = acc_icps[i] if use_icps else acc_base[i]
        arms["base"].append(acc_base[i])
        arms["icps"].append(acc_icps[i])
        arms["selected"].append(sel)
        writer.write(
            {
                "id": problem.id,
                "score_base": score_base[i],
                "score_icps": score_icps[i],
                "used_icps": use_icps,
                "prefix_norm": norms[i],
                "acc_base": acc_base[i],
                "acc_icps": acc_icps[i],
                "acc_selected": sel,
            }
        )

    summary = {
        "event": "summary",
        "n_problems": len(arms["base"]),
        "avg_acc_base": mean_or_nan(arms["base"]),
        "avg_acc_icps": mean_or_nan(arms["icps"]),
        "avg_acc_selected": mean_or_nan(arms["selected"]),
        "n_fallback_base": n_fallback,
        "prefer_zero_if_nonpositive": prefer,
        "delta_icps_vs_base_pp": (
            (mean_or_nan(arms["icps"]) - mean_or_nan(arms["base"])) * 100.0
            if arms["base"]
            else float("nan")
        ),
        "delta_selected_vs_base_pp": (
            (mean_or_nan(arms["selected"]) - mean_or_nan(arms["base"])) * 100.0
            if arms["base"]
            else float("nan")
        ),
        "method": (
            "opd_full_ft" if not use_prefix else "ce_batched+opsd_kl"
        ),
        "student_update": engine.student_update,
        "n_tokens": icps.n_tokens,
        "ce_batch_size": icps.ce_batch_size,
        "opsd_batch_size": icps.opsd_batch_size,
        "use_vllm_base": bool(icps.use_vllm_base),
    }
    writer.write(summary)

    if wb is not None:
        final_step = global_step + max(1, len(problems))
        wb.log(
            {
                "global_step": final_step,
                "train/phase": 2,
                "eval/acc_base": summary["avg_acc_base"],
                "eval/acc_icps": summary["avg_acc_icps"],
                "eval/acc_selected": summary["avg_acc_selected"],
                "eval/delta_icps_pp": summary["delta_icps_vs_base_pp"],
                "eval/delta_selected_pp": summary["delta_selected_vs_base_pp"],
                "eval/n_fallback_base": n_fallback,
                "eval/n_problems": summary["n_problems"],
                "eval/mean_prefix_norm": mean_or_nan(norms),
            },
            step=final_step,
        )
    return summary
