"""Prefix-tuning / ICPS train + eval entrypoint.

student_update='prefix': frozen generator; only the lightweight prefix
encoder is trained. student_update='full': no prefix; full-parameter OPD
on the student LM (carrier control).

Phases:
  1) Batched CE on reference solutions (encoder sees x only; never y*).
     Skipped for full-FT. EMA early-stop restores the best encoder.
  2) On-policy KL(p_T || p_S):
       distill='opd'  — larger frozen teacher, same solver prompt, never y*.
         Optional advantage gate: skip KL when the student's own rollout
         is already correct (gold is a binary skip, not teacher input).
       distill='opsd' — legacy same-model privileged teacher p_T(.|x, y*).
  3) Eval: original snapshot (vLLM) vs trained student (prefix or full-FT).

Usage:
    python scripts/run_icps_mvp.py --config configs/train_e2_opd_gated.yaml
    python scripts/run_icps_mvp.py --config configs/train_e2b_opd_full.yaml
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import _bootstrap  # noqa: F401
import torch
import yaml

from pssd.config import build_config
from pssd.data import Problem, load_problems
from pssd.icps import ICPSEngine, ICPSSettings
from pssd.icps_train import evaluate_icps, train_ce, train_opsd
from pssd.utils import JsonlWriter, ensure_dir, get_logger, set_seed

logger = get_logger("pssd.icps_mvp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefix-tuning / ICPS train + eval")
    p.add_argument("--config", required=True)
    p.add_argument("--train-data", default=None)
    p.add_argument("--eval-data", default=None)
    p.add_argument("--max-eval-problems", type=int, default=None)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument(
        "--skip-ce",
        action="store_true",
        help="Skip CE; load after_ce encoder and run OPD/OPSD + eval",
    )
    p.add_argument("--encoder-path", default=None)
    p.add_argument("--output-dir", default=None, help="Override train.output_dir")
    p.add_argument("--no-wandb", action="store_true", help="Disable wandb for this run")
    return p.parse_args()


def _load_icps_settings(section: Dict[str, Any]) -> tuple[ICPSSettings, str]:
    section = dict(section or {})
    train_path = str(section.pop("train_path", ""))
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"Unknown icps keys: {sorted(unknown)}; valid={sorted(known)}")
    if "train_problems" in section and section["train_problems"] in ("null", "None", ""):
        section["train_problems"] = None
    if "ce_steps" in section and section["ce_steps"] in ("null", "None", ""):
        section["ce_steps"] = None
    return ICPSSettings(**section), train_path


def _init_wandb(cfg, icps: ICPSSettings, train_path: str):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb logging is enabled but the wandb package is not installed"
        ) from exc
    from pssd.config import config_to_dict

    payload = config_to_dict(cfg)
    payload["icps"] = asdict(icps)
    payload["icps"]["train_path"] = train_path
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=payload,
        dir=str(Path(cfg.train.output_dir) / "wandb"),
    )
    wandb.define_metric("global_step")
    wandb.define_metric("ce/*", step_metric="global_step")
    wandb.define_metric("opsd/*", step_metric="global_step")
    wandb.define_metric("eval/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    logger.info("wandb run initialized: %s", getattr(run, "url", run.id))
    return wandb


def _eval_source(cfg, args: argparse.Namespace) -> str:
    return str(args.eval_data or cfg.data.eval_path or cfg.data.train_path)


def _load_eval_problems(path: str, max_n: Optional[int]) -> List[Problem]:
    problems = [p for p in load_problems(path, max_n) if p.answer]
    if not problems:
        raise SystemExit(f"No eval problems with gold answers in {path}")
    return problems


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    icps_section = dict(raw.pop("icps", {}) or {})
    cfg = build_config(raw)
    icps, train_path = _load_icps_settings(icps_section)
    if args.train_data:
        train_path = args.train_data
    if args.eval_data:
        cfg.data.eval_path = args.eval_data
    if args.max_eval_problems is not None:
        icps.max_eval_problems = args.max_eval_problems
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    if args.no_wandb:
        cfg.wandb.enabled = False
    # OPD keeps the 14B teacher on GPU during training. vLLM for base eval is
    # attached after release_teacher() at the end; do not force it off here.

    set_seed(cfg.train.seed)
    out_dir = ensure_dir(cfg.train.output_dir)
    wb = _init_wandb(cfg, icps, train_path)

    engine = ICPSEngine(cfg, icps)
    encoder_path = Path(args.encoder_path) if args.encoder_path else out_dir / "icps_encoder.pt"
    global_step = 0
    skip_ce = bool(args.skip_ce) or (not engine.uses_prefix())
    skip_opsd = bool(args.skip_train)
    if args.skip_train:
        skip_ce = True

    eval_path = _eval_source(cfg, args)
    eval_problems = _load_eval_problems(eval_path, icps.max_eval_problems)
    writer = JsonlWriter(Path(out_dir) / "icps_results.jsonl")
    logger.info(
        "Eval pool: %d problems from %s | midtrain every %d steps first_n=%d greedy=%s",
        len(eval_problems),
        eval_path,
        icps.eval_every,
        icps.eval_n_problems,
        icps.eval_greedy,
    )

    try:
        if skip_ce and not skip_opsd:
            if engine.uses_prefix():
                ce_ckpt = Path(args.encoder_path) if args.encoder_path else out_dir / "icps_encoder_after_ce.pt"
                if not ce_ckpt.exists():
                    raise SystemExit(f"after-CE encoder not found: {ce_ckpt}")
                engine.load_encoder(ce_ckpt)
                blob = torch.load(ce_ckpt, map_location="cpu", weights_only=False)
                global_step = int(blob.get("global_step", 0) or 0)
                logger.info("Skip CE; resume distill from %s (global_step=%d)", ce_ckpt, global_step)
            else:
                logger.info("Full-FT OPD: skip CE; start from instruct student weights")
        elif args.encoder_path or args.skip_train:
            if engine.uses_prefix():
                if not encoder_path.exists():
                    raise SystemExit(f"Encoder not found: {encoder_path}")
                engine.load_encoder(encoder_path)
                logger.info("Eval-only / resume encoder: %s", encoder_path)
            else:
                student_dir = Path(args.encoder_path) if args.encoder_path else out_dir / "student_opd"
                if not student_dir.exists():
                    raise SystemExit(f"Full-FT student not found: {student_dir}")
                engine.load_student(student_dir)
                logger.info("Eval-only / resume student: %s", student_dir)

        if not skip_ce:
            if not train_path:
                raise SystemExit("icps.train_path is required for training")
            train_problems = [
                p
                for p in load_problems(train_path, icps.train_problems)
                if p.reference_solution
            ]
            logger.info(
                "Prefix train: %d problems from %s | CE epochs=%d steps=%s batch=%d accum=%d | "
                "distill=%s steps=%d batch=%d | n_tokens=%d",
                len(train_problems),
                train_path,
                icps.ce_epochs,
                str(icps.ce_steps) if icps.ce_steps is not None else "full",
                icps.ce_batch_size,
                icps.ce_grad_accum,
                icps.distill,
                icps.opsd_steps,
                icps.opsd_batch_size,
                icps.n_tokens,
            )
            if wb is not None:
                wb.log(
                    {
                        "train/n_train_problems": len(train_problems),
                        "train/n_tokens": icps.n_tokens,
                        "train/encoder_params_m": sum(
                            p.numel() for p in engine.encoder.parameters()
                        )
                        / 1e6,
                        "train/ce_effective_batch": icps.ce_batch_size * icps.ce_grad_accum,
                        "train/opsd_effective_batch": icps.opsd_batch_size * icps.opsd_grad_accum,
                        "train/distill": icps.distill,
                    },
                    step=0,
                )

            n_ce, global_step = train_ce(
                engine,
                train_problems,
                icps.ce_epochs,
                wb=wb,
                global_step_start=0,
                checkpoint_dir=out_dir,
            )
            logger.info("CE finished: %d optimizer steps", n_ce)
            if n_ce < 1:
                raise RuntimeError("CE made 0 successful optimizer steps; aborting")
            engine.save_encoder(
                out_dir / "icps_encoder_after_ce.pt",
                extra={"phase": "ce_done", "global_step": global_step},
            )

        if not skip_opsd:
            if not train_path:
                raise SystemExit("icps.train_path is required for training")
            train_problems = [
                p
                for p in load_problems(train_path, icps.train_problems)
                if p.reference_solution
            ]
            if icps.distill == "opd" and bool(getattr(icps, "opsd_advantage_gate", False)):
                n_before = len(train_problems)
                train_problems = [p for p in train_problems if p.answer]
                if len(train_problems) < n_before:
                    logger.info(
                        "Advantage-gated OPD: dropped %d problems without gold answers",
                        n_before - len(train_problems),
                    )
            if icps.distill == "opd":
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                engine.ensure_teacher()
            logger.info(
                "%s train: %d problems from %s | student_update=%s | distill=%s "
                "steps=%d batch=%d accum=%d",
                "OPD" if icps.distill == "opd" else "OPSD",
                len(train_problems),
                train_path,
                engine.student_update,
                icps.distill,
                icps.opsd_steps,
                icps.opsd_batch_size,
                icps.opsd_grad_accum,
            )
            if wb is not None:
                n_train_p = sum(p.numel() for p in engine.trainable_parameters()) / 1e6
                wb.log(
                    {
                        "train/n_train_problems": len(train_problems),
                        "train/student_update": engine.student_update,
                        "train/trainable_params_m": n_train_p,
                        "train/opsd_effective_batch": icps.opsd_batch_size * icps.opsd_grad_accum,
                        "train/distill": icps.distill,
                    },
                    step=global_step,
                )
            n_opsd, global_step = train_opsd(
                engine,
                train_problems,
                icps.opsd_steps,
                wb=wb,
                global_step_start=global_step,
                checkpoint_dir=out_dir,
            )
            logger.info(
                "%s-KL finished: %d successful updates / %d steps",
                "OPD" if icps.distill == "opd" else "OPSD",
                n_opsd,
                icps.opsd_steps,
            )
            if engine.uses_prefix():
                engine.save_encoder(
                    encoder_path,
                    extra={"phase": "final", "global_step": global_step},
                )
            else:
                engine.save_student(
                    out_dir / "student_opd",
                    extra={"phase": "final", "global_step": global_step},
                )
            if wb is not None:
                wb.log({"train/encoder_saved": 1}, step=global_step)

        engine.release_teacher()
        logger.info(
            "Final eval: %d problems | distill=%s student_update=%s vllm_base=%s",
            len(eval_problems),
            icps.distill,
            engine.student_update,
            icps.use_vllm_base,
        )
        summary = evaluate_icps(
            engine, eval_problems, writer, wb=wb, global_step=global_step
        )
        logger.info("=" * 64)
        student_arm = "always prefix" if engine.uses_prefix() else "full-FT student"
        logger.info("ICPS summary (n=%d):", summary["n_problems"])
        logger.info("  base              : %.4f", summary["avg_acc_base"])
        logger.info(
            "  %s : %.4f  (Δ=%+.2fpp)",
            student_arm,
            summary["avg_acc_icps"],
            summary["delta_icps_vs_base_pp"],
        )
        if engine.uses_prefix():
            logger.info(
                "  ICPS+prefer_zero  : %.4f  (Δ=%+.2fpp)  fallback=%d/%d",
                summary["avg_acc_selected"],
                summary["delta_selected_vs_base_pp"],
                summary["n_fallback_base"],
                summary["n_problems"],
            )
        logger.info("Results: %s", Path(out_dir) / "icps_results.jsonl")
    finally:
        writer.close()
        if wb is not None:
            wb.finish()


if __name__ == "__main__":
    main()
