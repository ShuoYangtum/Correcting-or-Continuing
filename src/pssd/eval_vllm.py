"""vLLM-backed downstream eval (Base / PSSD), training-free.

Mirrors OPSD ``eval/evaluate_math.py`` for solver sampling while keeping PSSD's
deployment path: greedy 8B rewrite → frozen 1.7B Avg@N (two-phase when the
rewriter and solver are different checkpoints).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# vLLM engine cores fork workers; CUDA must not be initialized under "fork".
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from transformers import AutoTokenizer

from pssd.answers import grade_output
from pssd.config import PSSDConfig
from pssd.data import Problem
from pssd.eta import ConditionResult, _load_mode_progress
from pssd.rewrite_gates import rewrite_problems_with_retries
from pssd.templates import (
    privileged_solver_messages,
    rewriter_messages,
    solver_messages,
    strip_think_blocks,
)
from pssd.utils import get_logger, mean_or_nan

logger = get_logger(__name__)


def sanitize_cuda_visible_devices() -> Optional[str]:
    """vLLM parses CUDA_VISIBLE_DEVICES as ints; Slurm may export a GPU UUID."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts or all(p.isdigit() for p in parts):
        return None
    remapped = ",".join(str(i) for i in range(len(parts)))
    os.environ["CUDA_VISIBLE_DEVICES"] = remapped
    logger.warning(
        "CUDA_VISIBLE_DEVICES=%r is not integer indices; remapped to %s",
        raw,
        remapped,
    )
    return remapped


def _vllm_top_k(top_k: int) -> int:
    """Map our config (0 = disabled) to vLLM (``-1`` = disabled)."""
    return -1 if top_k <= 0 else top_k


class VLLMEvalEngine:
    """Tokenizer + vLLM LLM (full weights only; no LoRA).

    Load the 8B rewriter or the 1.7B solver by passing the matching config /
    ``model_name_or_path``. Do not mix roles in one engine.
    """

    def __init__(
        self,
        cfg: PSSDConfig,
        adapter_path: str | Path | None = None,
        *,
        model_name_or_path: Optional[str] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        max_lora_rank: int = 64,
    ) -> None:
        from vllm import LLM

        del max_lora_rank  # LoRA removed; kept so older call sites still parse.
        sanitize_cuda_visible_devices()
        self.cfg = cfg
        if adapter_path is not None:
            adapter = Path(adapter_path)
            if (adapter / "adapter_config.json").exists():
                raise ValueError(
                    f"{adapter} is a LoRA adapter; pass a full rewriter checkpoint"
                )
            model_name_or_path = model_name_or_path or str(adapter)
        model_name = model_name_or_path or cfg.model.name_or_path

        if max_model_len is None:
            max_model_len = 40960 if cfg.model.enable_thinking else 32768

        logger.info(
            "Loading vLLM model=%s max_model_len=%d",
            model_name,
            max_model_len,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=cfg.model.trust_remote_code
        )
        self.llm = LLM(
            model=model_name,
            trust_remote_code=cfg.model.trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=False,
        )

    def render_chat(
        self,
        messages: Sequence[dict],
        *,
        enable_thinking: bool,
    ) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                list(messages), tokenize=False, add_generation_prompt=True
            )

    def render_solver_prompt(self, problem_or_rewrite: str) -> str:
        return self.render_chat(
            solver_messages(
                problem_or_rewrite,
                append_boxed_instruction=self.cfg.rollout.append_boxed_instruction,
            ),
            enable_thinking=self.cfg.model.enable_thinking,
        )

    def render_rewriter_prompt(self, problem: str) -> str:
        return self.render_chat(
            rewriter_messages(
                problem,
                two_stage=self.cfg.rewrite.two_stage,
                style=self.cfg.rewrite.style,
                few_shot=self.cfg.rewrite.few_shot,
            ),
            enable_thinking=self.cfg.rewrite.enable_thinking,
        )

    def render_privileged_prompt(self, problem: Problem) -> str:
        return self.render_chat(
            privileged_solver_messages(problem.problem, problem.reference_solution),
            enable_thinking=self.cfg.model.enable_thinking,
        )

    def _sampling_params(
        self,
        *,
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        greedy: bool = False,
    ):
        from vllm import SamplingParams

        if greedy:
            return SamplingParams(
                n=1,
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
        return SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=_vllm_top_k(top_k),
            min_p=0.0,
            presence_penalty=0.0,
            max_tokens=max_new_tokens,
        )

    def generate_texts(
        self,
        prompts: Sequence[str],
        *,
        n: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        use_lora: bool = False,
        greedy: bool = False,
    ) -> List[List[str]]:
        """Generate ``n`` continuations per prompt; returns [prompt][sample] texts."""
        if not prompts:
            return []
        params = self._sampling_params(
            n=n,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            greedy=greedy,
        )
        if use_lora:
            raise ValueError("LoRA is disabled; load full rewriter weights instead")
        outputs = self.llm.generate(list(prompts), params, use_tqdm=True)
        return [[o.text for o in out.outputs] for out in outputs]

    def sample_rewrite_texts(self, problem_text: str, k: int, *, use_lora: bool = False) -> List[str]:
        """Sample K rewriter texts (optional greedy-first), matching HF ``sample_rewrites``."""
        rw = self.cfg.rewrite
        prompt = self.render_rewriter_prompt(problem_text)
        texts: List[str] = []
        n_sample = k
        if rw.include_greedy and k >= 1:
            greedy = self.generate_texts(
                [prompt],
                n=1,
                max_new_tokens=rw.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                use_lora=use_lora,
                greedy=True,
            )
            texts.extend(greedy[0] if greedy else [])
            n_sample = k - 1
        if n_sample > 0:
            sampled = self.generate_texts(
                [prompt],
                n=n_sample,
                max_new_tokens=rw.max_new_tokens,
                temperature=rw.temperature,
                top_p=rw.top_p,
                top_k=0,
                use_lora=use_lora,
                greedy=False,
            )
            texts.extend(sampled[0] if sampled else [])
        return texts

    def greedy_rewrites(self, problems: Sequence[Problem]) -> Dict[str, str]:
        """Validated rewrite map: greedy then sampled retries; else original."""
        if not problems:
            return {}
        rw = self.cfg.rewrite
        prompts = [self.render_rewriter_prompt(p.problem) for p in problems]
        originals = [p.problem for p in problems]

        def generate_batch(indices: List[int], greedy: bool) -> List[str]:
            batch_prompts = [prompts[i] for i in indices]
            texts = self.generate_texts(
                batch_prompts,
                n=1,
                max_new_tokens=rw.max_new_tokens,
                temperature=0.0 if greedy else rw.temperature,
                top_p=1.0 if greedy else rw.top_p,
                top_k=0,
                use_lora=False,
                greedy=greedy,
            )
            return [t[0] if t else "" for t in texts]

        decisions = rewrite_problems_with_retries(originals, self.cfg, generate_batch)
        out: Dict[str, str] = {}
        n_fallback = 0
        for problem, decision in zip(problems, decisions):
            out[problem.id] = decision.text
            if decision.fell_back:
                n_fallback += 1
                logger.info(
                    "Rewrite fallback for %s after %d attempt(s): %s",
                    problem.id, decision.attempts, ",".join(decision.reasons),
                )
        if n_fallback:
            logger.info(
                "Rewrite gates: %d/%d problems fell back to original",
                n_fallback, len(problems),
            )
        return out

    def close(self) -> None:
        """Release the vLLM engine so another model can occupy the GPU."""
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        shutdown = getattr(llm, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                logger.warning("vLLM shutdown() failed; falling back to delete", exc_info=True)
        self.llm = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def evaluate_condition_vllm(
    engine: VLLMEvalEngine,
    problems: List[Problem],
    mode: str,
    n_samples: int,
    progress_path: Optional[Path] = None,
    chunk_size: int = 4,
    rewrites: Optional[Dict[str, str]] = None,
) -> ConditionResult:
    """Batched Avg@n accuracy under one prompting condition (vLLM).

    Processes ``chunk_size`` problems at a time so long (38k) generations stay
    within GPU KV budget and progress is flushed frequently for 48h chaining.
    """
    if mode not in {"base", "pssd", "privileged"}:
        raise ValueError(f"Unknown condition: {mode!r}")

    grader = engine.cfg.reward.grader
    resumed = _load_mode_progress(progress_path)
    if resumed:
        logger.info("[%s] resume: %d/%d problems already scored", mode, len(resumed), len(problems))

    pending = [p for p in problems if p.answer is not None and p.id not in resumed]
    rewrite_map: Dict[str, str] = dict(rewrites or {})
    ro = engine.cfg.rollout

    progress_fh = progress_path.open("a", encoding="utf-8") if progress_path else None
    try:
        for start in range(0, len(pending), max(1, chunk_size)):
            batch = pending[start : start + max(1, chunk_size)]
            if mode == "pssd":
                if rewrites is None:
                    batch_rewrites = engine.greedy_rewrites(batch)
                    rewrite_map.update(batch_rewrites)
                else:
                    batch_rewrites = {p.id: rewrite_map[p.id] for p in batch}
                prompts = [
                    engine.render_solver_prompt(batch_rewrites[p.id]) for p in batch
                ]
            elif mode == "base":
                prompts = [engine.render_solver_prompt(p.problem) for p in batch]
            else:
                prompts = [engine.render_privileged_prompt(p) for p in batch]

            logger.info(
                "[%s] vLLM chunk %d-%d / %d pending × Avg@%d (max_new_tokens=%d)",
                mode,
                start + 1,
                start + len(batch),
                len(pending),
                n_samples,
                ro.max_new_tokens,
            )
            samples = engine.generate_texts(
                prompts,
                n=n_samples,
                max_new_tokens=ro.max_new_tokens,
                temperature=ro.temperature,
                top_p=ro.top_p,
                top_k=ro.top_k,
                use_lora=False,
                greedy=False,
            )
            for problem, gens in zip(batch, samples):
                correct = []
                for text in gens:
                    grade_text = text if grader == "math_verify" else strip_think_blocks(text)
                    correct.append(grade_output(grade_text, problem.answer, grader=grader))
                acc = sum(correct) / max(1, len(correct))
                resumed[problem.id] = acc
                if progress_fh is not None:
                    row = {"id": problem.id, "acc": acc}
                    if mode == "pssd" and problem.id in rewrite_map:
                        row["rewrite"] = rewrite_map[problem.id]
                    progress_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress_fh.flush()
                logger.info(
                    "[%s] %s: avg@%d = %.3f",
                    mode, problem.id, n_samples, acc,
                )
    finally:
        if progress_fh is not None:
            progress_fh.close()

    # Reload rewrites from progress for dump / completeness after resume.
    if mode == "pssd" and progress_path is not None and progress_path.exists():
        with progress_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "rewrite" in row and row.get("id") is not None:
                    rewrite_map[str(row["id"])] = row["rewrite"]

    per_problem: List[float] = []
    for problem in problems:
        if problem.answer is None:
            continue
        if problem.id not in resumed:
            logger.warning("Missing score for %s after vLLM eval", problem.id)
            continue
        per_problem.append(resumed[problem.id])

    return ConditionResult(
        mode,
        mean_or_nan(per_problem),
        per_problem,
        rewrites=rewrite_map or None,
    )
