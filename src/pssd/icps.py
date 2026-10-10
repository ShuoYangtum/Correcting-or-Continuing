"""Instance-conditioned prefix tuning (frozen generator + lightweight encoder).

Frozen solver + small encoder: problem text ``x`` → ``L`` soft-prefix tokens.
No rewriter. Only encoder parameters are trained.

Training:
  1) CE on reference solutions (encoder sees ``x`` only; never ``y*``).
     Skipped when ``student_update='full'`` (no prefix to warm-start).
  2) Distill with token-level ``KL(p_T || p_S)``:
       - student_update='prefix' (paper method): student weights frozen;
         only the instance encoder is trained. Prefix is kept at inference.
       - student_update='full' (carrier control): no prefix; all student
         weights are trained. Same OPD KL, same teacher prompt (never y*).
       - distill='opd': larger frozen teacher, same solver-facing prompt,
         never sees ``y*``. Optional advantage gate: skip KL on already-
         correct student rollouts. Gold is a binary skip, not teacher input.
       - distill='opsd': legacy same-model teacher conditioned on ``(x, y*)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pssd.answers import grade_output
from pssd.config import PSSDConfig
from pssd.templates import (
    privileged_solver_messages,
    solver_messages,
    strip_think_blocks,
    teacher_prefix,
)
from pssd.utils import get_logger, mean_or_nan, resolve_device, resolve_dtype

logger = get_logger(__name__)

TokenIds = List[int]


def _per_token_kl(
    stu_logits: torch.Tensor,
    tea_logits: torch.Tensor,
    *,
    clip: float,
    topk: int = 0,
    reverse: bool = False,
) -> torch.Tensor:
    """Token-wise clipped KL.

    ``reverse=False``: ``KL(p_T || p_S)`` (mode-covering; teacher top-k).
    ``reverse=True``:  ``KL(p_S || p_T)`` (mode-seeking; student top-k).
    Canonical on-policy distillation uses reverse KL on student rollouts.
    """
    sl = stu_logits.float()
    tl = tea_logits.float()
    k = int(topk)
    if k > 0:
        k_use = min(k, sl.shape[-1], tl.shape[-1])
        if reverse:
            stu_k, idx = sl.topk(k_use, dim=-1)
            tea_k = tl.gather(-1, idx)
        else:
            tea_k, idx = tl.topk(k_use, dim=-1)
            stu_k = sl.gather(-1, idx)
        log_p_s = F.log_softmax(stu_k, dim=-1)
        log_p_t = F.log_softmax(tea_k, dim=-1)
        del tea_k, stu_k, idx
    else:
        log_p_s = F.log_softmax(sl, dim=-1)
        log_p_t = F.log_softmax(tl, dim=-1)
    if reverse:
        kl = (log_p_s.exp() * (log_p_s - log_p_t)).sum(dim=-1)
    else:
        kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1)
    return kl.clamp(max=clip)


def _chunked_forward_kl(
    stu_logits: torch.Tensor,
    tea_logits: torch.Tensor,
    *,
    clip: float,
    chunk: int,
    topk: int = 0,
    reverse: bool = False,
) -> torch.Tensor:
    """Mean clipped token KL. ``reverse=True`` → ``KL(p_S || p_T)``."""
    n_tok = int(stu_logits.shape[0])
    if n_tok < 1:
        return stu_logits.new_zeros(())
    chunk = max(1, int(chunk))
    acc = stu_logits.new_zeros(())
    for start in range(0, n_tok, chunk):
        kl = _per_token_kl(
            stu_logits[start : start + chunk],
            tea_logits[start : start + chunk],
            clip=clip,
            topk=topk,
            reverse=reverse,
        )
        acc = acc + kl.sum()
        del kl
    return acc / float(n_tok)


def _per_token_forward_kl(
    stu_logits: torch.Tensor,
    tea_logits: torch.Tensor,
    *,
    clip: float,
    topk: int = 0,
    reverse: bool = False,
) -> torch.Tensor:
    """Token-wise clipped KL → ``[T]``. ``reverse=True`` is ``KL(p_S || p_T)``."""
    return _per_token_kl(
        stu_logits, tea_logits, clip=clip, topk=topk, reverse=reverse
    )


def _detach_past(past: Any) -> Any:
    """Detach KV cache tensors so the next shard is truncated BPTT."""
    if past is None:
        return None
    if torch.is_tensor(past):
        return past.detach()
    if isinstance(past, (tuple, list)):
        seq = [_detach_past(p) for p in past]
        return type(past)(seq) if isinstance(past, list) else tuple(seq)
    layers = getattr(past, "layers", None)
    if layers is not None:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            vals = getattr(layer, "values", None)
            if torch.is_tensor(keys):
                layer.keys = keys.detach()
            if torch.is_tensor(vals):
                layer.values = vals.detach()
        return past
    key_cache = getattr(past, "key_cache", None)
    value_cache = getattr(past, "value_cache", None)
    if isinstance(key_cache, list) and isinstance(value_cache, list):
        past.key_cache = [t.detach() if torch.is_tensor(t) else t for t in key_cache]
        past.value_cache = [t.detach() if torch.is_tensor(t) else t for t in value_cache]
        return past
    return past


def _past_seq_len(past: Any) -> int:
    if past is None:
        return 0
    get_len = getattr(past, "get_seq_length", None)
    if callable(get_len):
        try:
            return int(get_len())
        except Exception:
            pass
    if hasattr(past, "to_legacy_cache"):
        legacy = past.to_legacy_cache()
        if legacy:
            return int(legacy[0][0].shape[-2])
    if isinstance(past, (tuple, list)) and past:
        return int(past[0][0].shape[-2])
    return 0


def _crop_past(past: Any, keep_len: int) -> Any:
    if past is None or keep_len < 0:
        return past
    crop = getattr(past, "crop", None)
    if callable(crop):
        crop(int(keep_len))
        return past
    return past


@dataclass
class ICPSSettings:
    """Hyper-parameters for ICPS encoder training and eval."""

    n_tokens: int = 32
    bottleneck: int = 512
    # Encoder input representation for the soft-prefix encoder:
    #   "embed"   = mean of STATIC input embeddings (legacy; weak, no context).
    #   "context" = masked mean of the frozen model's LAST hidden state over the
    #               solver prompt (contextual; C1). Single-layer input prefix.
    encoder_input: str = "embed"
    # ---- CE phase ----
    ce_epochs: int = 1
    ce_steps: Optional[int] = 1000  # max optimizer steps; None = full epoch(s)
    ce_early_stop_patience: int = 200  # 0 disables; steps without EMA improvement
    ce_early_stop_min_delta: float = 0.002
    ce_early_stop_min_steps: int = 300
    train_problems: Optional[int] = None  # None = all
    ce_batch_size: int = 1
    ce_grad_accum: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    max_prompt_tokens: int = 1536
    max_target_tokens: int = 16384
    # Mild L2 on soft-prefix embeddings (discourages unbounded growth).
    prefix_l2_coef: float = 1e-4
    # ---- OPSD-KL phase ----
    opsd_steps: int = 2000
    opsd_batch_size: int = 1
    opsd_grad_accum: int = 1
    opsd_max_new_tokens: int = 16384
    opsd_kl_clip: float = 5.0
    opsd_lr_scale: float = 0.25
    opsd_temperature: float = 1.0
    opsd_kl_chunk: int = 64  # rows per KL chunk
    opsd_kl_topk: int = 64  # teacher top-k (forward) / student top-k (reverse)
    # "forward" = KL(p_T || p_S); "reverse" = KL(p_S || p_T). Canonical OPD
    # is reverse KL on student rollouts.
    opsd_kl_direction: str = "forward"
    # ESR-style: KL only on the first N tokens of the (possibly full) student
    # rollout. 0 = use the whole yhat. Generation length is still
    # opsd_max_new_tokens so the advantage gate can grade a complete answer.
    opsd_kl_max_tokens: int = 0
    # Sequence shard for full-FT KL. 0 = one forward over the whole yhat
    # (prefix path). Full-parameter 16k needs >0 so each shard can backward
    # and free activations (truncated BPTT + gradient accumulation).
    opsd_kl_seq_chunk: int = 0
    opsd_early_stop_patience: int = 150  # 0 disables
    opsd_early_stop_min_delta: float = 0.003
    opsd_early_stop_min_steps: int = 200
    # ---- Advantage-gated OPD (novel; y* never enters the teacher prompt) ----
    # After the prefixed student samples yhat ~ p_S(.|prefix, x), grade yhat
    # against the gold final answer. If already correct, skip KL on that
    # trajectory (leave native CoT untouched). If wrong, apply the usual
    # on-policy KL(p_T(.|x) || p_S(.|prefix, x)) along yhat.
    # Gold is a binary skip, not a privileged teacher condition.
    opsd_advantage_gate: bool = False
    opsd_gate_mode: str = "student_wrong"  # {"student_wrong"}; extensible
    # "opd" = larger frozen teacher, same prompt as the student, no y*.
    # "opsd" = legacy same-model privileged teacher p_T(.|x, y*).
    distill: str = "opsd"
    # Where OPD writes the student policy:
    #   "prefix" = freeze the LM; train the instance-conditioned soft prefix.
    #   "full"   = no prefix; full-parameter update of the student LM.
    student_update: str = "prefix"
    # ---- Logging / checkpoints ----
    log_every: int = 10
    save_every: int = 500  # CE checkpoint interval (optimizer steps)
    opsd_save_every: int = 100  # OPSD checkpoint interval (optimizer steps)
    # ---- Eval ----
    prefer_zero_if_nonpositive: bool = True
    g_score: int = 1
    n_solve: int = 1
    max_eval_problems: int = 50
    # During CE/OPD: every N optimizer steps, greedy pass@1 on the first
    # ``eval_n_problems`` scored eval items in file order (0 disables).
    eval_every: int = 0
    eval_n_problems: int = 10
    eval_max_new_tokens: Optional[int] = None  # None → rollout.eval_max or max_new
    eval_greedy: bool = True
    # vLLM for base-only (no soft prefix) generations.
    use_vllm_base: bool = True
    vllm_gpu_memory_utilization: float = 0.45
    vllm_max_model_len: int = 20480


class ICPSEncoder(nn.Module):
    """Mean-pooled problem embedding → ``L`` soft tokens (small-scale init)."""

    def __init__(
        self,
        hidden_size: int,
        n_tokens: int,
        bottleneck: int,
        output_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_tokens = n_tokens
        self.output_scale = float(output_scale)
        self.norm = nn.LayerNorm(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, n_tokens * hidden_size),
        )
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """``pooled``: ``[H]`` or ``[B, H]`` → ``[L, H]`` or ``[B, L, H]``."""
        single = pooled.dim() == 1
        if single:
            pooled = pooled.unsqueeze(0)
        pooled = self.norm(pooled.float())
        out = self.net(pooled).view(pooled.shape[0], self.n_tokens, self.hidden_size)
        out = out * self.output_scale
        return out[0] if single else out


class ICPSEngine:
    """Frozen causal LM + instance soft-prefix encoder (+ optional vLLM for base)."""

    def __init__(self, cfg: PSSDConfig, icps: ICPSSettings) -> None:
        self.cfg = cfg
        self.icps = icps
        self.device = resolve_device(cfg.model.device)
        self.dtype = resolve_dtype(cfg.model.dtype, self.device)
        self._vllm = None

        update = str(getattr(icps, "student_update", "prefix") or "prefix").lower()
        if update not in ("prefix", "full"):
            raise ValueError(
                f"icps.student_update must be 'prefix' or 'full', got {update!r}"
            )
        self.student_update = update
        logger.info(
            "ICPS: loading model=%s device=%s dtype=%s n_tokens=%d student_update=%s",
            cfg.model.name_or_path,
            self.device,
            self.dtype,
            icps.n_tokens,
            self.student_update,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name_or_path, trust_remote_code=cfg.model.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = dict(trust_remote_code=cfg.model.trust_remote_code)
        if cfg.model.attn_implementation:
            model_kwargs["attn_implementation"] = cfg.model.attn_implementation
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name_or_path, dtype=self.dtype, **model_kwargs
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                cfg.model.name_or_path, torch_dtype=self.dtype, **model_kwargs
            )
        self.model.to(self.device)
        freeze_student = self.student_update != "full"
        for p in self.model.parameters():
            p.requires_grad_(not freeze_student)
        if freeze_student:
            self.model.eval()
        else:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
            self.model.train()

        hidden = int(self.model.config.hidden_size)
        self.encoder = ICPSEncoder(hidden, icps.n_tokens, icps.bottleneck).to(
            device=self.device, dtype=torch.float32
        )
        if freeze_student:
            n_train = sum(p.numel() for p in self.encoder.parameters())
            logger.info("ICPS encoder trainable params: %.2fM", n_train / 1e6)
        else:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.info(
                "Full-FT OPD: student unfrozen (%.2fM params); prefix encoder unused",
                n_train / 1e6,
            )

        self.teacher = None
        self.teacher_tokenizer = None
        distill = str(getattr(icps, "distill", "opsd") or "opsd").lower()
        if distill not in ("opd", "opsd"):
            raise ValueError(f"icps.distill must be 'opd' or 'opsd', got {distill!r}")
        self.distill = distill

    def uses_prefix(self) -> bool:
        return self.student_update != "full"

    def kl_reverse(self) -> bool:
        d = str(getattr(self.icps, "opsd_kl_direction", "forward") or "forward").lower()
        if d not in ("forward", "reverse"):
            raise ValueError(f"icps.opsd_kl_direction must be 'forward' or 'reverse', got {d!r}")
        return d == "reverse"

    def trainable_parameters(self) -> List[nn.Parameter]:
        if self.uses_prefix():
            return [p for p in self.encoder.parameters() if p.requires_grad]
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------ #
    # Token / embed helpers
    # ------------------------------------------------------------------ #

    def _apply_chat(self, tokenizer, problem_text: str) -> TokenIds:
        msgs = solver_messages(
            problem_text,
            append_boxed_instruction=self.cfg.rollout.append_boxed_instruction,
        )
        try:
            ids = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=self.cfg.model.enable_thinking,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True
            )
        ids = list(ids)
        cap = int(self.icps.max_prompt_tokens)
        if cap > 0 and len(ids) > cap:
            ids = ids[:cap]
        return ids

    def chat_ids(self, problem_text: str) -> TokenIds:
        return self._apply_chat(self.tokenizer, problem_text)

    def teacher_chat_ids(self, problem_text: str) -> TokenIds:
        tok = self.teacher_tokenizer or self.tokenizer
        return self._apply_chat(tok, problem_text)

    def ensure_teacher(self) -> None:
        """Load the frozen OPD teacher (14B). No-op for legacy same-model OPSD."""
        if self.distill != "opd":
            return
        if self.teacher is not None:
            return
        path = str(self.cfg.model.teacher_name_or_path or "").strip()
        if not path:
            raise RuntimeError("distill='opd' requires model.teacher_name_or_path")
        logger.info("Loading frozen OPD teacher (no y*): %s", path)
        self.teacher_tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=self.cfg.model.trust_remote_code
        )
        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token
        model_kwargs = dict(trust_remote_code=self.cfg.model.trust_remote_code)
        if self.cfg.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.model.attn_implementation
        try:
            teacher = AutoModelForCausalLM.from_pretrained(
                path, dtype=self.dtype, **model_kwargs
            )
        except TypeError:
            teacher = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=self.dtype, **model_kwargs
            )
        teacher.to(self.device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        self.teacher = teacher
        stu_vocab = int(getattr(self.model.config, "vocab_size", 0) or 0)
        tea_vocab = int(getattr(teacher.config, "vocab_size", 0) or 0)
        if stu_vocab and tea_vocab and stu_vocab != tea_vocab:
            logger.warning(
                "OPD teacher vocab_size=%d != student vocab_size=%d; "
                "token-aligned KL may skip mismatched y hat",
                tea_vocab,
                stu_vocab,
            )
        logger.info("OPD teacher ready on %s dtype=%s", self.device, self.dtype)

    def release_teacher(self) -> None:
        """Free the 14B OPD teacher so eval / vLLM can use the GPU."""
        if self.teacher is None and self.teacher_tokenizer is None:
            return
        logger.info("Releasing OPD teacher from GPU")
        try:
            del self.teacher
        except Exception:
            pass
        self.teacher = None
        self.teacher_tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(self, text: str) -> TokenIds:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def encode_with(self, tokenizer, text: str) -> TokenIds:
        return tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def _embed_ids(self, ids: TokenIds) -> torch.Tensor:
        h = int(self.model.config.hidden_size)
        if not ids:
            return torch.zeros(0, h, device=self.device, dtype=self.dtype)
        t = torch.tensor(ids, dtype=torch.long, device=self.device)
        with torch.no_grad():
            emb = self.model.get_input_embeddings()(t)
        return emb.detach()

    def pool_problem(self, problem_text: str) -> torch.Tensor:
        return self.pool_problems([problem_text])[0]

    def pool_problems(self, problem_texts: Sequence[str]) -> torch.Tensor:
        """Mean-pool token embeddings for each problem → ``[B, H]`` (float32)."""
        h = int(self.model.config.hidden_size)
        if not problem_texts:
            return torch.zeros(0, h, device=self.device, dtype=torch.float32)

        id_lists: List[TokenIds] = []
        for text in problem_texts:
            ids = self.encode(text)
            if not ids:
                ids = [self.tokenizer.pad_token_id or 0]
            id_lists.append(ids)

        max_len = max(len(x) for x in id_lists)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        batch_ids = torch.full(
            (len(id_lists), max_len), pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros(len(id_lists), max_len, dtype=torch.float32, device=self.device)
        for i, ids in enumerate(id_lists):
            batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[i, : len(ids)] = 1.0

        with torch.no_grad():
            emb = self.model.get_input_embeddings()(batch_ids).float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def context_pool_problems(self, problem_texts: Sequence[str]) -> torch.Tensor:
        """Masked mean of the frozen model's LAST hidden state over the solver
        prompt (contextual representation; C1). Returns ``[B, H]`` float32.

        Unlike ``pool_problems`` (mean of static input embeddings), this runs the
        problem through the frozen transformer, so the pooled vector reflects the
        model's contextual understanding of the problem.
        """
        h = int(self.model.config.hidden_size)
        if not problem_texts:
            return torch.zeros(0, h, device=self.device, dtype=torch.float32)

        id_lists: List[TokenIds] = []
        for text in problem_texts:
            ids = self.chat_ids(text)  # solver-prompt view of the problem
            if not ids:
                ids = [self.tokenizer.pad_token_id or 0]
            id_lists.append(ids)

        max_len = max(len(x) for x in id_lists)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        batch_ids = torch.full(
            (len(id_lists), max_len), pad_id, dtype=torch.long, device=self.device
        )
        mask = torch.zeros(len(id_lists), max_len, dtype=torch.float32, device=self.device)
        for i, ids in enumerate(id_lists):
            batch_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            mask[i, : len(ids)] = 1.0

        with torch.no_grad():
            out = self.model(
                input_ids=batch_ids,
                attention_mask=mask.long(),
                output_hidden_states=True,
                use_cache=False,
            )
            last = out.hidden_states[-1].float()  # [B, T, H]
            del out
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (last * mask.unsqueeze(-1)).sum(dim=1) / denom
        return pooled.detach()

    def _pool_for_encoder(self, problem_texts: Sequence[str]) -> torch.Tensor:
        if str(getattr(self.icps, "encoder_input", "embed")).lower() == "context":
            return self.context_pool_problems(problem_texts)
        return self.pool_problems(problem_texts)

    def soft_prefix(self, problem_text: str) -> torch.Tensor:
        return self.encoder(self._pool_for_encoder([problem_text])[0])

    def soft_prefixes(self, problem_texts: Sequence[str]) -> torch.Tensor:
        """Batched soft prefixes → ``[B, L, H]``."""
        return self.encoder(self._pool_for_encoder(problem_texts))

    def prefix_norm(self, soft: torch.Tensor) -> float:
        t = soft.detach().float()
        if not torch.isfinite(t).all():
            return float("nan")
        # Mean Frobenius norm per example when batched.
        if t.dim() == 3:
            return float(t.flatten(1).norm(dim=1).mean().item())
        return float(t.norm().item())

    def _finite_soft(self, soft: torch.Tensor) -> torch.Tensor:
        soft = soft.float()
        if not torch.isfinite(soft).all():
            return torch.zeros_like(soft)
        return soft.clamp(-5.0, 5.0)

    def _prompt_embeds(self, problem_text: str) -> torch.Tensor:
        return self._embed_ids(self.chat_ids(problem_text))

    def set_model_dtype(self, dtype: torch.dtype) -> None:
        self.model.to(dtype=dtype)
        self.dtype = dtype
        logger.info("ICPS frozen model dtype -> %s", dtype)

    def encoder_is_finite(self) -> bool:
        params = self.trainable_parameters()
        if not params:
            return True
        return all(torch.isfinite(p).all() for p in params)

    # ------------------------------------------------------------------ #
    # CE / OPSD losses
    # ------------------------------------------------------------------ #

    def nll_on_reference(self, problem_text: str, reference: str) -> torch.Tensor:
        loss, _ = self.nll_on_reference_batch([problem_text], [reference])
        return loss

    def nll_on_reference_batch(
        self,
        problem_texts: Sequence[str],
        references: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Token-weighted CE on references under soft prefixes (batched).

        Sequences are right-padded. Soft tokens carry gradients; prompt/target
        token embeddings are detached. Empty targets are dropped from the
        token average (they contribute zero to the numerator and denominator).
        """
        if len(problem_texts) != len(references):
            raise ValueError("problem_texts and references must have the same length")
        if not problem_texts:
            z = torch.zeros((), device=self.device, dtype=torch.float32)
            return z, {"n_tokens": 0.0, "n_examples": 0.0, "prefix_norm": float("nan")}

        soft = self.soft_prefixes(list(problem_texts))  # [B, L, H]
        model_dtype = next(self.model.parameters()).dtype
        l_soft = int(soft.shape[1])
        h = int(soft.shape[2])
        bsz = int(soft.shape[0])

        prompt_embs: List[torch.Tensor] = []
        target_ids_list: List[TokenIds] = []
        target_embs: List[torch.Tensor] = []
        for text, ref in zip(problem_texts, references):
            prompt_embs.append(self._prompt_embeds(text))
            tids = self.encode(ref)[: self.icps.max_target_tokens]
            target_ids_list.append(tids)
            target_embs.append(
                self._embed_ids(tids) if tids else soft.new_zeros(0, h)
            )

        soft_m = soft.to(dtype=model_dtype)
        seqs: List[torch.Tensor] = []
        lengths: List[int] = []
        for i in range(bsz):
            pe = prompt_embs[i].to(dtype=model_dtype)
            te = target_embs[i].to(dtype=model_dtype)
            # soft_m[i] must stay in the graph; prompt/target embeds are detached.
            seq = torch.cat([soft_m[i], pe, te], dim=0)
            seqs.append(seq)
            lengths.append(int(seq.shape[0]))

        max_len = max(lengths)
        padded: List[torch.Tensor] = []
        attn = torch.zeros(bsz, max_len, dtype=torch.long, device=self.device)
        for i, seq in enumerate(seqs):
            n = lengths[i]
            if n < max_len:
                pad = seq.new_zeros(max_len - n, h)
                seq = torch.cat([seq, pad], dim=0)
            padded.append(seq)
            attn[i, :n] = 1
        inputs = torch.stack(padded, dim=0)

        max_tgt = max((len(t) for t in target_ids_list), default=0)
        loss_pos = torch.full(
            (bsz, max_tgt), -1, dtype=torch.long, device=self.device
        )
        tgt_pad = torch.zeros(bsz, max_tgt, dtype=torch.long, device=self.device)
        for i in range(bsz):
            p_len = l_soft + int(prompt_embs[i].shape[0])
            for j in range(len(target_ids_list[i])):
                loss_pos[i, j] = p_len - 1 + j
                tgt_pad[i, j] = target_ids_list[i][j]

        self.model.eval()
        # Keep logits in model dtype; only cast gathered target rows to fp32.
        logits = self.model(
            inputs_embeds=inputs,
            attention_mask=attn,
            use_cache=False,
        ).logits

        token_nll_sum = soft.new_zeros(())
        n_tokens = 0
        for i in range(bsz):
            n_tgt = len(target_ids_list[i])
            if n_tgt < 1:
                continue
            pos = loss_pos[i, :n_tgt]
            span = logits[i, pos, :].float()
            logprobs = F.log_softmax(span, dim=-1)
            tgt = tgt_pad[i, :n_tgt]
            nll = -logprobs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            token_nll_sum = token_nll_sum + nll.sum()
            n_tokens += n_tgt
        del logits

        if n_tokens == 0:
            # Keep a graph to the encoder so skipped batches stay safe.
            loss = soft.float().sum() * 0.0
        else:
            loss = token_nll_sum / float(n_tokens)

        if self.icps.prefix_l2_coef > 0:
            loss = loss + float(self.icps.prefix_l2_coef) * soft.float().pow(2).mean()

        if not torch.isfinite(loss):
            loss = soft.float().sum() * 0.0

        stats = {
            "n_tokens": float(n_tokens),
            "n_examples": float(bsz),
            "n_nonempty": float(sum(1 for t in target_ids_list if t)),
            "prefix_norm": self.prefix_norm(soft),
            "mean_seq_len": float(sum(lengths) / max(1, bsz)),
        }
        return loss, stats

    def opsd_kl_loss(self, problem_text: str, reference: str) -> torch.Tensor:
        loss, _ = self.opsd_kl_loss_batch([problem_text], [reference])
        return loss

    def opsd_kl_loss_batch(
        self,
        problem_texts: Sequence[str],
        references: Sequence[str],
        answers: Optional[Sequence[Optional[str]]] = None,
        *,
        grad_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Mean clipped token ``KL(p_T || p_S)`` over a mini-batch."""
        if self.distill == "opd":
            return self.opd_kl_loss_batch(
                problem_texts, answers=answers, grad_scale=grad_scale
            )
        return self._opsd_privileged_kl_loss_batch(problem_texts, references)

    def _grade_rollout(self, yhat: str, gold: Optional[str]) -> Optional[float]:
        """Grade a student rollout against the gold final answer. ``None`` if ungradeable."""
        if gold is None or not str(gold).strip():
            return None
        grader = self.cfg.reward.grader
        grade_text = yhat if grader == "math_verify" else strip_think_blocks(yhat)
        try:
            return float(grade_output(grade_text, gold, grader=grader))
        except Exception:
            return None

    def opd_kl_loss_batch(
        self,
        problem_texts: Sequence[str],
        answers: Optional[Sequence[Optional[str]]] = None,
        *,
        grad_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """On-policy ``KL(p_T(.|x) || p_S(.|z, x))``. Teacher never sees y*.

        If ``opsd_advantage_gate`` is on, gold answers gate *whether* to distill
        a trajectory (skip when the student is already correct). They are never
        concatenated into the teacher prompt.

        ``student_update='full'`` drops the prefix: sample and score
        ``p_S(.|x)``, then backprop the same token KL into student weights.
        """
        self.ensure_teacher()
        if self.teacher is None:
            raise RuntimeError("OPD teacher is not loaded")
        if not problem_texts:
            z = torch.zeros((), device=self.device, dtype=torch.float32)
            return z, {
                "n_tokens": 0.0,
                "n_examples": 0.0,
                "n_ok": 0.0,
                "n_gated": 0.0,
                "n_ungradeable": 0.0,
                "prefix_norm": float("nan"),
                "backward_done": 0.0,
            }

        use_prefix = self.uses_prefix()
        soft = self.soft_prefixes(list(problem_texts)) if use_prefix else None
        model_dtype = next(self.model.parameters()).dtype
        per_losses: List[torch.Tensor] = []
        n_tokens_total = 0
        n_ok = 0
        n_gated = 0
        n_ungradeable = 0
        gate = bool(getattr(self.icps, "opsd_advantage_gate", False))
        gate_mode = str(getattr(self.icps, "opsd_gate_mode", "student_wrong") or "student_wrong").lower()
        k = max(0, int(getattr(self.icps, "opsd_kl_topk", 0)))
        chunk = max(8, int(self.icps.opsd_kl_chunk))
        clip = float(self.icps.opsd_kl_clip)
        tea_tok = self.teacher_tokenizer or self.tokenizer

        self.model.eval()
        with torch.no_grad():
            yhats = self.generate_texts_batch(
                list(problem_texts),
                use_icps=use_prefix,
                soft_override=None if soft is None else soft.detach(),
                max_new_tokens=self.icps.opsd_max_new_tokens,
                temperature=self.icps.opsd_temperature,
            )
        if not use_prefix:
            self.model.train()

        seq_chunk = max(0, int(getattr(self.icps, "opsd_kl_seq_chunk", 0) or 0))
        use_shard = (not use_prefix) and seq_chunk > 0
        prepared: List[Tuple[str, TokenIds, TokenIds, TokenIds]] = []

        for i, text in enumerate(problem_texts):
            yhat = (yhats[i] if i < len(yhats) else "").strip()
            yhat_ids = self.encode(yhat)[: self.icps.opsd_max_new_tokens]
            yhat_ids_t = self.encode_with(tea_tok, yhat)[: self.icps.opsd_max_new_tokens]
            if len(yhat_ids) < 1 or len(yhat_ids_t) < 1:
                continue

            if gate:
                gold = answers[i] if (answers is not None and i < len(answers)) else None
                reward = self._grade_rollout(yhat, gold)
                if reward is None:
                    n_ungradeable += 1
                    continue
                if gate_mode == "student_wrong" and reward >= 1.0:
                    n_gated += 1
                    continue

            kl_cap = max(0, int(getattr(self.icps, "opsd_kl_max_tokens", 0) or 0))
            if kl_cap > 0:
                yhat_ids = yhat_ids[:kl_cap]
                yhat_ids_t = yhat_ids_t[:kl_cap]
                if len(yhat_ids) < 1 or len(yhat_ids_t) < 1:
                    continue

            if len(yhat_ids) != len(yhat_ids_t):
                continue
            t_prefix_ids = self.teacher_chat_ids(text)
            if not t_prefix_ids:
                continue

            if use_shard:
                prepared.append((text, yhat_ids, yhat_ids_t, t_prefix_ids))
                continue

            with torch.no_grad():
                t_ids = t_prefix_ids + yhat_ids_t
                t_input = torch.tensor([t_ids], dtype=torch.long, device=self.device)
                tea_out = self.teacher(
                    input_ids=t_input,
                    attention_mask=torch.ones_like(t_input),
                    use_cache=False,
                ).logits
                tea_span = tea_out[
                    0,
                    len(t_prefix_ids) - 1 : len(t_prefix_ids) - 1 + len(yhat_ids_t),
                    :,
                ]
                if tea_span.shape[0] != len(yhat_ids_t):
                    continue
                del tea_out

            stu_span = self._student_yhat_logits(
                text,
                yhat_ids,
                None if soft is None else soft[i],
                model_dtype,
            )
            if stu_span is None:
                continue

            kl = _chunked_forward_kl(
                stu_span,
                tea_span,
                clip=clip,
                chunk=chunk,
                topk=k,
                reverse=self.kl_reverse(),
            )
            del stu_span, tea_span

            per_losses.append(kl)
            n_tokens_total += len(yhat_ids)
            n_ok += 1

        backward_done = False
        if use_shard:
            n_ok = 0
            scale = float(grad_scale) / float(max(1, len(prepared)))
            shard_losses: List[torch.Tensor] = []
            for text, yhat_ids, yhat_ids_t, t_prefix_ids in prepared:
                mean_kl = self._opd_kl_sharded_example(
                    text,
                    yhat_ids,
                    yhat_ids_t,
                    t_prefix_ids,
                    seq_chunk=seq_chunk,
                    topk=k,
                    clip=clip,
                    grad_scale=scale,
                    reverse=self.kl_reverse(),
                )
                if mean_kl is None:
                    continue
                shard_losses.append(mean_kl)
                n_tokens_total += len(yhat_ids)
                n_ok += 1
            if shard_losses:
                loss = torch.stack(shard_losses).mean()
                backward_done = True
            else:
                loss = self._zero_trainable_loss()
        elif not per_losses:
            loss = self._zero_trainable_loss()
        else:
            loss = torch.stack(per_losses).mean()
            if use_prefix and soft is not None and self.icps.prefix_l2_coef > 0:
                loss = loss + float(self.icps.prefix_l2_coef) * soft.float().pow(2).mean()

        if not torch.isfinite(loss):
            loss = self._zero_trainable_loss()
            backward_done = False

        n_graded = n_gated + n_ok
        stats = {
            "n_tokens": float(n_tokens_total),
            "n_examples": float(len(problem_texts)),
            "n_ok": float(n_ok),
            "n_gated": float(n_gated),
            "n_ungradeable": float(n_ungradeable),
            "prefix_norm": (
                self.prefix_norm(soft) if soft is not None else float("nan")
            ),
            "gate_frac": float(n_gated) / float(max(1, len(problem_texts))),
            "onpolicy_acc": (float(n_gated) / float(n_graded) if n_graded > 0 else float("nan")),
            "backward_done": float(backward_done),
        }
        return loss, stats

    def _zero_trainable_loss(self) -> torch.Tensor:
        params = self.trainable_parameters()
        if params:
            return params[0].float().sum() * 0.0
        return torch.zeros((), device=self.device, dtype=torch.float32)

    def _lm_chunk_forward(
        self,
        model: nn.Module,
        input_ids: TokenIds,
        past: Any,
        *,
        use_grad: bool,
    ) -> Tuple[torch.Tensor, Any]:
        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        past_len = _past_seq_len(past)
        attn = torch.ones(
            1, past_len + int(ids.shape[1]), dtype=torch.long, device=self.device
        )
        ctx = torch.enable_grad() if use_grad else torch.no_grad()
        with ctx:
            out = model(
                input_ids=ids,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
        logits = out.logits[0]
        new_past = out.past_key_values
        del out
        return logits, new_past

    def _opd_kl_sharded_example(
        self,
        text: str,
        yhat_ids: TokenIds,
        yhat_ids_t: TokenIds,
        t_prefix_ids: TokenIds,
        *,
        seq_chunk: int,
        topk: int,
        clip: float,
        grad_scale: float,
        reverse: bool = False,
    ) -> Optional[torch.Tensor]:
        """Full-rollout KL in sequence shards; backward each shard (truncated BPTT)."""
        prompt_ids = self.chat_ids(text)
        t_len = len(yhat_ids)
        if len(prompt_ids) < 1 or t_len < 1:
            return None
        chunk = max(1, int(seq_chunk))
        was_gc = bool(getattr(self.model, "is_gradient_checkpointing", False))
        if was_gc:
            self.model.gradient_checkpointing_disable()
        if hasattr(self.model, "config"):
            self.model.config.use_cache = True
        stu_past: Any = None
        tea_past: Any = None
        kl_sum = 0.0
        try:
            for start in range(0, t_len, chunk):
                end = min(t_len, start + chunk)
                n_need = end - start
                if start == 0:
                    stu_in = prompt_ids + yhat_ids[:end]
                    tea_in = t_prefix_ids + yhat_ids_t[:end]
                    stu_logits, stu_past = self._lm_chunk_forward(
                        self.model, stu_in, None, use_grad=True
                    )
                    tea_logits, tea_past = self._lm_chunk_forward(
                        self.teacher, tea_in, None, use_grad=False
                    )
                    p = len(prompt_ids)
                    tp = len(t_prefix_ids)
                    stu_span = stu_logits[p - 1 : p - 1 + n_need]
                    tea_span = tea_logits[tp - 1 : tp - 1 + n_need]
                else:
                    stu_past = _crop_past(
                        _detach_past(stu_past), _past_seq_len(stu_past) - 1
                    )
                    tea_past = _crop_past(
                        _detach_past(tea_past), _past_seq_len(tea_past) - 1
                    )
                    stu_logits, stu_past = self._lm_chunk_forward(
                        self.model, yhat_ids[start - 1 : end], stu_past, use_grad=True
                    )
                    tea_logits, tea_past = self._lm_chunk_forward(
                        self.teacher,
                        yhat_ids_t[start - 1 : end],
                        tea_past,
                        use_grad=False,
                    )
                    stu_span = stu_logits[:n_need]
                    tea_span = tea_logits[:n_need]
                if stu_span.shape[0] != n_need or tea_span.shape[0] != n_need:
                    return None
                kl_tok = _per_token_forward_kl(
                    stu_span, tea_span, clip=clip, topk=topk, reverse=reverse
                )
                piece = kl_tok.sum() / float(t_len) * float(grad_scale)
                piece.backward()
                kl_sum += float(kl_tok.detach().sum().item())
                del stu_logits, tea_logits, stu_span, tea_span, kl_tok, piece
                stu_past = _detach_past(stu_past)
                tea_past = _detach_past(tea_past)
        finally:
            if was_gc:
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        return torch.tensor(kl_sum / float(t_len), device=self.device, dtype=torch.float32)

    def _student_yhat_logits(
        self,
        text: str,
        yhat_ids: TokenIds,
        soft_i: Optional[torch.Tensor],
        model_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Student next-token logits over ``yhat_ids``. Graph hits prefix or weights."""
        if self.uses_prefix():
            if soft_i is None:
                return None
            yhat_emb = self._embed_ids(yhat_ids)
            prompt_emb = self._prompt_embeds(text)
            soft_m = soft_i.to(dtype=model_dtype)
            stu_inputs = torch.cat(
                [soft_m, prompt_emb.to(dtype=model_dtype), yhat_emb.to(dtype=model_dtype)],
                dim=0,
            ).unsqueeze(0)
            stu_attn = torch.ones(stu_inputs.shape[:2], dtype=torch.long, device=self.device)
            self.model.eval()
            stu_out = self.model(
                inputs_embeds=stu_inputs,
                attention_mask=stu_attn,
                use_cache=False,
            ).logits
            stu_prefix = int(soft_i.shape[0] + prompt_emb.shape[0])
            stu_span = stu_out[
                0, stu_prefix - 1 : stu_prefix - 1 + len(yhat_ids), :
            ]
            del stu_out
            if stu_span.shape[0] != len(yhat_ids):
                return None
            return stu_span

        prompt_ids = self.chat_ids(text)
        if len(prompt_ids) < 1:
            return None
        stu_ids = prompt_ids + list(yhat_ids)
        stu_input = torch.tensor([stu_ids], dtype=torch.long, device=self.device)
        stu_out = self.model(
            input_ids=stu_input,
            attention_mask=torch.ones_like(stu_input),
            use_cache=False,
        ).logits
        stu_span = stu_out[
            0, len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(yhat_ids), :
        ]
        del stu_out
        if stu_span.shape[0] != len(yhat_ids):
            return None
        return stu_span

    def _opsd_privileged_kl_loss_batch(
        self,
        problem_texts: Sequence[str],
        references: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Legacy same-model privileged teacher ``p_T(.|x, y*)``."""
        if len(problem_texts) != len(references):
            raise ValueError("problem_texts and references must have the same length")
        if not problem_texts:
            z = torch.zeros((), device=self.device, dtype=torch.float32)
            return z, {"n_tokens": 0.0, "n_examples": 0.0, "prefix_norm": float("nan")}

        soft = self.soft_prefixes(list(problem_texts))
        model_dtype = next(self.model.parameters()).dtype
        per_losses: List[torch.Tensor] = []
        n_tokens_total = 0
        n_ok = 0

        for i, (text, ref) in enumerate(zip(problem_texts, references)):
            soft_i = soft[i]
            with torch.no_grad():
                texts = self.generate_texts(
                    text,
                    n=1,
                    use_icps=True,
                    soft_override=soft_i.detach(),
                    max_new_tokens=self.icps.opsd_max_new_tokens,
                    temperature=self.icps.opsd_temperature,
                )
            yhat = (texts[0] if texts else "").strip()
            yhat_ids = self.encode(yhat)[: self.icps.opsd_max_new_tokens]
            if len(yhat_ids) < 1:
                continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            yhat_emb = self._embed_ids(yhat_ids)
            prompt_emb = self._prompt_embeds(text)
            t_prefix_ids = self.encode(teacher_prefix(text, ref))
            if not t_prefix_ids:
                continue

            k = max(0, int(getattr(self.icps, "opsd_kl_topk", 0)))
            chunk = max(8, int(self.icps.opsd_kl_chunk))
            clip = float(self.icps.opsd_kl_clip)

            # Teacher first: keep only top-k support (or compact span if topk=0).
            with torch.no_grad():
                t_ids = t_prefix_ids + yhat_ids
                t_input = torch.tensor([t_ids], dtype=torch.long, device=self.device)
                tea_out = self.model(
                    input_ids=t_input,
                    attention_mask=torch.ones_like(t_input),
                    use_cache=False,
                ).logits
                tea_span = tea_out[
                    0, len(t_prefix_ids) - 1 : len(t_prefix_ids) - 1 + len(yhat_ids), :
                ]
                if tea_span.shape[0] != len(yhat_ids):
                    continue
                tea_k = None
                tea_idx = None
                if k > 0:
                    k_use = min(k, int(tea_span.shape[-1]))
                    vals: List[torch.Tensor] = []
                    idxs: List[torch.Tensor] = []
                    for start in range(0, tea_span.shape[0], chunk):
                        tl = tea_span[start : start + chunk].float()
                        v, ix = tl.topk(k_use, dim=-1)
                        vals.append(v)
                        idxs.append(ix)
                        del tl
                    tea_k = torch.cat(vals, dim=0)
                    tea_idx = torch.cat(idxs, dim=0)
                    del vals, idxs, tea_span, tea_out
                else:
                    del tea_out

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            soft_m = soft_i.to(dtype=model_dtype)
            stu_inputs = torch.cat(
                [soft_m, prompt_emb.to(dtype=model_dtype), yhat_emb.to(dtype=model_dtype)],
                dim=0,
            ).unsqueeze(0)
            stu_attn = torch.ones(stu_inputs.shape[:2], dtype=torch.long, device=self.device)
            self.model.eval()
            stu_out = self.model(
                inputs_embeds=stu_inputs,
                attention_mask=stu_attn,
                use_cache=False,
            ).logits
            stu_prefix = int(soft_i.shape[0] + prompt_emb.shape[0])
            stu_span = stu_out[
                0, stu_prefix - 1 : stu_prefix - 1 + len(yhat_ids), :
            ]
            if stu_span.shape[0] != len(yhat_ids):
                continue

            if k > 0:
                assert tea_k is not None and tea_idx is not None
                stu_parts: List[torch.Tensor] = []
                for start in range(0, stu_span.shape[0], chunk):
                    stu_parts.append(
                        stu_span[start : start + chunk].gather(-1, tea_idx[start : start + chunk])
                    )
                stu_k = torch.cat(stu_parts, dim=0)
                del stu_out, stu_span, stu_parts
                log_p_s = F.log_softmax(stu_k.float(), dim=-1)
                log_p_t = F.log_softmax(tea_k.float(), dim=-1)
                kl = (log_p_t.exp() * (log_p_t - log_p_s)).sum(dim=-1).clamp(max=clip)
                kl = kl.mean()
                del stu_k, tea_k, tea_idx, log_p_s, log_p_t
            else:
                del stu_out
                kl = _chunked_forward_kl(
                    stu_span,
                    tea_span,
                    clip=clip,
                    chunk=chunk,
                    topk=0,
                )
                del stu_span, tea_span

            per_losses.append(kl)
            n_tokens_total += len(yhat_ids)
            n_ok += 1

        if not per_losses:
            loss = soft.float().sum() * 0.0
        else:
            loss = torch.stack(per_losses).mean()
            if self.icps.prefix_l2_coef > 0:
                loss = loss + float(self.icps.prefix_l2_coef) * soft.float().pow(2).mean()

        if not torch.isfinite(loss):
            loss = soft.float().sum() * 0.0

        stats = {
            "n_tokens": float(n_tokens_total),
            "n_examples": float(len(problem_texts)),
            "n_ok": float(n_ok),
            "prefix_norm": self.prefix_norm(soft),
        }
        return loss, stats

    # ------------------------------------------------------------------ #
    # Generation / accuracy
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_texts(
        self,
        problem_text: str,
        n: int,
        *,
        use_icps: bool,
        soft_override: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        greedy: bool = False,
    ) -> List[str]:
        ro = self.cfg.rollout
        max_new = int(max_new_tokens if max_new_tokens is not None else ro.max_new_tokens)
        temp = float(temperature if temperature is not None else ro.temperature)
        greedy = bool(greedy)
        if greedy:
            n = 1

        if (not use_icps) and (not greedy) and self._vllm is not None:
            return self._vllm_generate_base(
                [problem_text], n=n, max_new_tokens=max_new, temperature=temp
            )[0]

        prompt_emb = self._prompt_embeds(problem_text)
        if use_icps:
            soft = soft_override if soft_override is not None else self.soft_prefix(problem_text)
            soft = self._finite_soft(soft.detach()).to(dtype=self.dtype)
            inputs = torch.cat([soft, prompt_emb], dim=0).unsqueeze(0)
        else:
            inputs = prompt_emb.unsqueeze(0)
        attn = torch.ones(inputs.shape[:2], dtype=torch.long, device=self.device)

        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=not greedy,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
            use_cache=True,
        )
        if not greedy:
            gen_kwargs.update(temperature=temp, top_p=ro.top_p)
            if ro.top_k > 0:
                gen_kwargs["top_k"] = ro.top_k

        texts: List[str] = []
        chunk = 1 if max_new >= 1024 else min(4, n)
        remaining = n
        while remaining > 0:
            m = min(chunk, remaining)
            emb = inputs.repeat(m, 1, 1)
            mask = attn.repeat(m, 1)
            out = self.model.generate(
                inputs_embeds=emb,
                attention_mask=mask,
                eos_token_id=self.tokenizer.eos_token_id,
                **gen_kwargs,
            )
            prompt_len = int(inputs.shape[1])
            for row in out:
                if row.shape[0] > prompt_len:
                    cont = row[prompt_len:].tolist()
                else:
                    cont = row.tolist()
                while cont and cont[-1] in {
                    self.tokenizer.pad_token_id,
                    self.tokenizer.eos_token_id,
                }:
                    cont.pop()
                texts.append(self.decode(cont))
            remaining -= m
        return texts

    @torch.no_grad()
    def generate_texts_batch(
        self,
        problem_texts: Sequence[str],
        *,
        use_icps: bool,
        soft_override: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Batched prefix+prompt generation (left-padded ``inputs_embeds``)."""
        if len(problem_texts) <= 1:
            soft_i = None if soft_override is None else soft_override[0]
            return self.generate_texts(
                problem_texts[0] if problem_texts else "",
                n=1,
                use_icps=use_icps,
                soft_override=soft_i,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

        ro = self.cfg.rollout
        max_new = int(max_new_tokens if max_new_tokens is not None else ro.max_new_tokens)
        temp = float(temperature if temperature is not None else ro.temperature)
        rows: List[torch.Tensor] = []
        for i, text in enumerate(problem_texts):
            prompt_emb = self._prompt_embeds(text)
            if use_icps:
                if soft_override is None:
                    soft = self.soft_prefix(text)
                else:
                    soft = soft_override[i]
                soft = self._finite_soft(soft.detach()).to(dtype=self.dtype)
                rows.append(torch.cat([soft, prompt_emb], dim=0))
            else:
                rows.append(prompt_emb)

        max_len = max(int(r.shape[0]) for r in rows)
        h = int(rows[0].shape[-1])
        bsz = len(rows)
        emb = torch.zeros(bsz, max_len, h, device=self.device, dtype=self.dtype)
        mask = torch.zeros(bsz, max_len, dtype=torch.long, device=self.device)
        for i, row in enumerate(rows):
            n = int(row.shape[0])
            emb[i, max_len - n :] = row.to(dtype=self.dtype)
            mask[i, max_len - n :] = 1

        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
            use_cache=True,
            temperature=temp,
            top_p=ro.top_p,
        )
        if ro.top_k > 0:
            gen_kwargs["top_k"] = ro.top_k
        out = self.model.generate(
            inputs_embeds=emb,
            attention_mask=mask,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )
        texts: List[str] = []
        for row in out:
            if row.shape[0] > max_len:
                cont = row[max_len:].tolist()
            else:
                cont = row.tolist()
            while cont and cont[-1] in {
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            }:
                cont.pop()
            texts.append(self.decode(cont))
        return texts

    def accuracy(
        self,
        problem_text: str,
        gold: str,
        n: int,
        *,
        use_icps: bool,
        soft_override: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = None,
        greedy: bool = False,
    ) -> float:
        grader = self.cfg.reward.grader
        texts = self.generate_texts(
            problem_text,
            n,
            use_icps=use_icps,
            soft_override=soft_override,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
        )
        scores = []
        for t in texts:
            grade_text = t if grader == "math_verify" else strip_think_blocks(t)
            scores.append(float(grade_output(grade_text, gold, grader=grader)))
        return mean_or_nan(scores)

    # ------------------------------------------------------------------ #
    # Optional vLLM for base-only generations
    # ------------------------------------------------------------------ #

    def attach_vllm_base(self) -> None:
        """Load vLLM for fast base sampling; temporarily offload HF weights to CPU."""
        if not self.icps.use_vllm_base:
            return
        if self._vllm is not None:
            return
        self.release_teacher()
        import os

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM

        # Qwen3-1.7B position limit is 40960; keep headroom for the prompt.
        max_len = min(int(self.icps.vllm_max_model_len), 40960)
        logger.info(
            "Attaching vLLM base sampler (gpu_mem=%.2f max_model_len=%d); offloading HF → CPU",
            self.icps.vllm_gpu_memory_utilization,
            max_len,
        )
        self._hf_device = self.device
        self.model.to("cpu")
        self.encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self._vllm = LLM(
                model=self.cfg.model.name_or_path,
                trust_remote_code=self.cfg.model.trust_remote_code,
                gpu_memory_utilization=self.icps.vllm_gpu_memory_utilization,
                max_model_len=max_len,
                enforce_eager=False,
            )
        except Exception:
            # Restore HF if vLLM construction fails after offload.
            device = getattr(self, "_hf_device", None) or resolve_device(self.cfg.model.device)
            self.device = device if isinstance(device, torch.device) else torch.device(str(device))
            self.model.to(self.device)
            self.encoder.to(device=self.device, dtype=torch.float32)
            self.model.eval()
            raise
        logger.info("vLLM base sampler ready")

    def release_vllm_restore_hf(self) -> None:
        """Free vLLM and move HF model/encoder back to GPU for soft-prefix gens."""
        if self._vllm is None:
            return
        logger.info("Releasing vLLM; restoring HF model to GPU for ICPS gens")
        try:
            del self._vllm
        except Exception:
            pass
        self._vllm = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        device = getattr(self, "_hf_device", None)
        if device is None:
            device = resolve_device(self.cfg.model.device)
        self.device = device if isinstance(device, torch.device) else torch.device(str(device))
        self.model.to(self.device)
        self.encoder.to(device=self.device, dtype=torch.float32)
        self.model.eval()
        logger.info("HF restored on %s", self.device)

    def _render_solver_prompt(self, problem_text: str) -> str:
        msgs = solver_messages(
            problem_text,
            append_boxed_instruction=self.cfg.rollout.append_boxed_instruction,
        )
        try:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.model.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def _vllm_generate_base(
        self,
        problems: Sequence[str],
        *,
        n: int,
        max_new_tokens: int,
        temperature: float,
    ) -> List[List[str]]:
        from vllm import SamplingParams

        assert self._vllm is not None
        ro = self.cfg.rollout
        top_k = -1 if ro.top_k <= 0 else ro.top_k
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=ro.top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            # Match OPSD README eval knobs.
            min_p=0.0,
            presence_penalty=0.0,
            # Chat turn terminator. Correct for instruct (eos=<|im_end|>) and
            # essential for base models whose eos=<|endoftext|> != template end.
            stop=["<|im_end|>"],
        )
        prompts = [self._render_solver_prompt(p) for p in problems]
        outputs = self._vllm.generate(prompts, params, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outputs]

    def accuracy_base_batch(
        self,
        problems: Sequence[str],
        golds: Sequence[str],
        n: int,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> List[float]:
        """Batched base Acc; uses vLLM when attached."""
        max_new = int(
            max_new_tokens if max_new_tokens is not None else self.cfg.rollout.max_new_tokens
        )
        grader = self.cfg.reward.grader
        if self._vllm is not None:
            gens = self._vllm_generate_base(
                problems, n=n, max_new_tokens=max_new, temperature=self.cfg.rollout.temperature
            )
        else:
            gens = [
                self.generate_texts(p, n, use_icps=False, max_new_tokens=max_new)
                for p in problems
            ]
        out: List[float] = []
        for texts, gold in zip(gens, golds):
            scores = []
            for t in texts:
                grade_text = t if grader == "math_verify" else strip_think_blocks(t)
                scores.append(float(grade_output(grade_text, gold, grader=grader)))
            out.append(mean_or_nan(scores))
        return out

    def accuracy_snapshot_batch(
        self,
        problems: Sequence[str],
        golds: Sequence[str],
        n: int,
        snapshot: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> List[float]:
        """Score a frozen HF snapshot, then restore the current student.

        Full-FT keeps the trained weights in ``self.model``. Scoring that
        object as "base" would leak the update. This path loads
        ``snapshot`` (the original instruct checkpoint) for the base arm.
        """
        snapshot = str(snapshot)
        logger.info("Eval: scoring frozen snapshot via HF: %s", snapshot)
        held = self.model
        held_device = self.device
        held.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_kwargs = dict(trust_remote_code=self.cfg.model.trust_remote_code)
        if self.cfg.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.model.attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot, dtype=self.dtype, **model_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                snapshot, torch_dtype=self.dtype, **model_kwargs
            )
        model.to(held_device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if hasattr(model, "config"):
            model.config.use_cache = True
        self.model = model
        try:
            return self.accuracy_base_batch(
                problems, golds, n, max_new_tokens=max_new_tokens
            )
        finally:
            del self.model
            self.model = held
            self.model.to(held_device)
            self.model.eval()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Restored trained student after snapshot base eval")

    # ------------------------------------------------------------------ #
    # Privileged upper anchor (eval-only): frozen solver sees y* in context.
    # Used for eta_acc = (Acc(icps)-Acc(base)) / (Acc(priv)-Acc(base)).
    # y* is NEVER used for OPD KL; this path is diagnostic only.
    # ------------------------------------------------------------------ #

    def _render_privileged_prompt(
        self, problem_text: str, reference_solution: str
    ) -> str:
        msgs = privileged_solver_messages(problem_text, reference_solution)
        try:
            return self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.cfg.model.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

    def _vllm_generate_privileged(
        self,
        problems: Sequence[str],
        refs: Sequence[str],
        *,
        n: int,
        max_new_tokens: int,
        temperature: float,
    ) -> List[List[str]]:
        from vllm import SamplingParams

        assert self._vllm is not None
        ro = self.cfg.rollout
        top_k = -1 if ro.top_k <= 0 else ro.top_k
        params = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=ro.top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            min_p=0.0,
            presence_penalty=0.0,
            stop=["<|im_end|>"],
        )
        prompts = [
            self._render_privileged_prompt(p, r) for p, r in zip(problems, refs)
        ]
        outputs = self._vllm.generate(prompts, params, use_tqdm=False)
        return [[o.text for o in out.outputs] for out in outputs]

    def accuracy_privileged_batch(
        self,
        problems: Sequence[str],
        refs: Sequence[str],
        golds: Sequence[str],
        n: int,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> List[float]:
        """Batched privileged Acc (frozen solver, y* in context); vLLM when attached."""
        max_new = int(
            max_new_tokens if max_new_tokens is not None else self.cfg.rollout.max_new_tokens
        )
        grader = self.cfg.reward.grader
        if self._vllm is not None:
            gens = self._vllm_generate_privileged(
                problems,
                refs,
                n=n,
                max_new_tokens=max_new,
                temperature=self.cfg.rollout.temperature,
            )
        else:
            raise NotImplementedError(
                "Privileged eval requires the vLLM base sampler; set "
                "icps.use_vllm_base=true (HF privileged path not implemented)."
            )
        out: List[float] = []
        for texts, gold in zip(gens, golds):
            scores = []
            for t in texts:
                grade_text = t if grader == "math_verify" else strip_think_blocks(t)
                scores.append(float(grade_output(grade_text, gold, grader=grader)))
            out.append(mean_or_nan(scores))
        return out

    def save_encoder(self, path: str | Path, *, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "encoder": self.encoder.state_dict(),
            "icps": self.icps.__dict__,
            "model_name": self.cfg.model.name_or_path,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        logger.info("Saved ICPS encoder to %s", path)

    def load_encoder(self, path: str | Path) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(blob["encoder"])
        logger.info("Loaded ICPS encoder from %s", path)

    def save_student(self, path: str | Path, *, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {
            "icps": self.icps.__dict__,
            "model_name": self.cfg.model.name_or_path,
            "student_update": self.student_update,
        }
        if extra:
            meta.update(extra)
        (path / "pssd_train.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        logger.info("Saved full-FT student to %s", path)

    def load_student(self, path: str | Path) -> None:
        path = Path(path)
        logger.info("Loading full-FT student from %s", path)
        model_kwargs = dict(trust_remote_code=self.cfg.model.trust_remote_code)
        if self.cfg.model.attn_implementation:
            model_kwargs["attn_implementation"] = self.cfg.model.attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), dtype=self.dtype, **model_kwargs
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                str(path), torch_dtype=self.dtype, **model_kwargs
            )
        model.to(self.device)
        for p in model.parameters():
            p.requires_grad_(True)
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.train()
        del self.model
        self.model = model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Loaded full-FT student from %s", path)
