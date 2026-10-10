"""PSSD engine: 8B rewriter (full FT), frozen 1.7B solver, frozen 8B OPD teacher.

- Rewriter q_phi : Qwen3-8B, all parameters trainable.
- Solver p_theta : frozen 1.7B.
- Teacher p_T    : frozen 8B, scoring-only, same prompt as the solver (no y*).

OPD scoring: y ~ p_S(.|r), then clip(log p_T(y|r) − log p_S(y|r)).
The teacher never sees the reference solution.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from pssd.config import PSSDConfig
from pssd.templates import Message, clean_rewrite_text
from pssd.utils import get_logger, resolve_device, resolve_dtype

logger = get_logger(__name__)

TokenIds = List[int]


def _assert_not_lora_checkpoint(path: str | Path) -> None:
    p = Path(path)
    if p.is_dir() and (p / "adapter_config.json").exists():
        raise RuntimeError(
            f"{p} is a LoRA adapter. PSSD fine-tunes the full 8B rewriter; "
            "pass a full model directory (config.json + weights)."
        )


def _load_causal_lm(
    path: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool,
    attn_implementation: Optional[str],
):
    _assert_not_lora_checkpoint(path)
    model_kwargs = dict(trust_remote_code=trust_remote_code)
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, **model_kwargs
        )
    model.to(device)
    return model


def _load_tokenizer(path: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class PSSDEngine:
    """Tokenizer + rewriter (trainable 8B) + frozen solver/teacher."""

    def __init__(
        self,
        cfg: PSSDConfig,
        rewriter_path: Optional[str | Path] = None,
        load_rewriter: bool = True,
        load_teacher: bool = True,
        *,
        adapter_path: Optional[str | Path] = None,
        create_adapter: Optional[bool] = None,
    ) -> None:
        self.cfg = cfg
        self.device = resolve_device(cfg.model.device)
        self.dtype = resolve_dtype(cfg.model.dtype, self.device)
        if adapter_path is not None and rewriter_path is None:
            rewriter_path = adapter_path
        if create_adapter is not None:
            load_rewriter = create_adapter

        solver_name = cfg.model.name_or_path
        logger.info(
            "Loading frozen solver: %s (device=%s, dtype=%s)",
            solver_name, self.device, self.dtype,
        )
        self.tokenizer = _load_tokenizer(solver_name, cfg.model.trust_remote_code)
        self.solver = _load_causal_lm(
            solver_name,
            dtype=self.dtype,
            device=self.device,
            trust_remote_code=cfg.model.trust_remote_code,
            attn_implementation=cfg.model.attn_implementation,
        )
        for p in self.solver.parameters():
            p.requires_grad_(False)
        self.solver.eval()

        self.rewriter = None
        self.rewriter_tokenizer = None
        self.has_adapter = False  # retained so older callers can query; always False
        if load_rewriter:
            rw_name = str(rewriter_path or cfg.model.rewriter_name_or_path)
            trainable = rewriter_path is None
            logger.info(
                "Loading %s rewriter: %s",
                "trainable" if trainable else "frozen",
                rw_name,
            )
            self.rewriter_tokenizer = _load_tokenizer(
                rw_name, cfg.model.trust_remote_code
            )
            self.rewriter = _load_causal_lm(
                rw_name,
                dtype=self.dtype,
                device=self.device,
                trust_remote_code=cfg.model.trust_remote_code,
                attn_implementation=cfg.model.attn_implementation,
            )
            if trainable:
                for p in self.rewriter.parameters():
                    p.requires_grad_(True)
                if cfg.model.gradient_checkpointing:
                    if hasattr(self.rewriter, "enable_input_require_grads"):
                        self.rewriter.enable_input_require_grads()
                    try:
                        self.rewriter.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False}
                        )
                    except TypeError:
                        self.rewriter.gradient_checkpointing_enable()
                    if hasattr(self.rewriter.config, "use_cache"):
                        self.rewriter.config.use_cache = False
                n_train = sum(p.numel() for p in self.rewriter.parameters() if p.requires_grad)
                logger.info("Full-parameter rewriter (%.2fB trainable params)", n_train / 1e9)
            else:
                for p in self.rewriter.parameters():
                    p.requires_grad_(False)
            self.rewriter.eval()

        self.teacher = None
        self.teacher_tokenizer = None
        teacher_name = str(cfg.model.teacher_name_or_path or cfg.model.rewriter_name_or_path)
        if load_teacher:
            can_share = (
                self.rewriter is not None
                and self.rewriter_tokenizer is not None
                and not any(p.requires_grad for p in self.rewriter.parameters())
                and Path(teacher_name).resolve() == Path(
                    str(rewriter_path or cfg.model.rewriter_name_or_path)
                ).resolve()
            )
            if can_share:
                logger.info("Sharing frozen rewriter weights as OPD teacher: %s", teacher_name)
                self.teacher = self.rewriter
                self.teacher_tokenizer = self.rewriter_tokenizer
            else:
                logger.info("Loading frozen OPD teacher (no y*): %s", teacher_name)
                self.teacher_tokenizer = _load_tokenizer(
                    teacher_name, cfg.model.trust_remote_code
                )
                self.teacher = _load_causal_lm(
                    teacher_name,
                    dtype=self.dtype,
                    device=self.device,
                    trust_remote_code=cfg.model.trust_remote_code,
                    attn_implementation=cfg.model.attn_implementation,
                )
                for p in self.teacher.parameters():
                    p.requires_grad_(False)
                self.teacher.eval()

    def _resolve_role(
        self,
        *,
        rewriter: bool = False,
        teacher: bool = False,
        use_rewriter: bool = False,
        use_adapter: Optional[bool] = None,
        role: Optional[str] = None,
    ) -> str:
        if role is not None:
            return role
        if use_adapter is not None:
            return "rewriter" if use_adapter else "solver"
        if teacher:
            return "teacher"
        if rewriter or use_rewriter:
            return "rewriter"
        return "solver"

    def _role_model(self, role: str):
        if role == "rewriter":
            if self.rewriter is None:
                raise RuntimeError("rewriter was not loaded")
            return self.rewriter
        if role == "teacher":
            if self.teacher is None:
                raise RuntimeError("OPD teacher was not loaded")
            return self.teacher
        if self.solver is None:
            raise RuntimeError("solver was not loaded")
        return self.solver

    def _role_tokenizer(self, role: str):
        if role == "rewriter":
            if self.rewriter_tokenizer is None:
                raise RuntimeError("rewriter tokenizer was not loaded")
            return self.rewriter_tokenizer
        if role == "teacher":
            if self.teacher_tokenizer is None:
                raise RuntimeError("teacher tokenizer was not loaded")
            return self.teacher_tokenizer
        return self.tokenizer

    # ------------------------------------------------------------------ #
    # Tokenization helpers
    # ------------------------------------------------------------------ #

    def chat_ids(
        self,
        messages: Sequence[Message],
        enable_thinking: Optional[bool] = None,
        *,
        rewriter: bool = False,
        teacher: bool = False,
    ) -> TokenIds:
        """Render chat messages to prompt token ids (generation prompt added).

        ``enable_thinking`` defaults to ``model.enable_thinking`` (solver path).
        Pass an explicit value for the rewriter (typically False) so rewrite
        prompts do not invite truncated ``<think>`` dumps into ``r``.
        """
        thinking = (
            self.cfg.model.enable_thinking if enable_thinking is None else enable_thinking
        )
        role = self._resolve_role(rewriter=rewriter, teacher=teacher)
        tokenizer = self._role_tokenizer(role)
        try:
            ids = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
        except TypeError:
            ids = tokenizer.apply_chat_template(
                list(messages), tokenize=True, add_generation_prompt=True
            )
        return list(ids)

    def chat_text(
        self,
        messages: Sequence[Message],
        enable_thinking: Optional[bool] = None,
        *,
        rewriter: bool = False,
        teacher: bool = False,
    ) -> str:
        """Rendered prompt text (used for structural leakage assertions)."""
        role = self._resolve_role(rewriter=rewriter, teacher=teacher)
        tok = self._role_tokenizer(role)
        return tok.decode(
            self.chat_ids(
                messages,
                enable_thinking=enable_thinking,
                rewriter=rewriter,
                teacher=teacher,
            ),
            skip_special_tokens=False,
        )

    def encode(self, text: str, *, rewriter: bool = False, teacher: bool = False) -> TokenIds:
        role = self._resolve_role(rewriter=rewriter, teacher=teacher)
        return self._role_tokenizer(role).encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int], *, rewriter: bool = False, teacher: bool = False) -> str:
        role = self._resolve_role(rewriter=rewriter, teacher=teacher)
        return self._role_tokenizer(role).decode(list(ids), skip_special_tokens=True)

    def token_count(self, text: str, *, rewriter: bool = False, teacher: bool = False) -> int:
        return len(self.encode(text, rewriter=rewriter, teacher=teacher))

    def continuation_for_teacher(self, solver_cont_ids: TokenIds) -> TokenIds:
        """Map a solver continuation onto the teacher tokenizer.

        Qwen3-8B and Qwen3-1.7B share a vocabulary, so we reuse the solver
        token ids when sizes match (token-aligned OPD). Otherwise decode and
        re-encode the same text.
        """
        if self.teacher_tokenizer is None:
            raise RuntimeError("OPD teacher was not loaded")
        if self.teacher_tokenizer is self.tokenizer:
            return list(solver_cont_ids)
        t_vocab = getattr(self.teacher_tokenizer, "vocab_size", None)
        s_vocab = getattr(self.tokenizer, "vocab_size", None)
        if t_vocab is not None and t_vocab == s_vocab:
            return list(solver_cont_ids)
        return self.encode(self.decode(solver_cont_ids), teacher=True)

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    def _strip_trailing_special(self, ids: List[int], *, role: str = "solver") -> List[int]:
        tok = self._role_tokenizer(role)
        specials = {tok.pad_token_id, tok.eos_token_id}
        specials.discard(None)
        while ids and ids[-1] in specials:
            ids = ids[:-1]
        return ids

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: TokenIds,
        n: int,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        do_sample: bool = True,
        use_rewriter: bool = False,
        use_adapter: Optional[bool] = None,
    ) -> List[TokenIds]:
        """Generate n continuations for one prompt; returns continuation ids only."""
        use_rewriter = self._resolve_role(
            use_rewriter=use_rewriter, use_adapter=use_adapter
        ) == "rewriter"
        if not do_sample and n > 1:
            raise ValueError("Greedy decoding with n > 1 would produce identical samples")
        role = "rewriter" if use_rewriter else "solver"
        model = self._role_model(role)
        tok = self._role_tokenizer(role)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        model.eval()
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=n,
            do_sample=do_sample,
            pad_token_id=tok.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
        sequences = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )
        prompt_len = input_ids.shape[1]
        outputs: List[TokenIds] = []
        for row in sequences:
            cont = self._strip_trailing_special(
                row[prompt_len:].tolist(), role=role
            )
            outputs.append(cont)
        return outputs

    def sample_rewrites(
        self,
        prompt_ids: TokenIds,
        k: int,
        use_rewriter: bool = True,
        use_adapter: Optional[bool] = None,
    ) -> Tuple[List[str], List[TokenIds]]:
        """Sample K rewrites from the rewriter; returns (cleaned texts, token ids).

        Think residue and trailing ``\\boxed`` / answer artifacts are stripped from
        the solver-facing text. When cleanup is a no-op we keep the original
        sampled ids (strictly on-policy); otherwise we re-encode the cleaned text
        so imitation / outer-loop targets match what the frozen solver sees.

        If ``rewrite.include_greedy`` is set, the first of the K rewrites is
        greedy and the remaining ``k-1`` are temperature samples.
        """
        use_rewriter = self._resolve_role(
            use_rewriter=use_rewriter, use_adapter=use_adapter
        ) == "rewriter"
        rw = self.cfg.rewrite
        ids_list: List[TokenIds] = []
        n_sample = k
        if rw.include_greedy and k >= 1:
            ids_list.extend(
                self.generate(
                    prompt_ids,
                    n=1,
                    max_new_tokens=rw.max_new_tokens,
                    do_sample=False,
                    use_rewriter=use_rewriter,
                )
            )
            n_sample = k - 1
        if n_sample > 0:
            ids_list.extend(
                self.generate(
                    prompt_ids,
                    n=n_sample,
                    max_new_tokens=rw.max_new_tokens,
                    temperature=rw.temperature,
                    top_p=rw.top_p,
                    do_sample=True,
                    use_rewriter=use_rewriter,
                )
            )
        texts: List[str] = []
        target_ids: List[TokenIds] = []
        for ids in ids_list:
            raw = self.decode(ids, rewriter=use_rewriter).strip()
            text = clean_rewrite_text(self.decode(ids, rewriter=use_rewriter))
            texts.append(text)
            target_ids.append(
                ids if text == raw else self.encode(text, rewriter=use_rewriter)
            )
        return texts, target_ids

    def greedy_rewrite(
        self,
        prompt_ids: TokenIds,
        use_rewriter: bool = True,
        use_adapter: Optional[bool] = None,
    ) -> str:
        """Deterministic single rewrite for inference (doc Sec. 3.5)."""
        use_rewriter = self._resolve_role(
            use_rewriter=use_rewriter, use_adapter=use_adapter
        ) == "rewriter"
        ids = self.generate(
            prompt_ids, n=1, max_new_tokens=self.cfg.rewrite.max_new_tokens,
            do_sample=False, use_rewriter=use_rewriter,
        )[0]
        return clean_rewrite_text(self.decode(ids, rewriter=use_rewriter))

    def sample_rollouts(self, solver_prompt_ids: TokenIds, g: int) -> List[TokenIds]:
        """Sample G answer rollouts from the FROZEN solver."""
        ro = self.cfg.rollout
        outs: List[TokenIds] = []
        chunk = 1 if ro.max_new_tokens >= 8192 else 4
        remaining = g
        while remaining > 0:
            n = min(chunk, remaining)
            outs.extend(
                self.generate(
                    solver_prompt_ids,
                    n=n,
                    max_new_tokens=ro.max_new_tokens,
                    temperature=ro.temperature,
                    top_p=ro.top_p,
                    top_k=ro.top_k,
                    do_sample=True,
                    use_rewriter=False,
                )
            )
            remaining -= n
        return outs

    # ------------------------------------------------------------------ #
    # Teacher-forced scoring (no grad)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def score_target_logprobs(
        self,
        pairs: Sequence[Tuple[TokenIds, TokenIds]],
        use_rewriter: bool = False,
        use_adapter: Optional[bool] = None,
        role: Optional[str] = None,
        batch_size: int = 4,
    ) -> List[torch.Tensor]:
        """Per-token log p(target_t | prefix, target_<t) for each (prefix, target).

        Solver: prefix = solver prompt on r. Teacher: the same solver-facing
        prompt tokenized by the 8B OPD teacher (no y*).
        Returns float32 CPU tensors, one per pair (length = len(target)).
        """
        resolved = role or self._resolve_role(
            use_rewriter=use_rewriter, use_adapter=use_adapter
        )
        results: List[torch.Tensor] = [torch.empty(0)] * len(pairs)
        model = self._role_model(resolved)
        model.eval()
        indexed = list(enumerate(pairs))
        for start in range(0, len(indexed), batch_size):
            chunk = indexed[start : start + batch_size]
            logps = self._forward_target_logprobs(
                model,
                self._role_tokenizer(resolved),
                [p for _, (p, _) in chunk],
                [t for _, (_, t) in chunk],
            )
            for (orig_idx, _), lp in zip(chunk, logps):
                results[orig_idx] = lp
        return results

    def _forward_target_logprobs(
        self,
        model,
        tokenizer,
        prefixes: Sequence[TokenIds],
        targets: Sequence[TokenIds],
    ) -> List[torch.Tensor]:
        """One padded forward pass; gather log-probs of target tokens (no grad)."""
        pad_id = tokenizer.pad_token_id
        lengths = [len(p) + len(t) for p, t in zip(prefixes, targets)]
        max_len = max(lengths)
        batch = len(prefixes)
        input_ids = torch.full((batch, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        for i, (p, t) in enumerate(zip(prefixes, targets)):
            seq = p + t
            input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[i, : len(seq)] = 1
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        out: List[torch.Tensor] = []
        for i, (p, t) in enumerate(zip(prefixes, targets)):
            if len(t) == 0:
                out.append(torch.empty(0))
                continue
            span = logits[i, len(p) - 1 : len(p) + len(t) - 1, :].float()
            logprobs = F.log_softmax(span, dim=-1)
            tgt = torch.tensor(t, dtype=torch.long, device=self.device).unsqueeze(-1)
            out.append(logprobs.gather(-1, tgt).squeeze(-1).cpu())
        return out

    # ------------------------------------------------------------------ #
    # Gradient paths (rewriter only; solver/teacher are never differentiated)
    # ------------------------------------------------------------------ #

    def rewrite_logprobs_with_grad(
        self, prefix_ids: TokenIds, target_lists: Sequence[TokenIds]
    ) -> List[torch.Tensor]:
        """log q_phi(r_t | x, r_<t) for each rewrite, WITH gradient on the 8B rewriter."""
        if self.rewriter is None:
            raise RuntimeError("Gradient path requires the 8B rewriter")
        self.rewriter.train()
        results: List[torch.Tensor] = []
        prefix = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        for target in target_lists:
            if len(target) == 0:
                results.append(torch.empty(0, device=self.device, requires_grad=False))
                continue
            tgt = torch.tensor([target], dtype=torch.long, device=self.device)
            input_ids = torch.cat([prefix, tgt], dim=1)
            attention_mask = torch.ones_like(input_ids)
            logits = self.rewriter(input_ids=input_ids, attention_mask=attention_mask).logits
            span = logits[0, prefix.shape[1] - 1 : -1, :].float()
            logprobs = F.log_softmax(span, dim=-1)
            results.append(logprobs.gather(-1, tgt[0].unsqueeze(-1)).squeeze(-1))
        self.rewriter.eval()
        return results

    def full_vocab_logprobs(
        self,
        prefix_ids: TokenIds,
        target_ids: TokenIds,
        use_rewriter: bool = False,
        use_adapter: Optional[bool] = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Full-vocabulary log-probs [T, V] at each target position."""
        use_rewriter = self._resolve_role(
            use_rewriter=use_rewriter, use_adapter=use_adapter
        ) == "rewriter"
        if len(target_ids) == 0:
            raise ValueError("full_vocab_logprobs: empty target")
        model = self._role_model("rewriter" if use_rewriter else "solver")
        input_ids = torch.tensor([prefix_ids + target_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        grad_ctx = nullcontext() if requires_grad else torch.no_grad()
        if requires_grad:
            model.train()
        else:
            model.eval()
        with grad_ctx:
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            span = logits[0, len(prefix_ids) - 1 : -1, :].float()
            result = F.log_softmax(span, dim=-1)
        model.eval()
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def trainable_parameters(self) -> List[torch.nn.Parameter]:
        if self.rewriter is None:
            return []
        return [p for p in self.rewriter.parameters() if p.requires_grad]

    def save_rewriter(self, output_dir: str | Path) -> None:
        """Save the full rewriter weights (solver is never written)."""
        if self.rewriter is None or self.rewriter_tokenizer is None:
            raise RuntimeError("No rewriter to save")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.rewriter.save_pretrained(str(output_dir))
        self.rewriter_tokenizer.save_pretrained(str(output_dir))
        logger.info("Saved full rewriter to %s", output_dir)

    def save_adapter(self, output_dir: str | Path) -> None:
        """Backward-compatible alias; writes full rewriter weights, not LoRA."""
        self.save_rewriter(output_dir)
