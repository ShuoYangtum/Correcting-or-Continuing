"""Typed configuration for PSSD.

All hyper-parameters live in nested dataclasses and can be loaded from a YAML
file.  Unknown keys raise immediately so that config typos never fail
silently.  Defaults follow the technical document Section 3.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


class ConfigError(ValueError):
    """Raised on malformed or inconsistent configuration."""


# Kept only so historical YAML still parses. LoRA is never applied.
_DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Path markers that must never be trained or evaluated on.
_BANNED_DATASET_MARKERS = (
    "math500",
    "math-500",
    "math_500",
    "mvp_500",
    "eval_math50",
    "huggingfaceh4/math-500",
)


def dataset_path_is_banned(path: Optional[str | Path]) -> bool:
    if path is None:
        return False
    text = str(path).replace("\\", "/").lower()
    return any(marker in text for marker in _BANNED_DATASET_MARKERS)


def assert_dataset_allowed(path: Optional[str | Path], *, field: str = "dataset") -> None:
    if dataset_path_is_banned(path):
        raise ConfigError(
            f"{field} path is permanently banned: {path}"
        )


@dataclass
class ModelConfig:
    name_or_path: str = "Qwen/Qwen3-1.7B"  # frozen solver (student)
    rewriter_name_or_path: str = "Qwen/Qwen3-8B"
    # Frozen OPD teacher (larger model, never sees y*). Empty → rewriter_name_or_path.
    teacher_name_or_path: Optional[str] = None
    dtype: str = "auto"                 # auto | bfloat16 | float16 | float32
    device: str = "auto"                # auto | cuda | cuda:N | cpu
    trust_remote_code: bool = True
    enable_thinking: bool = False       # TM-off during training (doc Sec. 3.1)
    attn_implementation: Optional[str] = None  # e.g. "flash_attention_2"
    gradient_checkpointing: bool = True


@dataclass
class LoraSettings:
    """Ignored. Historical YAML may still contain this section."""
    r: int = 32
    alpha: int = 64
    dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: list(_DEFAULT_LORA_TARGETS))


@dataclass
class RewriteSampling:
    """Sampling config for the rewriter q_phi (K rewrites per problem)."""
    k: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    max_new_tokens: int = 256
    # If True: rewriter emits <rationale>+<rewrite>; only the rewrite body is
    # shown to the frozen solver for Acc/teacher_lik. Outer PG (if enabled) and
    # inner distillation still train on the full tagged generation.
    two_stage: bool = False
    # Rewriter chat template thinking switch (independent of solver TM).
    # Keep False so r is a problem restatement, not a truncated CoT dump.
    enable_thinking: bool = False
    # Drop rewrites whose body looks like a solution (\\boxed / #### / ...).
    invalidate_solution_leak: bool = True
    # Rewriter content policy:
    #   "strict" — clarify / reorder only (no strategy hints)
    #   "hint"   — may append a short Hint: section (no answers / no new numbers)
    style: str = "strict"
    # If True, prepend a few (original -> rewrite) demos before the real problem.
    # Default off for mainline: demos are short/easy and transfer poorly to contest
    # math; keep the code path for optional ablations.
    few_shot: bool = False
    # If True and k>=1: first rewrite is greedy, remaining k-1 are sampled.
    include_greedy: bool = False
    # Optional vLLM writer for offline SFT collection. Empty/None = rewriter_name_or_path.
    demo_model_name_or_path: Optional[str] = None
    # Inference / eval: validate rewrite, resample on failure, else use original.
    inference_validate: bool = True
    max_rewrite_retries: int = 3
    min_stem_ratio: float = 0.5
    max_hint_blocks: int = 1
    max_hint_bullets: int = 3
    # Stem-only cap vs original (Hint does not count).
    max_length_ratio: float = 2.0
    # Hint-body budget: max(max_hint_chars, max_hint_ratio * |original|).
    # 0 disables that term. Short originals rely on the char floor.
    max_hint_ratio: float = 2.0
    max_hint_chars: int = 480
    ban_assertive_hints: bool = True


@dataclass
class RolloutSampling:
    """Sampling config for the frozen solver p_theta (G rollouts per rewrite)."""
    g: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    # 0 disables top-k (OPSD eval: top-k disabled). HF generate uses this only when sampling.
    top_k: int = 0
    max_new_tokens: int = 1024
    # MVP / eval-only: greedy pass@1 budget. None = use max_new_tokens.
    eval_max_new_tokens: Optional[int] = None
    # OPSD eval appends: "Please reason step by step, and put your final answer within \\boxed{}."
    append_boxed_instruction: bool = False


@dataclass
class RewardSettings:
    clip_c: float = 5.0                 # pointwise clip on log p_T - log p_S (doc Sec. 3.3)
    length_budget: float = 1.5          # |r| / |x| token-length budget (doc Sec. 3.4)
    length_penalty_weight: float = 1.0
    use_discriminator: bool = False     # lenient injection judge (doc Sec. 3.4)
    discriminator_penalty: float = 0.2  # kept mild; v1 0.5 saturated near 1.0
    coverage_penalty_weight: float = 1.0  # penalize dropped critical numbers
    invalidate_missing_numbers: bool = True  # drop rewrites that omit any critical number
    group_std_eps: float = 1e-4         # DAPO-style dynamic filtering threshold
    # R(r) = score(r) - score(x) ≈ D(x)-D(r); better cross-problem pooling / η-local.
    relative_to_original: bool = False
    # If the best rewrite's relative KL gain ≤ 0, keep the original prompt.
    prefer_original_if_nonpositive: bool = False
    # Dense outer-loop reward source:
    #   "kl"      — clipped 8B-teacher / 1.7B-solver OPD log-ratio (no y*)
    #   "acc"     — scoring-rollout accuracy only
    #   "acc_lik" — Acc + lik_weight * teacher sequence log-prob
    dense_reward: str = "kl"
    acc_weight: float = 1.0
    lik_weight: float = 0.1             # small teacher-lik term when dense_reward=acc_lik
    # How MVP / ranking picks the winning rewrite:
    #   "kl" | "acc" | "teacher_lik"
    selection_criterion: str = "kl"
    # Accuracy grader: "regex" (pilot default) or "math_verify" (boxed).
    grader: str = "regex"


@dataclass
class OuterLoopSettings:
    """Outer loop: group-relative policy gradient in rewrite space.

    Disable it to obtain the PG-free fallback variant
    (best-of-K selection + inner distillation only, doc Sec. 3.3).
    """
    enabled: bool = True


@dataclass
class InnerLoopSettings:
    """Inner loop: amortize best-of-K winner into the rewriter.

    Modes:
      - "opsd": same-model rewrite-space KL (rejected when rewriter != solver)
      - "imitation": supervised NLL on the KL-selected winning rewrite tokens
    """
    enabled: bool = True
    weight: float = 0.5                 # lambda_in
    clip_tau: float = 5.0               # pointwise clip on per-token forward KL (opsd)
    mode: str = "imitation"             # "opsd" | "imitation"


@dataclass
class TrainSettings:
    steps: int = 100
    batch_size: int = 8                 # problems per optimizer step (paper spec: 64)
    learning_rate: float = 1e-5
    warmup_steps: int = 10
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    scoring_batch_size: int = 4         # micro-batch for teacher-forced scoring
    optimizer: str = "adamw8bit"        # adamw8bit (8B full FT) | adamw
    log_every: int = 1
    save_every: int = 50
    seed: int = 42
    output_dir: str = "outputs/pssd"


@dataclass
class WarmupSettings:
    """Optional KL-winner SFT warmup before the main dual-loop / imitation train.

    Scans training problems, keeps rewrites with KL strictly above the original,
    then runs supervised NLL on those (prompt, rewrite) pairs.
    """
    enabled: bool = False
    target_pairs: int = 16
    max_problems: int = 64
    sft_epochs: int = 2
    sft_batch_size: int = 2
    # If set, skip online collection and SFT these JSONL pairs
    # ({problem, rewrite, ...} from scripts/collect_rewrite_sft_pairs.py).
    pairs_path: Optional[str] = None


@dataclass
class DataSettings:
    train_path: str = "data/sample_math.jsonl"
    eval_path: Optional[str] = None
    max_problems: Optional[int] = None


@dataclass
class WandbSettings:
    """Optional Weights & Biases logging (disabled by default)."""
    enabled: bool = False
    project: str = "pssd"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    mode: str = "online"  # online | offline | disabled


@dataclass
class PSSDConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraSettings = field(default_factory=LoraSettings)
    rewrite: RewriteSampling = field(default_factory=RewriteSampling)
    rollout: RolloutSampling = field(default_factory=RolloutSampling)
    reward: RewardSettings = field(default_factory=RewardSettings)
    outer: OuterLoopSettings = field(default_factory=OuterLoopSettings)
    inner: InnerLoopSettings = field(default_factory=InnerLoopSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    warmup: WarmupSettings = field(default_factory=WarmupSettings)
    data: DataSettings = field(default_factory=DataSettings)
    wandb: WandbSettings = field(default_factory=WandbSettings)

    def validate(self) -> None:
        if self.rewrite.k < 1:
            raise ConfigError("rewrite.k must be >= 1")
        if self.outer.enabled and self.rewrite.k < 2:
            raise ConfigError("rewrite.k must be >= 2 when the outer loop is enabled "
                              "(group-relative advantages need a group)")
        if self.rollout.g < 1:
            raise ConfigError("rollout.g must be >= 1")
        if self.reward.clip_c <= 0:
            raise ConfigError("reward.clip_c must be > 0")
        if self.reward.selection_criterion not in ("kl", "acc", "teacher_lik"):
            raise ConfigError(
                "reward.selection_criterion must be 'kl', 'acc', or 'teacher_lik'"
            )
        if self.reward.dense_reward not in ("kl", "acc", "acc_lik"):
            raise ConfigError("reward.dense_reward must be 'kl', 'acc', or 'acc_lik'")
        if self.reward.grader not in ("regex", "math_verify"):
            raise ConfigError("reward.grader must be 'regex' or 'math_verify'")
        if self.reward.acc_weight < 0 or self.reward.lik_weight < 0:
            raise ConfigError("reward.acc_weight and lik_weight must be >= 0")
        if self.rollout.top_k < 0:
            raise ConfigError("rollout.top_k must be >= 0 (0 disables top-k)")
        if self.rollout.eval_max_new_tokens is not None and self.rollout.eval_max_new_tokens < 1:
            raise ConfigError("rollout.eval_max_new_tokens must be >= 1 when set")
        if self.inner.weight < 0:
            raise ConfigError("inner.weight must be >= 0")
        if self.inner.mode not in ("opsd", "imitation"):
            raise ConfigError("inner.mode must be 'opsd' or 'imitation'")
        if self.rewrite.style not in ("strict", "hint"):
            raise ConfigError("rewrite.style must be 'strict' or 'hint'")
        if self.rewrite.max_rewrite_retries < 0:
            raise ConfigError("rewrite.max_rewrite_retries must be >= 0")
        if not (0.0 < self.rewrite.min_stem_ratio <= 1.0):
            raise ConfigError("rewrite.min_stem_ratio must be in (0, 1]")
        if self.rewrite.max_hint_blocks < 0 or self.rewrite.max_hint_bullets < 0:
            raise ConfigError("rewrite max_hint_blocks/bullets must be >= 0")
        if self.rewrite.max_length_ratio <= 0:
            raise ConfigError("rewrite.max_length_ratio must be > 0")
        if self.rewrite.max_hint_ratio < 0:
            raise ConfigError("rewrite.max_hint_ratio must be >= 0")
        if self.rewrite.max_hint_chars < 0:
            raise ConfigError("rewrite.max_hint_chars must be >= 0")
        if self.train.batch_size < 1:
            raise ConfigError("train.batch_size must be >= 1")
        if self.train.steps < 0:
            raise ConfigError("train.steps must be >= 0")
        # steps=0 + warmup.enabled = SFT-only (skip outer/inner main loop).
        if self.train.steps == 0 and not self.warmup.enabled:
            raise ConfigError("train.steps=0 requires warmup.enabled (SFT-only mode)")
        if self.train.steps >= 1 and (not self.outer.enabled and not self.inner.enabled):
            raise ConfigError("At least one of outer/inner loops must be enabled when train.steps>=1")
        if self.train.optimizer not in ("adamw", "adamw8bit"):
            raise ConfigError("train.optimizer must be 'adamw' or 'adamw8bit'")
        if not str(self.model.rewriter_name_or_path or "").strip():
            raise ConfigError("model.rewriter_name_or_path is required (full 8B rewriter)")
        if not str(self.model.teacher_name_or_path or "").strip():
            raise ConfigError("model.teacher_name_or_path is required (8B OPD teacher)")
        if self.inner.mode == "opsd":
            rewriter = str(self.model.rewriter_name_or_path).rstrip("/")
            solver = str(self.model.name_or_path).rstrip("/")
            if rewriter != solver:
                raise ConfigError(
                    "inner.mode=opsd requires rewriter_name_or_path == model.name_or_path; "
                    "use inner.mode=imitation with a separate 8B rewriter"
                )
        assert_dataset_allowed(self.data.train_path, field="data.train_path")
        assert_dataset_allowed(self.data.eval_path, field="data.eval_path")
        if self.warmup.enabled:
            if self.warmup.sft_epochs < 1 or self.warmup.sft_batch_size < 1:
                raise ConfigError("warmup.sft_epochs and sft_batch_size must be >= 1")
            if not self.warmup.pairs_path:
                if self.warmup.target_pairs < 1 or self.warmup.max_problems < 1:
                    raise ConfigError("warmup.target_pairs and max_problems must be >= 1")
                if self.warmup.target_pairs > self.warmup.max_problems:
                    raise ConfigError("warmup.target_pairs cannot exceed warmup.max_problems")


def _make(cls: type, mapping: Mapping[str, Any], section: str) -> Any:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if mapping is None:
        mapping = {}
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"Config section '{section}' must be a mapping, got {type(mapping).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(mapping) - known
    if unknown:
        raise ConfigError(f"Unknown key(s) {sorted(unknown)} in config section '{section}'. "
                          f"Valid keys: {sorted(known)}")
    return cls(**dict(mapping))


def build_config(raw: Mapping[str, Any]) -> PSSDConfig:
    """Build a validated PSSDConfig from a plain dict (e.g. parsed YAML)."""
    raw = dict(raw or {})
    sections = {
        "model": ModelConfig, "lora": LoraSettings, "rewrite": RewriteSampling,
        "rollout": RolloutSampling, "reward": RewardSettings, "outer": OuterLoopSettings,
        "inner": InnerLoopSettings, "train": TrainSettings, "warmup": WarmupSettings,
        "data": DataSettings, "wandb": WandbSettings,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise ConfigError(f"Unknown top-level config section(s): {sorted(unknown)}. "
                          f"Valid sections: {sorted(sections)}")
    kwargs = {name: _make(cls, raw.get(name, {}), name) for name, cls in sections.items()}
    cfg = PSSDConfig(**kwargs)
    if cfg.rewrite.demo_model_name_or_path in ("", "null", "None"):
        cfg.rewrite.demo_model_name_or_path = None
    if cfg.warmup.pairs_path in ("", "null", "None"):
        cfg.warmup.pairs_path = None
    if cfg.model.teacher_name_or_path in ("", "null", "None", None):
        cfg.model.teacher_name_or_path = cfg.model.rewriter_name_or_path
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> PSSDConfig:
    """Load and validate a PSSDConfig from a YAML file."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return build_config(raw)


def config_to_dict(cfg: PSSDConfig) -> dict:
    """Serialize a config back to a plain dict (for logging / checkpoints)."""
    def convert(obj: Any) -> Any:
        if is_dataclass(obj):
            return {f.name: convert(getattr(obj, f.name)) for f in fields(obj)}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    return convert(cfg)
