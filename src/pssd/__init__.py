"""PSSD: Prompt-Space Self-Distillation.

Distill a privileged teacher distribution into the *input prompt* instead of
the model weights: the solver is fully frozen, and an instance-level prompt
rewriter (same base model + LoRA) is trained so that the frozen solver's
answer distribution, conditioned on the rewritten prompt, approaches the
privileged teacher distribution p_theta(. | x, y*).

Package layout
--------------
- config.py       typed configuration (dataclasses + YAML loading)
- templates.py    all prompt templates + structural leakage guards
- data.py         JSONL problem dataset
- answers.py      final-answer extraction and matching
- modeling.py     single-base-model engine (rewriter / solver / teacher roles)
- rewards.py      token-level distribution reward, group-relative advantages
- constraints.py  anti-leakage penalties and diagnostics
- trainer.py      dual-loop trainer (outer GRPO-style PG + inner OPSD)
- eta.py          elicitation fraction (eta_acc / eta_KL) measurement
- utils.py        seeding, logging, JSONL metrics, small math helpers
"""

__version__ = "0.1.0"

from pssd.config import PSSDConfig, load_config  # noqa: F401
