"""Shared utilities: seeding, logging, JSONL metric writing, small math helpers."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "pssd", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger; handlers are attached only once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (and CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(spec: str) -> torch.device:
    """Resolve a device spec ('auto' | 'cuda' | 'cuda:N' | 'cpu') to a torch.device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def resolve_dtype(spec: str, device: torch.device) -> torch.dtype:
    """Resolve a dtype spec ('auto' | 'bfloat16' | 'float16' | 'float32')."""
    if spec == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if spec not in mapping:
        raise ValueError(f"Unknown dtype spec: {spec!r} (expected one of {sorted(mapping)} or 'auto')")
    return mapping[spec]


class JsonlWriter:
    """Append-only JSONL writer used for training / evaluation metrics."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: Mapping) -> None:
        self._fh.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    """Rank transform with average ranks for ties (1-based)."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    sorted_v = v[order]
    ranks_sorted = np.empty(len(v), dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks_sorted[i : j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    ranks = np.empty(len(v), dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation (scipy-free, average ranks for ties)."""
    if len(a) != len(b):
        raise ValueError("spearman: input lengths differ")
    if len(a) < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def mean_or_nan(values: Iterable[float]) -> float:
    """Mean of an iterable, NaN if empty (safe for metric aggregation)."""
    values = list(values)
    return float(np.mean(values)) if values else float("nan")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean flag from the environment ('1'/'true'/'yes')."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes"}
