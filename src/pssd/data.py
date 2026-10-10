"""Problem dataset: JSONL with fields {problem, reference_solution, answer?, id?}.

- `problem`            : the question x shown to the rewriter/solver.
- `reference_solution` : privileged y* (CoT + final answer); scoring-side only.
- `answer` (optional)  : gold final answer used for accuracy-based evaluation;
                         if absent we try to extract it from the reference.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from pssd.answers import extract_final_answer


@dataclass(frozen=True)
class Problem:
    problem: str
    reference_solution: str
    answer: Optional[str] = None
    id: str = ""


def load_problems(path: str | Path, max_problems: Optional[int] = None) -> List[Problem]:
    """Load and validate problems from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    problems: List[Problem] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from exc
            problem = str(record.get("problem", "")).strip()
            reference = str(record.get("reference_solution", "")).strip()
            if not problem or not reference:
                raise ValueError(f"{path}:{lineno}: both 'problem' and 'reference_solution' are required")
            answer = record.get("answer")
            if answer is None:
                answer = extract_final_answer(reference)
            problems.append(Problem(
                problem=problem,
                reference_solution=reference,
                answer=str(answer).strip() if answer is not None else None,
                id=str(record.get("id", f"{path.stem}-{lineno}")),
            ))
            if max_problems is not None and len(problems) >= max_problems:
                break
    if not problems:
        raise ValueError(f"Dataset is empty: {path}")
    return problems


class ProblemSampler:
    """Endless shuffled sampler over a problem list (reshuffles per epoch)."""

    def __init__(self, problems: List[Problem], seed: int = 0) -> None:
        if not problems:
            raise ValueError("ProblemSampler needs a non-empty problem list")
        self._problems = list(problems)
        self._rng = random.Random(seed)
        self._iter = self._make_iter()

    def _make_iter(self) -> Iterator[Problem]:
        while True:
            epoch = list(self._problems)
            self._rng.shuffle(epoch)
            yield from epoch

    def sample_batch(self, batch_size: int) -> List[Problem]:
        return [next(self._iter) for _ in range(batch_size)]
