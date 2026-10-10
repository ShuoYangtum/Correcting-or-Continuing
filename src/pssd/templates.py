"""All prompt templates used by PSSD, plus structural anti-leakage guards.

Three roles share one base model (doc Sec. 3.1):

- Rewriter q_phi   : sees ONLY the problem x (never the reference solution).
- Solver p_theta   : sees the rewrite r; frozen.
- Teacher p_T      : sees (x, y*); scores rollouts token by token, never generates.

The leakage guard is structural: every prompt that is allowed to reach the
rewriter is checked against the reference solution before use.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

Message = Dict[str, str]


class LeakageError(RuntimeError):
    """Raised when privileged information would enter a non-privileged context."""


# --- Rewriter (doc Sec. 3.1, training and inference use the same template) ---

_REWRITER_RULES_STRICT = (
    "Hard rules:\n"
    "(1) Information may only grow or stay: keep EVERY number, unit, constraint, "
    "name, and condition from the original. You may drop redundant wording, but "
    "never drop a quantity or condition needed to solve the problem.\n"
    "(2) If the problem contains a figure/diagram (e.g. [asy]...[/asy] or ASCII art), "
    "either keep the figure intact OR explicitly copy every labeled value from the "
    "figure into plain text (examples: r=3, 124 degrees). Do not delete figure-only "
    "information.\n"
    "(3) Do NOT add any fact, number, hint, or derivation step that is not already "
    "present in the original (including inside figures).\n"
    "(4) Do NOT solve the problem.\n"
    "(5) Keep the rewrite within about 1.5x the original length.\n"
)

_REWRITER_RULES_HINT = (
    "Hard rules:\n"
    "(1) Keep EVERY number, unit, constraint, name, and condition from the original. "
    "Never drop a quantity needed to solve the problem.\n"
    "(2) If the problem contains a figure/diagram (e.g. [asy]...[/asy] or ASCII art), "
    "either keep the figure intact OR explicitly copy every labeled value from the "
    "figure into plain text.\n"
    "(3) Keep the PROBLEM STEM close to the original wording and length. Light cleanup "
    "is fine; do not paraphrase at length, add worked steps, or expand the stem into "
    "an essay. If the original is already clear, copy it with only minor edits.\n"
    "(4) Extra guidance goes ONLY after the stem in a Hint section (at most 3 short "
    "bullets, one line each). Hints may name methods or what to compute first "
    "(e.g. convert bases, use similarity, count cases), but must NOT introduce new "
    "numeric facts or solve the problem.\n"
    "(5) Do NOT give the final answer, intermediate numeric results that solve the "
    "problem, or a full step-by-step derivation.\n"
    "(6) Do NOT use <think> tags or hidden chain-of-thought. Write plain text only.\n"
    "(7) Stem length should stay within about 2x the original. The Hint is extra and "
    "must stay short — a few bullets, not a solution.\n"
)

REWRITER_SYSTEM = (
    "You are a prompt canonicalizer. Rewrite the given problem so that it is "
    "maximally clear and easy for a language model to solve.\n"
    + _REWRITER_RULES_STRICT
    + "Output ONLY the rewritten problem."
)

REWRITER_SYSTEM_HINT = (
    "You are a prompt rewriter that may add brief strategy hints. Keep the problem "
    "statement close to the original; put strategy only in Hint, not in the stem.\n"
    + _REWRITER_RULES_HINT
    + "Output format:\n"
    "1) The problem stem (required): near-copy or light cleanup of the original.\n"
    "2) Optional blank line then 'Hint:' followed by at most 3 short bullets.\n"
    "Do not output anything else."
)

# Two-stage variant: deconstruction rationale + solver-facing rewrite body.
# Only the <rewrite> body is shown to the frozen solver (Acc / teacher_lik).
REWRITER_SYSTEM_TWO_STAGE = (
    "You are a prompt deconstructor. First briefly plan how to clarify the problem, "
    "then emit a clean rewritten problem for a solver.\n"
    + _REWRITER_RULES_STRICT
    + "Output format (exact tags, nothing else outside them):\n"
    "<rationale>\n"
    "Short plan: key quantities, easy-to-miss conditions, and how you will reorder/"
    "clarify the wording. Do NOT give the final answer.\n"
    "</rationale>\n"
    "<rewrite>\n"
    "The rewritten problem only (what the solver should see).\n"
    "</rewrite>"
)

REWRITER_SYSTEM_TWO_STAGE_HINT = (
    "You are a prompt deconstructor. First briefly plan clarification + optional "
    "strategy hints, then emit the solver-facing text.\n"
    + _REWRITER_RULES_HINT
    + "Output format (exact tags, nothing else outside them):\n"
    "<rationale>\n"
    "Short plan only. Do NOT give the final answer.\n"
    "</rationale>\n"
    "<rewrite>\n"
    "Problem stem (close to the original), optionally ending with 'Hint:' and "
    "at most 3 short bullets.\n"
    "</rewrite>"
)


def rewriter_system_prompt(two_stage: bool = False, style: str = "strict") -> str:
    style = (style or "strict").lower()
    if style not in ("strict", "hint"):
        raise ValueError(f"Unknown rewriter style: {style!r}")
    if two_stage:
        return REWRITER_SYSTEM_TWO_STAGE_HINT if style == "hint" else REWRITER_SYSTEM_TWO_STAGE
    return REWRITER_SYSTEM_HINT if style == "hint" else REWRITER_SYSTEM


# Few-shot demos (optional ablation only; mainline configs keep few_shot=false).
# Short/easy demos transfer poorly to contest math (over-compress + Hint habit).
# Source: OPSD held-out pool = Openthoughts_math_30k_opsd minus pilot train_2k.
# IDs: opsd30k-23693, opsd30k-23309, opsd30k-21940.
REWRITE_FEW_SHOT_STRICT: List[Tuple[str, str]] = [
    (
        "What is the sum of the whole-number factors of 24?",
        "Find the sum of all positive whole-number factors of 24.",
    ),
    (
        "The hypotenuse of an isosceles right triangle is $4\\sqrt{2}$ units. "
        "How many square units are in the area of the triangle?",
        "An isosceles right triangle has hypotenuse $4\\sqrt{2}$. What is its area in square units?",
    ),
    (
        "Solve for $r$: $\\frac{r+9}{r-3} = \\frac{r-2}{r+5}$",
        "Solve the equation $\\frac{r+9}{r-3} = \\frac{r-2}{r+5}$ for the unknown $r$.",
    ),
]

REWRITE_FEW_SHOT_HINT: List[Tuple[str, str]] = [
    (
        "What is the sum of the whole-number factors of 24?",
        "Find the sum of all positive whole-number factors of 24.\n"
        "\n"
        "Hint:\n"
        "- List the positive factors of 24 by checking divisors up to 24.\n"
        "- Add those factors together.",
    ),
    (
        "The hypotenuse of an isosceles right triangle is $4\\sqrt{2}$ units. "
        "How many square units are in the area of the triangle?",
        "An isosceles right triangle has hypotenuse $4\\sqrt{2}$. What is its area in square units?\n"
        "\n"
        "Hint:\n"
        "- In an isosceles right triangle the legs are equal; relate a leg to the hypotenuse "
        "using the $\\sqrt{2}$ factor.\n"
        "- Then compute area as one-half times leg squared.",
    ),
    (
        "Solve for $r$: $\\frac{r+9}{r-3} = \\frac{r-2}{r+5}$",
        "Solve the equation $\\frac{r+9}{r-3} = \\frac{r-2}{r+5}$ for the unknown $r$.\n"
        "\n"
        "Hint:\n"
        "- Cross-multiply to clear the fractions (watch excluded values that zero a denominator).\n"
        "- Expand, collect like terms, and solve the resulting linear equation for $r$.",
    ),
]


def rewriter_few_shot_examples(style: str = "strict") -> List[Tuple[str, str]]:
    style = (style or "strict").lower()
    if style == "hint":
        return list(REWRITE_FEW_SHOT_HINT)
    return list(REWRITE_FEW_SHOT_STRICT)


def rewriter_messages(
    problem: str,
    two_stage: bool = False,
    style: str = "strict",
    few_shot: bool = False,
) -> List[Message]:
    """Chat messages for the rewriter. Context contains ONLY the problem (+ optional demos)."""
    # Drop dataset answer-format preambles so the rewriter is not primed to emit \\boxed.
    problem = strip_opsd_answer_preamble(problem)
    messages: List[Message] = [
        {"role": "system", "content": rewriter_system_prompt(two_stage, style=style)},
    ]
    if few_shot:
        for orig, rewrite in rewriter_few_shot_examples(style):
            messages.append({"role": "user", "content": strip_opsd_answer_preamble(orig)})
            messages.append({"role": "assistant", "content": rewrite})
    messages.append({"role": "user", "content": problem})
    return messages


# --- Solver: standard chat template, user content is the (rewritten) problem ---

_OPSD_BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def solver_messages(
    problem_or_rewrite: str,
    *,
    append_boxed_instruction: bool = False,
) -> List[Message]:
    content = problem_or_rewrite
    if append_boxed_instruction:
        content = f"{problem_or_rewrite}\n\n{_OPSD_BOXED_INSTRUCTION}"
    return [{"role": "user", "content": content}]


# --- Privileged solver prefix (eta_acc upper anchor ONLY; never used for OPD KL)
# y* is shown to the frozen 1.7B *solver* for the privileged accuracy condition.
# The 8B OPD teacher scores the same solver-facing prompt as the student.

TEACHER_PREFIX_TEMPLATE = (
    "Problem: {problem}\n"
    "Here is a reference solution:\n"
    "{reference}\n"
    "After understanding the reference solution, please try to solve this "
    "problem using your own approach below:\n"
    "Answer: "
)


def teacher_prefix(problem: str, reference_solution: str) -> str:
    return TEACHER_PREFIX_TEMPLATE.format(problem=problem, reference=reference_solution)


# --- Inner-loop rewrite teacher (doc Sec. 2.3 / 3.1) --------------------------
# Privileged info here is "one rewrite proven good"; it never contains y*.

def inner_teacher_messages(
    problem: str,
    winning_rewrite: str,
    two_stage: bool = False,
    style: str = "strict",
    few_shot: bool = False,
) -> List[Message]:
    """Inner-loop imitation demo. ``winning_rewrite`` may be two-stage tagged."""
    label = "Example output" if two_stage else "Rewrite"
    user = (
        "Here is an example of a high-quality rewrite.\n"
        f"Original: {problem}\n"
        f"{label}: {winning_rewrite}\n"
        "Now rewrite the same problem again, in the same quality.\n"
        f"Original: {problem}"
    )
    messages: List[Message] = [
        {"role": "system", "content": rewriter_system_prompt(two_stage, style=style)},
    ]
    if few_shot:
        for orig, rewrite in rewriter_few_shot_examples(style):
            messages.append({"role": "user", "content": orig})
            messages.append({"role": "assistant", "content": rewrite})
    messages.append({"role": "user", "content": user})
    return messages


# --- Added-content discriminator (doc Sec. 3.4, constraint #3) ----------------

DISCRIMINATOR_SYSTEM = (
    "You are a LENIENT verifier. Flag only clear answer leakage or truly NEW "
    "quantitative facts. Answer with exactly one word: Yes or No."
)


def discriminator_messages(problem: str, rewrite: str) -> List[Message]:
    user = (
        f"Original problem:\n{problem}\n\n"
        f"Rewritten problem:\n{rewrite}\n\n"
        "Answer Yes ONLY if the rewrite clearly introduces a new number, fact, "
        "hint, or solution step that cannot be found anywhere in the original "
        "(including inside figures / [asy] blocks / labels).\n"
        "Answer No for: rephrasing, reordering, clarifying wording, or moving "
        "values that already appear in the original (e.g. writing 'radius is 3' "
        "when the figure already labels r=3).\n"
        "When unsure, answer No.\n"
        "One word only: Yes or No."
    )
    return [
        {"role": "system", "content": DISCRIMINATOR_SYSTEM},
        {"role": "user", "content": user},
    ]


# --- Privileged solving prompt (used only for eta_acc's Acc(privileged)) ------

def privileged_solver_messages(problem: str, reference_solution: str) -> List[Message]:
    user = (
        f"Problem: {problem}\n"
        "Here is a reference solution:\n"
        f"{reference_solution}\n"
        "After understanding the reference solution, please try to solve this "
        "problem using your own approach."
    )
    return [{"role": "user", "content": user}]


# --- Structural leakage guard (doc Sec. 3.4, constraint #1) --------------------

_MIN_GUARD_LINE_LEN = 20


def assert_no_leakage(
    rendered_prompt: str,
    reference_solution: str,
    allowed_text: str = "",
) -> None:
    """Assert that no substantial part of y* appears in a non-privileged prompt.

    Checks the full reference and every reasonably long line of it.  Raises
    LeakageError on violation; called on every rewriter prompt during training.

    Lines that already occur in ``allowed_text`` (typically the original problem
    ``x``) are ignored: math statements often repeat equations from the problem
    inside the reference CoT, which must not be treated as leakage.
    """
    ref = (reference_solution or "").strip()
    if not ref:
        return
    allowed = allowed_text or ""
    if ref in rendered_prompt and ref not in allowed:
        raise LeakageError("Reference solution found verbatim in a non-privileged prompt.")
    for line in ref.splitlines():
        line = line.strip()
        if len(line) < _MIN_GUARD_LINE_LEN:
            continue
        if allowed and line in allowed:
            continue
        if line in rendered_prompt:
            raise LeakageError(
                f"Reference-solution line leaked into a non-privileged prompt: {line[:60]!r}..."
            )


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK_RE = re.compile(r"<think>.*\Z", flags=re.DOTALL | re.IGNORECASE)
_STRAY_THINK_CLOSE_RE = re.compile(r"</think>", flags=re.IGNORECASE)
_SOLUTION_LEAK_RE = re.compile(
    r"(\\boxed\b|####\b|\\answer\b|final answer\s*[:=])",
    flags=re.IGNORECASE,
)
# OPSD / OpenThoughts answer-format instruction glued onto Question text.
_OPSD_ANSWER_PREAMBLE_RE = re.compile(
    r"^\s*Return your final response within\s*\\boxed\{\}\s*\.?\s*",
    flags=re.IGNORECASE,
)
_TRAILING_FINAL_ANSWER_RE = re.compile(
    r"(?:\n|\r|\s)*final answer\s*[:=]\s*.*$",
    flags=re.IGNORECASE | re.DOTALL,
)
_TRAILING_HASH_ANSWER_RE = re.compile(
    r"(?:\n|\r|\s)*####\s*.*$",
    flags=re.DOTALL,
)
_RATIONALE_BLOCK_RE = re.compile(
    r"<rationale>(.*?)</rationale>", flags=re.DOTALL | re.IGNORECASE
)
_REWRITE_BLOCK_RE = re.compile(
    r"<rewrite>(.*?)</rewrite>", flags=re.DOTALL | re.IGNORECASE
)


def strip_opsd_answer_preamble(text: str) -> str:
    """Remove ``Return your final response within \\boxed{}.`` style prefixes."""
    return _OPSD_ANSWER_PREAMBLE_RE.sub("", text or "").strip()


def _matching_brace_end(text: str, open_idx: int) -> int:
    """Return index of closing ``}`` matching ``text[open_idx]=='{'``, or -1."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return -1
    depth = 0
    for i in range(open_idx, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def strip_trailing_boxed(text: str) -> str:
    """Force-remove a trailing ``\\boxed{...}`` (possibly repeated) from rewrite text."""
    text = (text or "").rstrip()
    while True:
        idx = text.rfind("\\boxed")
        if idx < 0:
            break
        # Only strip when \\boxed is a trailing artifact (whitespace after the close).
        brace = text.find("{", idx)
        if brace < 0:
            # Bare trailing \\boxed with no args.
            if text[idx:].strip() == "\\boxed":
                text = text[:idx].rstrip()
                continue
            break
        end = _matching_brace_end(text, brace)
        if end < 0:
            # Truncated \\boxed{... — drop from \\boxed to EOS.
            if not text[idx:brace].strip().lower().replace("\\boxed", ""):
                text = text[:idx].rstrip()
            break
        if text[end + 1 :].strip():
            break  # not trailing
        text = text[:idx].rstrip()
    return text


def strip_trailing_answer_artifacts(text: str) -> str:
    """Remove trailing answer dumps: ``\\boxed{...}``, ``Final answer:``, ``####``."""
    text = strip_trailing_boxed(text)
    text = _TRAILING_FINAL_ANSWER_RE.sub("", text or "").rstrip()
    text = _TRAILING_HASH_ANSWER_RE.sub("", text).rstrip()
    # Preamble sometimes copied into the rewrite body (not only as a prefix).
    text = strip_opsd_answer_preamble(text)
    return text.strip()


def clean_rewrite_text(text: str) -> str:
    """Solver-facing cleanup: strip think residue, then trailing answer artifacts."""
    return strip_trailing_answer_artifacts(strip_think_blocks(text))


def strip_think_blocks(text: str) -> str:
    """Remove thinking residue from decoded text.

    Handles both complete ``<think>...</think>`` blocks and *truncated*
    unclosed ``<think>...`` spans (common when max_new_tokens cuts mid-thought).
    """
    text = _THINK_BLOCK_RE.sub("", text or "")
    text = _UNCLOSED_THINK_RE.sub("", text)
    text = _STRAY_THINK_CLOSE_RE.sub("", text)
    return text.strip()


def looks_like_solution_leak(text: str) -> bool:
    """True if rewrite body looks like a solution dump rather than a problem restatement.

    Call after ``clean_rewrite_text`` / trailing-box strip. Remaining mid-text
    ``\\boxed`` / ``Final answer:`` / ``####`` still count as leaks.
    """
    return bool(_SOLUTION_LEAK_RE.search(text or ""))


def parse_two_stage_rewrite(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse ``<rationale>...</rationale><rewrite>...</rewrite>``.

    Returns ``(rationale, rewrite_body)``; either may be ``None`` if the tag is missing.
    """
    rat = _RATIONALE_BLOCK_RE.search(text or "")
    rew = _REWRITE_BLOCK_RE.search(text or "")
    rationale = rat.group(1).strip() if rat else None
    body = rew.group(1).strip() if rew else None
    return rationale, body


def extract_rewrite_body(text: str, *, require_tags: bool = False) -> Tuple[str, bool]:
    """Extract the solver-facing rewrite body from a (possibly two-stage) generation.

    When ``require_tags`` is True (training), missing/empty ``<rewrite>`` → ``("", False)``.
    When False (inference), fall back to dropping rationale blocks, then the full text.
    """
    text = (text or "").strip()
    _, body = parse_two_stage_rewrite(text)
    if body:
        return body, True
    if require_tags:
        return "", False
    cleaned = _RATIONALE_BLOCK_RE.sub("", text).strip()
    fallback = cleaned or text
    return fallback, bool(fallback)
