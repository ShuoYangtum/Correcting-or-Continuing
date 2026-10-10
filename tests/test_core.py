"""Model-free unit tests for PSSD's pure logic (rewards, answers, guards, config).

Run with:  pytest tests/ -q
These tests require no GPU and download no model.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pssd.answers import answers_match, extract_final_answer, grade_output
from pssd.config import ConfigError, build_config
from pssd.constraints import (
    coverage_penalty,
    extract_critical_numbers,
    length_penalty,
    new_ngram_rate,
    number_coverage,
    reference_overlap,
)
from pssd.rewards import (
    clipped_token_reward,
    group_advantages,
    group_is_informative,
    mean_token_logprob,
    relative_kl_reward,
)
from pssd.templates import (
    LeakageError,
    assert_no_leakage,
    extract_rewrite_body,
    inner_teacher_messages,
    parse_two_stage_rewrite,
    rewriter_messages,
    solver_messages,
    strip_think_blocks,
    teacher_prefix,
)
from pssd.utils import spearman


# --------------------------- rewards --------------------------------------- #

def test_clipped_token_reward_basic():
    lp_t = torch.tensor([-1.0, -2.0, -3.0])
    lp_s = torch.tensor([-2.0, -2.0, -1.0])
    # diffs: 1, 0, -2 -> mean = -1/3
    assert clipped_token_reward(lp_t, lp_s, clip_c=5.0) == pytest.approx(-1 / 3)


def test_clipped_token_reward_clips_heavy_tails():
    lp_t = torch.tensor([100.0, -100.0])
    lp_s = torch.tensor([0.0, 0.0])
    # diffs clipped to +5, -5 -> mean 0
    assert clipped_token_reward(lp_t, lp_s, clip_c=5.0) == pytest.approx(0.0)


def test_clipped_token_reward_empty_is_nan():
    assert np.isnan(clipped_token_reward(torch.empty(0), torch.empty(0), 5.0))


def test_clipped_token_reward_length_mismatch_uses_sequence_means():
    lp_t = torch.tensor([-1.0, -3.0])
    lp_s = torch.tensor([-2.0, -2.0, -2.0])
    # mean_T=-2, mean_S=-2 → 0
    assert clipped_token_reward(lp_t, lp_s, clip_c=5.0) == pytest.approx(0.0)


def test_relative_kl_reward():
    assert relative_kl_reward(-0.1, -0.3) == pytest.approx(0.2)
    assert relative_kl_reward(-0.3, -0.1) == pytest.approx(-0.2)
    assert np.isnan(relative_kl_reward(float("nan"), -0.1))


def test_mean_token_logprob():
    assert mean_token_logprob(torch.tensor([-1.0, -3.0])) == pytest.approx(-2.0)
    assert np.isnan(mean_token_logprob(torch.empty(0)))


def test_rewrite_score_acc_lik_reward():
    from pssd.rewards import RewriteScore
    s = RewriteScore(
        text="x",
        acc_score=0.5,
        teacher_lik=-1.0,
        dense_reward="acc_lik",
        acc_weight=1.0,
        lik_weight=0.1,
    )
    assert s.dense_score == pytest.approx(0.5 + 0.1 * (-1.0))
    assert s.reward == pytest.approx(0.4)


def test_group_advantages_zero_mean_unit_scale():
    adv = group_advantages([1.0, 2.0, 3.0, 4.0])
    assert adv.mean() == pytest.approx(0.0, abs=1e-8)
    assert adv.std() == pytest.approx(1.0, abs=1e-3)
    assert adv[0] < adv[-1]


def test_group_informative_filtering():
    assert not group_is_informative([0.5, 0.5, 0.5], std_eps=1e-4)  # DAPO-style skip
    assert group_is_informative([0.1, 0.9], std_eps=1e-4)
    assert not group_is_informative([0.1, float("nan")], std_eps=1e-4)


# --------------------------- answers --------------------------------------- #

def test_extract_boxed_nested():
    assert extract_final_answer(r"thus \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"


def test_extract_hash_and_last_number():
    assert extract_final_answer("blah blah\n#### 42") == "42"
    assert extract_final_answer("we get 7 then 13 finally") == "13"


def test_extract_answer_is_empty_capture_does_not_crash():
    # "answer is" with empty capture previously crashed on splitlines()[0].
    out = extract_final_answer("answer is:\n\n\nthen we conclude 17")
    assert out == "17"


def test_answers_match_numeric_and_formats():
    assert answers_match("1,000", "1000.0")
    assert answers_match("1/2", "0.5")
    assert not answers_match("12", "13")
    assert grade_output("The answer is 36.", "36", grader="regex")


def test_grade_output_math_verify_opsd_style():
    # OPSD: only \\boxed counts; bare "answer is" does not.
    assert grade_output(r"reasoning... \boxed{204}", "204", grader="math_verify")
    assert not grade_output("The answer is 204.", "204", grader="math_verify")
    assert not grade_output(r"\boxed{203}", "204", grader="math_verify")


def test_inference_rewrite_gates_and_fallback():
    from pssd.config import build_config
    from pssd.rewrite_gates import select_rewrite_with_retries, validate_rewrite

    cfg = build_config(
        {
            "rewrite": {
                "inference_validate": True,
                "max_rewrite_retries": 2,
                "min_stem_ratio": 0.5,
                "max_hint_blocks": 1,
                "max_hint_bullets": 3,
                "max_length_ratio": 2.0,
                "ban_assertive_hints": True,
                "invalidate_solution_leak": True,
            },
            "reward": {"invalidate_missing_numbers": True},
            "outer": {"enabled": False},
            "inner": {"enabled": True, "mode": "imitation"},
        }
    )
    orig = (
        "Every morning Aya walks 9 kilometers. When she walks at speed s the walk "
        "takes 4 hours including t minutes at a shop. When she walks s+2 the walk "
        "takes 2 hours and 24 minutes including t minutes at the shop. Find the "
        "number of minutes when she walks at s+1/2."
    )
    bad_drop = (
        "Find the minutes when she walks at s+1/2.\n\n"
        "Hint:\n- Set up equations.\n- Solve for s and t."
    )
    assert any(r.startswith("missing_numbers") for r in validate_rewrite(orig, bad_drop, cfg))
    assert any(r.startswith("stem_too_short") for r in validate_rewrite(orig, bad_drop, cfg))

    bad_dup = (
        orig
        + "\n\nHint:\n- Use modular arithmetic.\n\nHint:\n- Use modular arithmetic."
    )
    assert any(r.startswith("too_many_hint_blocks") for r in validate_rewrite(orig, bad_dup, cfg))

    bad_assert = orig + "\n\nHint:\n- The answer must be 204."
    assert "assertive_hint" in validate_rewrite(orig, bad_assert, cfg)

    good = (
        orig
        + "\n\nHint:\n- Set up two equations from the given speeds.\n"
        "- Solve for s and t, then evaluate at s+1/2."
    )
    assert validate_rewrite(orig, good, cfg) == []

    # Short MATH stem + Hint used to fail a *full-text* 2x cap; stem/Hint split must accept it.
    short = r"Simplify $\sqrt{242}$."
    short_hinted = (
        short
        + "\n\nHint:\n- Factor 242 into primes.\n"
        "- Pull out perfect-square factors.\n"
        "- Use $\\sqrt{ab}=\\sqrt{a}\\sqrt{b}$."
    )
    assert len(short_hinted) / len(short) > 2.0
    assert validate_rewrite(short, short_hinted, cfg) == []

    long_stem = orig + (" Please restate every condition in more words." * 12)
    assert any(r.startswith("stem_too_long") for r in validate_rewrite(orig, long_stem, cfg))

    huge_hint = orig + "\n\nHint:\n- " + ("repeat this strategy phrase, " * 80)
    assert any(r.startswith("hint_too_long") for r in validate_rewrite(orig, huge_hint, cfg))

    from pssd.rewrite_gates import filter_rewrite_texts

    filt = filter_rewrite_texts(orig, [bad_drop, bad_dup, good, ""], cfg, two_stage=False)
    assert good.strip() in filt.accepted or any("Set up two equations" in a for a in filt.accepted)
    assert filt.n_empty_or_unparsed >= 1
    assert filt.reject_family_counts.get("missing_numbers", 0) >= 1
    assert "missing_numbers" in filt.reject_examples or "stem_too_short" in filt.reject_examples

    calls = {"n": 0}

    def gen(attempt: int) -> str:
        calls["n"] += 1
        return bad_drop

    decision = select_rewrite_with_retries(orig, cfg, gen)
    assert decision.fell_back
    assert decision.text == orig
    assert calls["n"] == 1 + cfg.rewrite.max_rewrite_retries

    calls["n"] = 0

    def gen_ok(attempt: int) -> str:
        calls["n"] += 1
        return bad_drop if attempt == 0 else good

    decision2 = select_rewrite_with_retries(orig, cfg, gen_ok)
    assert decision2.accepted and not decision2.fell_back
    assert decision2.text == good
    assert calls["n"] == 2


# --------------------------- constraints ----------------------------------- #

def test_length_penalty_only_above_budget():
    assert length_penalty(1.2, budget=1.5, weight=1.0) == 0.0
    assert length_penalty(2.0, budget=1.5, weight=2.0) == pytest.approx(1.0)


def test_ngram_diagnostics():
    x = "a b c d e f g h"
    assert new_ngram_rate(x, x) == 0.0
    assert new_ngram_rate(x, "totally different words here now") == 1.0
    assert reference_overlap("a b c d", "a b c d e") == 1.0
    assert reference_overlap("p q r s", "a b c d e") == 0.0


def test_critical_numbers_include_asy_labels_not_plot_coords():
    original = (
        "Volume is $45\\pi$. What is the height?\n"
        "[asy]\ndraw((1,0)--(1,-2));\n"
        'label("$r=3$", (2.2,0), N);\n'
        "[/asy]"
    )
    nums = extract_critical_numbers(original)
    assert "45" in nums and "3" in nums
    # bare drawing coordinates should not be required
    assert "1" not in nums and "2" not in nums


def test_number_coverage_detects_dropped_figure_label():
    original = (
        "Volume is $45\\pi$.\n[asy]\nlabel(\"$r=3$\", (0,0), N);\n[/asy]"
    )
    good = "A cylinder has volume 45 pi and radius 3. Find the height."
    bad = "A cylinder has volume 45 pi. Find the height."
    cov_g, miss_g, _ = number_coverage(original, good)
    cov_b, miss_b, missing = number_coverage(original, bad)
    assert cov_g == 1.0 and miss_g == 0.0
    assert cov_b < 1.0 and "3" in missing
    assert coverage_penalty(miss_b, weight=1.0) == pytest.approx(miss_b)


# --------------------------- leakage guards -------------------------------- #

def test_leakage_guard_raises_on_reference_in_prompt():
    ref = "The sale price is 15 * (1 - 0.20) = 12 dollars.\n#### 10"
    with pytest.raises(LeakageError):
        assert_no_leakage(f"system stuff... {ref} ...", ref)


def test_leakage_guard_passes_clean_prompt():
    ref = "Let j be Jerry's marbles; 4j = 48 so Tom has 36."
    prompt = "Rewrite the given problem: Tom has 3 times as many marbles as Jerry."
    assert_no_leakage(prompt, ref)  # must not raise


def test_leakage_guard_allows_equation_shared_with_problem():
    # CoT often restates an equation that already appears in the problem.
    problem = "Given AR(AP + AR) = AQ(AQ + AS), prove the triangles are similar."
    ref = (
        "We start from the given identity:\n"
        "AR(AP + AR) = AQ(AQ + AS)\n"
        "Then rearrange to obtain the similarity ratios.\n"
        "#### QED"
    )
    prompt = f"system\n{problem}"
    assert_no_leakage(prompt, ref, allowed_text=problem)  # must not raise
    with pytest.raises(LeakageError):
        assert_no_leakage(prompt, ref)  # without allowlist, shared line trips the guard


def test_rewriter_and_inner_templates_never_contain_reference():
    x, ref = "Some problem?", "A long enough reference solution line for the guard."
    for messages in (rewriter_messages(x), inner_teacher_messages(x, "a rewrite"), solver_messages(x)):
        joined = " ".join(m["content"] for m in messages)
        assert_no_leakage(joined, ref, allowed_text=x)
        assert ref not in joined
    # Privileged prefix is eval-only (eta_acc); OPD KL must not use it.
    assert ref in teacher_prefix(x, ref)


def test_strip_think_blocks():
    assert strip_think_blocks("<think>internal\nstuff</think>Hello") == "Hello"
    assert strip_think_blocks("no blocks") == "no blocks"
    # Truncated / unclosed think must not leak into the solver-facing rewrite.
    assert strip_think_blocks("<think>\nOkay, let's solve...\n6*7=42") == ""
    assert strip_think_blocks("<think>plan</think>\nClean problem.") == "Clean problem."


def test_looks_like_solution_leak():
    from pssd.templates import looks_like_solution_leak
    assert looks_like_solution_leak(r"Therefore \boxed{42}")
    assert looks_like_solution_leak("final answer: 7")
    assert not looks_like_solution_leak("Find the product of 6_8 and 7_8.")


def test_strip_opsd_preamble_and_trailing_boxed():
    from pssd.templates import (
        clean_rewrite_text,
        looks_like_solution_leak,
        strip_opsd_answer_preamble,
        strip_trailing_boxed,
        rewriter_messages,
    )

    raw = r"Return your final response within \boxed{}. What is 2+2?"
    assert strip_opsd_answer_preamble(raw) == "What is 2+2?"
    msgs = rewriter_messages(raw, style="hint", few_shot=False)
    assert msgs[-1]["content"] == "What is 2+2?"
    assert "Return your final" not in msgs[-1]["content"]

    with_box = (
        "What is 2+2?\n\nHint:\n- Add the numbers.\n\n"
        r"\boxed{4}"
    )
    assert strip_trailing_boxed(with_box).endswith("Add the numbers.")
    cleaned = clean_rewrite_text("<think>ignore</think>\n" + with_box)
    assert r"\boxed" not in cleaned
    assert not looks_like_solution_leak(cleaned)

    nested = "Rewrite me.\n\n" + r"\boxed{\frac{1}{2}}"
    assert strip_trailing_boxed(nested) == "Rewrite me."

    # Mid-text boxed answer still counts as a leak after trailing strip.
    mid = r"Note that \boxed{4} is wrong. What is 2+2?"
    assert looks_like_solution_leak(strip_trailing_boxed(mid))


def test_parse_two_stage_rewrite():
    text = (
        "<rationale>\nKeep the 3 and reorder.\n</rationale>\n"
        "<rewrite>\nTom has 3 marbles. How many?\n</rewrite>"
    )
    rat, body = parse_two_stage_rewrite(text)
    assert rat is not None and "reorder" in rat
    assert body == "Tom has 3 marbles. How many?"
    body_ok, ok = extract_rewrite_body(text, require_tags=True)
    assert ok and body_ok == "Tom has 3 marbles. How many?"
    missing, bad = extract_rewrite_body("just a plain rewrite", require_tags=True)
    assert not bad and missing == ""
    fallback, ok2 = extract_rewrite_body(
        "<rationale>plan</rationale>\nplain body", require_tags=False
    )
    assert ok2 and fallback == "plain body"


def test_two_stage_rewriter_template():
    msgs = rewriter_messages("x=1?", two_stage=True)
    assert "<rewrite>" in msgs[0]["content"]
    assert "Output ONLY the rewritten problem." not in msgs[0]["content"]


def test_hint_rewriter_template():
    msgs = rewriter_messages("x=1?", style="hint")
    assert "Hint:" in msgs[0]["content"]
    assert "Do NOT give the final answer" in msgs[0]["content"]
    strict = rewriter_messages("x=1?", style="strict")
    assert "Do NOT add any fact, number, hint" in strict[0]["content"]


def test_few_shot_rewriter_messages():
    zero = rewriter_messages("Real problem?", style="hint", few_shot=False)
    few = rewriter_messages("Real problem?", style="hint", few_shot=True)
    assert len(zero) == 2
    assert len(few) == 2 + 2 * 3  # system + 3*(user,assistant) + final user
    assert few[-1]["content"] == "Real problem?"
    assert few[1]["role"] == "user" and few[2]["role"] == "assistant"
    assert "Hint:" in few[2]["content"]
    assert "\\boxed" not in few[2]["content"]
    # Held-out OPSD demos.
    assert "196" not in few[1]["content"]
    assert "6_8" not in few[1]["content"]
    assert "factors of 24" in few[1]["content"] or "24" in few[1]["content"]


def test_config_rewrite_style():
    cfg = build_config({"rewrite": {"style": "hint"}, "outer": {"enabled": False}, "inner": {"enabled": True}})
    assert cfg.rewrite.style == "hint"
    with pytest.raises(ConfigError):
        build_config({"rewrite": {"style": "cot"}})
    cfg2 = build_config({"rewrite": {"few_shot": True}, "outer": {"enabled": False}, "inner": {"enabled": True}})
    assert cfg2.rewrite.few_shot


# --------------------------- config ---------------------------------------- #

def test_config_defaults_valid():
    cfg = build_config({})
    assert cfg.rewrite.k == 8 and cfg.reward.clip_c == 5.0
    assert cfg.inner.mode == "imitation"
    assert cfg.model.rewriter_name_or_path.endswith("Qwen3-8B")
    assert cfg.model.teacher_name_or_path.endswith("Qwen3-8B")
    assert cfg.train.optimizer == "adamw8bit"


def test_config_rejects_banned_dataset_paths():
    with pytest.raises(ConfigError, match="permanently banned"):
        build_config(
            {
                "outer": {"enabled": False},
                "inner": {"enabled": True},
                "data": {"train_path": "${PSSD_ROOT}/data/math500/mvp_500.jsonl"},
            }
        )
    with pytest.raises(ConfigError, match="permanently banned"):
        build_config(
            {
                "outer": {"enabled": False},
                "inner": {"enabled": True},
                "data": {
                    "train_path": "data/sample_math.jsonl",
                    "eval_path": "/tmp/eval_math50.jsonl",
                },
            }
        )


def test_config_rejects_opsd_inner_with_separate_rewriter():
    with pytest.raises(ConfigError, match="imitation"):
        build_config({"inner": {"mode": "opsd"}, "outer": {"enabled": False}})
    cfg = build_config(
        {
            "model": {
                "name_or_path": "Qwen/Qwen3-0.6B",
                "rewriter_name_or_path": "Qwen/Qwen3-0.6B",
            },
            "inner": {"mode": "opsd"},
            "outer": {"enabled": False},
        }
    )
    assert cfg.inner.mode == "opsd"


def test_config_still_parses_unused_lora_section():
    cfg = build_config(
        {
            "lora": {"r": 8, "alpha": 16},
            "outer": {"enabled": False},
            "inner": {"enabled": True},
        }
    )
    assert cfg.lora.r == 8


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        build_config({"rewrite": {"k": 8, "tempratuer": 1.0}})
    with pytest.raises(ConfigError):
        build_config({"not_a_section": {}})


def test_config_rejects_inconsistent_settings():
    with pytest.raises(ConfigError):
        build_config({"rewrite": {"k": 1}})  # outer loop needs a group
    with pytest.raises(ConfigError):
        build_config({"outer": {"enabled": False}, "inner": {"enabled": False}})


def test_config_allows_outer_off_with_inner():
    cfg = build_config({"outer": {"enabled": False}, "inner": {"enabled": True}, "rewrite": {"k": 1, "two_stage": True}})
    assert not cfg.outer.enabled and cfg.inner.enabled and cfg.rewrite.two_stage


def test_config_inner_imitation_mode():
    cfg = build_config({"inner": {"mode": "imitation", "weight": 1.0}, "outer": {"enabled": False}})
    assert cfg.inner.mode == "imitation"
    with pytest.raises(ConfigError):
        build_config({"inner": {"mode": "sft"}})


def test_sanitize_cuda_visible_devices_uuid(monkeypatch):
    from pssd.eval_vllm import sanitize_cuda_visible_devices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-57a96e2b-9815-34f9-2ab7-7be04aea5a8f")
    assert sanitize_cuda_visible_devices() == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
    assert sanitize_cuda_visible_devices() is None


def test_config_sft_pairs_path_skips_collection_bounds():
    cfg = build_config(
        {
            "outer": {"enabled": False},
            "inner": {"enabled": True, "mode": "imitation"},
            "train": {"steps": 0},
            "warmup": {
                "enabled": True,
                "pairs_path": "/tmp/sft_pairs.jsonl",
                "sft_epochs": 2,
                "sft_batch_size": 2,
            },
            "rewrite": {"demo_model_name_or_path": "Qwen/Qwen3-8B"},
        }
    )
    assert cfg.warmup.pairs_path.endswith("sft_pairs.jsonl")
    assert cfg.rewrite.demo_model_name_or_path == "Qwen/Qwen3-8B"
    assert cfg.train.steps == 0
    blank = build_config(
        {
            "outer": {"enabled": False},
            "inner": {"enabled": True},
            "rewrite": {"demo_model_name_or_path": ""},
        }
    )
    assert blank.rewrite.demo_model_name_or_path is None


# --------------------------- utils ----------------------------------------- #

def test_spearman_monotonic_and_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    rho = spearman([1, 1, 2, 3], [5, 5, 6, 7])
    assert rho == pytest.approx(1.0)


# --------------------------- prefix / OPD ---------------------------------- #

def test_icps_encoder_shapes_and_size():
    from pssd.icps import ICPSEncoder

    enc = ICPSEncoder(hidden_size=32, n_tokens=8, bottleneck=16)
    out = enc(torch.randn(32))
    assert tuple(out.shape) == (8, 32)
    batched = enc(torch.randn(2, 32))
    assert tuple(batched.shape) == (2, 8, 32)
    n_params = sum(p.numel() for p in enc.parameters())
    assert n_params < 10_000


def test_opd_teacher_prompt_never_sees_y_star():
    x = "Find 1+1."
    ystar = "PRIVILEGED_REFERENCE_SOLUTION_TOKEN"
    msgs = solver_messages(x, append_boxed_instruction=True)
    joined = " ".join(m["content"] for m in msgs)
    assert ystar not in joined
    assert x in joined
    assert r"\boxed{}" in joined
    assert ystar in teacher_prefix(x, ystar)


def test_opd_kl_source_uses_solver_chat_not_privileged_prefix():
    import inspect

    from pssd.icps import ICPSEngine

    src = inspect.getsource(ICPSEngine.opd_kl_loss_batch)
    assert "teacher_prefix" not in src
    assert "teacher_chat_ids" in src
    dispatch = inspect.getsource(ICPSEngine.opsd_kl_loss_batch)
    assert 'distill == "opd"' in dispatch
    legacy = inspect.getsource(ICPSEngine._opsd_privileged_kl_loss_batch)
    assert "teacher_prefix" in legacy


def test_advantage_gate_uses_gold_only_as_skip():
    import inspect

    from pssd.icps import ICPSEngine, ICPSSettings

    settings = ICPSSettings()
    assert settings.opsd_advantage_gate is False
    assert settings.opsd_gate_mode == "student_wrong"
    src = inspect.getsource(ICPSEngine.opd_kl_loss_batch)
    assert "opsd_advantage_gate" in src
    assert "_grade_rollout" in src
    assert "teacher_chat_ids(text)" in src
    # Gold may appear as a local for grading, but must not be teacher input.
    assert "teacher_chat_ids(gold" not in src
    assert "teacher_prefix" not in src
    grade_src = inspect.getsource(ICPSEngine._grade_rollout)
    assert "grade_output" in grade_src


def test_e2_opd_gated_yaml_validates():
    import yaml
    from dataclasses import fields

    from pssd.icps import ICPSSettings

    path = Path(__file__).resolve().parents[1] / "configs" / "train_e2_opd_gated.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    icps = dict(raw.pop("icps"))
    icps.pop("train_path", None)
    cfg = build_config(raw)
    assert "Qwen3-14B" in str(cfg.model.teacher_name_or_path)
    assert cfg.model.enable_thinking is True
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(icps) - known
    assert not unknown, unknown
    settings = ICPSSettings(**icps)
    assert settings.distill == "opd"
    assert settings.opsd_advantage_gate is True
    assert settings.encoder_input == "context"
    assert settings.n_tokens == 64


def test_prefix_opd_yaml_validates():
    import yaml
    from dataclasses import fields

    from pssd.icps import ICPSSettings

    path = Path(__file__).resolve().parents[1] / "configs" / "train_prefix_opd_14b.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    icps = dict(raw.pop("icps"))
    train_path = icps.pop("train_path")
    cfg = build_config(raw)
    assert "Qwen3-14B" in str(cfg.model.teacher_name_or_path)
    assert cfg.model.enable_thinking is False
    assert not cfg.outer.enabled
    assert cfg.reward.grader == "math_verify"
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(icps) - known
    assert not unknown, unknown
    settings = ICPSSettings(**icps)
    assert settings.distill == "opd"
    assert settings.n_tokens == 64
    assert settings.opsd_max_new_tokens == 16384
    assert settings.opsd_batch_size == 1
    assert settings.opsd_grad_accum == 2
    assert settings.eval_every == 50
    assert settings.eval_n_problems == 10
    assert settings.eval_greedy is True
    assert settings.use_vllm_base is False
    assert settings.ce_early_stop_patience > 0
    assert settings.opsd_early_stop_patience > 0
    assert "opsd_math_30k" in train_path


def test_e2b_opd_4b_yaml_validates():
    import yaml
    from dataclasses import fields

    from pssd.icps import ICPSSettings

    path = Path(__file__).resolve().parents[1] / "configs" / "train_e2b_opd_4b.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    icps = dict(raw.pop("icps"))
    icps.pop("train_path", None)
    cfg = build_config(raw)
    assert "Qwen3-4B" in str(cfg.model.teacher_name_or_path)
    assert "14B" not in str(cfg.model.teacher_name_or_path)
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(icps) - known
    assert not unknown, unknown
    settings = ICPSSettings(**icps)
    assert settings.distill == "opd"
    assert settings.opsd_advantage_gate is True
    assert settings.opsd_kl_max_tokens == 512
    assert settings.encoder_input == "context"
    assert settings.ce_batch_size == 4
    assert settings.opsd_batch_size == 4


def test_e2b_opd_full_yaml_validates():
    import yaml
    from dataclasses import fields

    from pssd.icps import ICPSSettings

    path = Path(__file__).resolve().parents[1] / "configs" / "train_e2b_opd_full.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    icps = dict(raw.pop("icps"))
    icps.pop("train_path", None)
    cfg = build_config(raw)
    assert "Qwen3-4B" in str(cfg.model.teacher_name_or_path)
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(icps) - known
    assert not unknown, unknown
    settings = ICPSSettings(**icps)
    assert settings.distill == "opd"
    assert settings.student_update == "full"
    assert settings.opsd_advantage_gate is True
    assert settings.ce_epochs == 0
    assert settings.prefix_l2_coef == 0.0
    assert settings.opsd_kl_max_tokens == 0
    assert settings.opsd_kl_seq_chunk == 512
    assert settings.opsd_max_new_tokens == 16384
    assert cfg.rollout.max_new_tokens == 16384
    assert settings.prefer_zero_if_nonpositive is False
    assert settings.use_vllm_base is True


def test_e2b_opd_revkl_yaml_validates():
    import yaml
    from dataclasses import fields

    from pssd.icps import ICPSSettings

    path = Path(__file__).resolve().parents[1] / "configs" / "train_e2b_opd_revkl.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    icps = dict(raw.pop("icps"))
    icps.pop("train_path", None)
    cfg = build_config(raw)
    assert "Qwen3-4B" in str(cfg.model.teacher_name_or_path)
    known = {f.name for f in fields(ICPSSettings)}
    unknown = set(icps) - known
    assert not unknown, unknown
    settings = ICPSSettings(**icps)
    assert settings.distill == "opd"
    assert settings.student_update == "full"
    assert settings.opsd_kl_direction == "reverse"
    assert settings.opsd_advantage_gate is False
    assert settings.opsd_early_stop_min_steps == 500
    assert settings.opsd_steps == 800
    assert settings.opsd_batch_size == 4
    assert settings.opsd_kl_max_tokens == 0
    assert settings.opsd_max_new_tokens == 16384
    assert cfg.rollout.max_new_tokens == 16384


def test_full_ft_eval_scores_original_snapshot_if_vllm_fails():
    import inspect

    from pssd.icps import ICPSEngine
    from pssd.icps_train import evaluate_icps

    ev = inspect.getsource(evaluate_icps)
    assert "accuracy_snapshot_batch" in ev
    assert "HF fallback would score the trained student as 'base'" not in ev
    src = inspect.getsource(ICPSEngine.accuracy_snapshot_batch)
    assert "cfg.model.name_or_path" not in src or "snapshot" in src
    assert "Restored trained student" in src


def test_full_ft_opd_kl_does_not_require_prefix():
    import inspect

    from pssd.icps import ICPSEngine, ICPSSettings

    settings = ICPSSettings()
    assert settings.student_update == "prefix"
    src = inspect.getsource(ICPSEngine.opd_kl_loss_batch)
    assert "uses_prefix" in src
    assert "teacher_prefix" not in src
    assert "teacher_chat_ids" in src
    stu_src = inspect.getsource(ICPSEngine._student_yhat_logits)
    assert "input_ids=stu_input" in stu_src
    assert "soft_m" in stu_src
    shard_src = inspect.getsource(ICPSEngine._opd_kl_sharded_example)
    assert "piece.backward()" in shard_src
    assert "seq_chunk" in shard_src
    assert "reverse" in shard_src
    kl_src = inspect.getsource(__import__("pssd.icps", fromlist=["_per_token_kl"])._per_token_kl)
    assert "sl.topk" in kl_src
    assert "reverse" in kl_src


def test_per_token_kl_mean_matches_chunked():
    from pssd.icps import _chunked_forward_kl, _detach_past, _per_token_forward_kl

    torch.manual_seed(0)
    stu = torch.randn(7, 11)
    tea = torch.randn(7, 11)
    tok = _per_token_forward_kl(stu, tea, clip=5.0, topk=0)
    mean = _chunked_forward_kl(stu, tea, clip=5.0, chunk=3, topk=0)
    assert tok.shape == (7,)
    assert float(tok.mean()) == pytest.approx(float(mean), rel=1e-5)
    rev = _per_token_forward_kl(stu, tea, clip=5.0, topk=0, reverse=True)
    assert (rev >= 0).all()
    assert not torch.allclose(tok, rev)
    t = torch.tensor([1.0, 2.0], requires_grad=True)
    out = _detach_past((t, t + 1))
    assert not out[0].requires_grad
    assert torch.equal(out[0], t.detach())

