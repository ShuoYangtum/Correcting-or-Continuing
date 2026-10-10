# Anonymous supplementary code

This archive contains only the scripts and configs needed to reproduce
those runs.

## Hardware (all GPU runs)

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA H200 NVL, 143771 MiB (~141 GB), SM 9.0. The node had two such GPUs; each job used one. Student and teacher shared that GPU. |
| CPU (compute node) | AMD EPYC 9654, 1 socket, 96 cores / 192 threads |
| Host RAM | 755 GiB. Jobs requested 8 CPUs and 96 GiB. |
| OS | Ubuntu 22.04.5 LTS, Linux 5.15.0-191-generic, x86_64 |
| NVIDIA driver | 580.178.04 |
| System CUDA | 12.9 (`nvcc` 12.9.41) |
| PyTorch CUDA | 12.8 (`torch==2.10.0+cu128`) |

The login node is a different machine and was not used for training or evaluation.

## Software

Python 3.11, `torch==2.10.0`, `transformers==4.57.6`, `trl==1.13.0`, `vllm==0.18.0`. See `requirements.txt`.

## Layout

```
src/pssd/          in-house trainer (prefix / full-parameter OPD)
trl/               HuggingFace TRL DistillationTrainer entry points
scripts/           eval, ceiling, data split, ICPS CLI
configs/           YAML for the isolation-study rows
tests/             CPU unit tests (`pytest tests/ -q`)
```

## What each file reproduces

| File | Experiment |
|---|---|
| `trl/phase0_train.py`, `trl/phase0_eval_gsm8k.py` | Positive control: Qwen2.5-0.5B ← 1.5B on GSM8K |
| `trl/train_opd.py` | Isolation TRL reverse-KL row and the 3-seed sweep |
| `trl/p6_train.py` | Phi-4-mini instruct ← reasoning (full band and hard subset) |
| `trl/p6_eval.py` | Greedy HF eval used for the first Phi MATH-500 / AIME numbers |
| `scripts/p7_eval.py` | vLLM eval with Phi stop ids `[200020, 199999]` (preferred) |
| `scripts/p7_ceiling.py` | Phi elicitation ceiling + hard-subset carve |
| `scripts/run_icps_mvp.py` + `configs/train_e2b_opd_*.yaml` | In-house full-FT / prefix isolation rows |
| `scripts/run_e05_difficulty.py` + `configs/eval_e05_band.yaml` | Qwen3 elicitation ceiling |
| `scripts/harvest_hard_trainset.py`, `split_hard_eval.py` | Hard-slice construction |
| `scripts/build_math_band_e0.py` | MATH band with worked solutions |

`trl/p6_train.py` accepts `--eos-token '<|end|>'`. Pass this for Phi so
generation stops on `<|end|>` (id 200020) rather than `<|endoftext|>`.

## Data (not in this archive)

JSONL with fields `id`, `problem` / `question`, `answer`, and optionally
`reference_solution`. Point `--train-data` / `--data` at local files.
Set `PSSD_DATA_DIR` if you run `scripts/p7_ceiling.py` (it writes the
hard train/eval splits there).

## Notes for reviewers

- On-policy distillation never trains on `y*`. Reference solutions appear
  only in the elicitation-ceiling privileged arm.
- TRL 1.13 `DistillationTrainer` with `beta=1.0` is reverse KL on the
  full vocabulary. `temperature` is both the sampling temperature and
  the softmax temperature of the KL.
- Isolation-table AIME scores are pass@1 at temperature 0.6, not greedy.
