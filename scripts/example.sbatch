#!/bin/bash
#SBATCH --job-name=opd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#
# Generic launcher. Set REPO_ROOT, PSSD_ROOT, HF_HOME, and the command.
# Hardware used in the paper: 1x NVIDIA H200 NVL (~141 GB), 8 CPUs, 96 GB RAM.
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
PSSD_ROOT="${PSSD_ROOT:-$REPO_ROOT}"
cd "${REPO_ROOT}"
mkdir -p logs
export HF_HOME="${HF_HOME:-$PSSD_ROOT/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_PROJECT="${WANDB_PROJECT:-opd}"
# Do not hard-code a W&B entity.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi
# Usage example:
#   python scripts/run_icps_mvp.py --config configs/train_e2b_opd_revkl.yaml
#   python trl/train_opd.py --student Qwen/Qwen3-1.7B --teacher Qwen/Qwen3-4B ...
exec "$@"
