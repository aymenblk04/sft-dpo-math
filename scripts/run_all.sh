#!/usr/bin/env bash
# Full pipeline + ablations. Roughly 6-8 GPU hours on one A100 for 0.5B.
set -euo pipefail

BASE_MODEL=$(python -c "import yaml; print(yaml.safe_load(open('configs/sft.yaml'))['model_name'])")

python scripts/evaluate.py --model "$BASE_MODEL" --out results/base

# 1. SFT
python scripts/train_sft.py --config configs/sft.yaml
python scripts/evaluate.py --model outputs/sft/merged --out results/sft

# 2. verified pairs ("hard" negatives, and a random-negatives control)
python scripts/build_pairs.py --config configs/sampling.yaml
python scripts/build_pairs.py --config configs/sampling.yaml strategy=random \
  output_path=data/generated/pairs_random.jsonl

# 3. DPO runs
dpo() {
  local name=$1; shift
  echo "=== $name ==="
  python scripts/train_dpo.py --config configs/dpo.yaml output_dir="outputs/$name" "$@"
  python scripts/evaluate.py --model "outputs/$name/merged" --out "results/$name"
}

dpo dpo                                         # beta=0.1, sigmoid, hard negatives
dpo dpo_beta0.05     train.beta=0.05
dpo dpo_beta0.5      train.beta=0.5
dpo dpo_ipo          "train.loss_type=[ipo]"
dpo dpo_random_neg   pairs_path=data/generated/pairs_random.jsonl
