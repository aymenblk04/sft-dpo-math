"""Step 1: supervised fine-tuning on GSM8K reference solutions.

    python scripts/train_sft.py --config configs/sft.yaml
"""

import argparse
from pathlib import Path

from sftdpo.config import load_config
from sftdpo.data import build_sft_dataset, load_raw
from sftdpo.train import save_merged, train_sft


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    raw = load_raw("train", cfg["data"].get("train_samples"), cfg.get("seed", 0))
    trainer = train_sft(cfg, build_sft_dataset(raw))
    merged = save_merged(trainer, Path(cfg["output_dir"]) / "merged")
    print(f"merged SFT model saved to {merged}")


if __name__ == "__main__":
    main()
