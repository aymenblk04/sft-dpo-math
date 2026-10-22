"""Step 3: DPO on the verified pairs.

    python scripts/train_dpo.py --config configs/dpo.yaml
    python scripts/train_dpo.py --config configs/dpo.yaml train.loss_type=[ipo] output_dir=outputs/dpo_ipo
"""

import argparse
from pathlib import Path

from datasets import load_dataset

from sftdpo.config import load_config
from sftdpo.train import save_merged, train_dpo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    pairs = load_dataset("json", data_files=cfg["pairs_path"], split="train")
    trainer = train_dpo(cfg, pairs.shuffle(seed=cfg.get("seed", 0)))
    save_merged(trainer, Path(cfg["output_dir"]) / "merged")


if __name__ == "__main__":
    main()
