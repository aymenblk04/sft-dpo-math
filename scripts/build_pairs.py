"""Step 2: sample K answers per question from the SFT model, verify them and
write (chosen, rejected) pairs for DPO.

    python scripts/build_pairs.py --config configs/sampling.yaml
"""

import argparse
import json
from pathlib import Path

from sftdpo.config import load_config
from sftdpo.data import build_eval_dataset, load_raw
from sftdpo.generation import generate, load_model
from sftdpo.pairs import build_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)

    # questions from the train split only: the test split stays unseen
    data = build_eval_dataset(load_raw("train", cfg["num_questions"], cfg["seed"]))
    model, tok = load_model(cfg["model_name"])
    samples = generate(
        model,
        tok,
        data["prompt"],
        max_new_tokens=cfg["max_new_tokens"],
        batch_size=cfg["batch_size"],
        num_return_sequences=cfg["num_samples"],
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
    )
    rows, stats = build_pairs(
        data["prompt"],
        samples,
        data["answer"],
        max_pairs_per_prompt=cfg["max_pairs_per_prompt"],
        strategy=cfg["strategy"],
        seed=cfg["seed"],
    )

    out = Path(cfg["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    out.with_suffix(".stats.json").write_text(json.dumps(stats.as_dict(), indent=2))
    print(json.dumps(stats.as_dict(), indent=2))


if __name__ == "__main__":
    main()
