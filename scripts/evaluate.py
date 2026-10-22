"""Evaluate any model directory (or hub id) on the GSM8K test split.

    python scripts/evaluate.py --model outputs/dpo/merged --out results/dpo
"""

import argparse
import json

from sftdpo.data import build_eval_dataset, load_raw
from sftdpo.evaluate import evaluate
from sftdpo.generation import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=320)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model, tok = load_model(args.model)
    data = build_eval_dataset(load_raw("test", args.num_samples))
    print(json.dumps(evaluate(model, tok, data, args.max_new_tokens, args.batch_size, args.out), indent=2))


if __name__ == "__main__":
    main()
