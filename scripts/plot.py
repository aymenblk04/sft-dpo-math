"""DPO training curves and results table.

    python scripts/plot.py --runs dpo dpo_beta0.05 dpo_beta0.5 dpo_ipo dpo_random_neg
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURVES = {
    "loss": "DPO loss",
    "rewards/accuracies": "Preference accuracy (chosen > rejected)",
    "rewards/margins": "Reward margin",
    "logps/chosen": "log p(chosen)",
}


def load_history(run_dir: Path) -> list[dict]:
    states = sorted(run_dir.glob("**/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(states[-1].read_text())["log_history"] if states else []


def plot_curves(runs, outputs: Path, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (key, title) in zip(axes.flat, CURVES.items()):
        for run in runs:
            hist = [h for h in load_history(outputs / run) if key in h]
            if hist:
                ax.plot([h["step"] for h in hist], [h[key] for h in hist], label=run)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(out / "dpo_curves.png", dpi=150)


def results_table(runs, results: Path) -> str:
    lines = ["| model | accuracy | format rate | mean chars |", "|---|---|---|---|"]
    for run in runs:
        f = results / run / "metrics.json"
        if f.exists():
            m = json.loads(f.read_text())
            lines.append(f"| {run} | {m['accuracy']:.1%} | {m['format_rate']:.1%} | {m['mean_chars']:.0f} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--outputs", default="outputs")
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    plot_curves(args.runs, Path(args.outputs), out)
    table = results_table(["base", "sft", *args.runs], Path(args.results))
    (out / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
