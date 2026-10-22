import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import plot  # noqa: E402


def test_plot_and_table(tmp_path):
    run = tmp_path / "outputs" / "dpo"
    run.mkdir(parents=True)
    hist = [{"step": s, "loss": 0.7 - 0.1 * s, "rewards/margins": 0.2 * s} for s in range(1, 4)]
    (run / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    for name, acc in [("sft", 0.3), ("dpo", 0.35)]:
        (tmp_path / "results" / name).mkdir(parents=True)
        (tmp_path / "results" / name / "metrics.json").write_text(
            json.dumps({"accuracy": acc, "format_rate": 1.0, "mean_chars": 250.0})
        )
    figs = tmp_path / "figs"
    figs.mkdir()
    plot.plot_curves(["dpo"], tmp_path / "outputs", figs)
    assert (figs / "dpo_curves.png").exists()
    table = plot.results_table(["base", "sft", "dpo"], tmp_path / "results")
    assert table.splitlines()[2:] == ["| sft | 30.0% | 100.0% | 250 |", "| dpo | 35.0% | 100.0% | 250 |"]
