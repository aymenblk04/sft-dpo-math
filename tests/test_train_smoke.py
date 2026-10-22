"""SFT then DPO on a tiny random model, end to end on CPU."""

import pytest
from datasets import Dataset

from sftdpo.config import load_config
from sftdpo.data import build_sft_dataset, make_prompt
from sftdpo.train import save_merged, train_dpo, train_sft

SMALL = [
    "train.per_device_train_batch_size=2",
    "train.gradient_accumulation_steps=1",
    # the tiny tokenizer splits the system prompt into many tokens, so a short
    # max_length would truncate away the whole completion
    "train.max_length=512",
    "train.logging_steps=1",
    "train.gradient_checkpointing=false",
]


def overrides(tmp_path, lora):
    return SMALL + [f"output_dir={tmp_path}", f"lora.enabled={str(lora).lower()}", "train.max_steps=2"]


@pytest.mark.parametrize("lora", [False, True])
def test_sft_then_dpo(lora, tiny_model_path, tmp_path):
    raw = Dataset.from_dict({"question": ["2+2?"] * 4, "answer": ["2+2=4\n#### 4"] * 4})
    sft_cfg = load_config("configs/sft.yaml", overrides(tmp_path / "sft", lora))
    sft = train_sft(sft_cfg, build_sft_dataset(raw), model=tiny_model_path)
    assert sft.state.global_step == 2
    merged = save_merged(sft, tmp_path / "sft" / "merged")
    assert (merged / "config.json").exists()

    pairs = Dataset.from_list(
        [
            {
                "prompt": make_prompt("2+2?"),
                "chosen": [{"role": "assistant", "content": "<think>2+2=4</think><answer>4</answer>"}],
                "rejected": [{"role": "assistant", "content": "<think>2+2=5</think><answer>5</answer>"}],
            }
        ]
        * 4
    )
    dpo_cfg = load_config("configs/dpo.yaml", overrides(tmp_path / "dpo", lora))
    dpo = train_dpo(dpo_cfg, pairs, model=str(merged))
    logs = [h for h in dpo.state.log_history if "rewards/margins" in h]
    assert logs, "DPO should log reward margins"
