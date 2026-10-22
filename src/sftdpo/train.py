"""SFT and DPO trainers built from config dicts."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer


def build_peft_config(lora_cfg: dict | None) -> LoraConfig | None:
    if not lora_cfg or not lora_cfg.get("enabled"):
        return None
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def _train_kwargs(cfg: dict) -> dict:
    kwargs = dict(cfg["train"])
    if not torch.cuda.is_available():
        kwargs["bf16"] = False
        kwargs["use_cpu"] = True
    return dict(output_dir=cfg["output_dir"], seed=cfg.get("seed", 42), **kwargs)


def train_sft(cfg: dict, train_dataset, model=None, tokenizer=None) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model if model is not None else cfg["model_name"],
        args=SFTConfig(**_train_kwargs(cfg)),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg.get("lora")),
    )
    trainer.train()
    return trainer


def train_dpo(cfg: dict, train_dataset, model=None, tokenizer=None) -> DPOTrainer:
    trainer = DPOTrainer(
        model=model if model is not None else cfg["model_name"],
        args=DPOConfig(**_train_kwargs(cfg)),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=build_peft_config(cfg.get("lora")),
    )
    trainer.train()
    return trainer


def save_merged(trainer, out_dir: str | Path) -> Path:
    """Save the full model with any LoRA adapter merged in."""
    out = Path(out_dir)
    model = trainer.model
    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()
    model.save_pretrained(out)
    trainer.processing_class.save_pretrained(out)
    return out
