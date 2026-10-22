"""Batched generation with plain transformers (no vLLM dependency)."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_dtype() -> torch.dtype:
    """bf16 on Ampere+, fp16 on older GPUs such as the Colab T4, fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(model_path: str, adapter_path: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_path)
    tokenizer.padding_side = "left"  # decoder-only batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = pick_dtype()
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path).merge_and_unload()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), tokenizer


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompts: list[list[dict]],
    max_new_tokens: int = 256,
    batch_size: int = 16,
    num_return_sequences: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> list[list[str]]:
    """Return ``num_return_sequences`` completions for every prompt.

    temperature 0 means greedy decoding.
    """
    sample = temperature > 0
    results: list[list[str]] = []
    for i in range(0, len(prompts), batch_size):
        texts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts[i : i + batch_size]
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(model.device)
        sampling = dict(temperature=temperature, top_p=top_p) if sample else {}
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=sample,
            **sampling,
        )
        decoded = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for j in range(len(texts)):
            results.append(decoded[j * num_return_sequences : (j + 1) * num_return_sequences])
    return results
