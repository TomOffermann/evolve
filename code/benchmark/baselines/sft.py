"""SFT baseline using HuggingFace Trainer.

Fine-tunes the model on countdown solutions (or GSM8K solutions). This is the
supervised ceiling — it directly trains on known-good answers, which no RL or
ES method gets to see.

Wall-clock is matched to the ES methods: SFT gets the same total training time.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Simple prompt+completion dataset for SFT."""

    def __init__(self, data: list[dict[str, str]], tokenizer, max_length: int = 128):
        self.examples = []
        for item in data:
            text = item["prompt"] + item["completion"]
            enc = tokenizer(text, truncation=True, max_length=max_length,
                            padding="max_length", return_tensors="pt")
            # Create labels: mask the prompt tokens
            prompt_enc = tokenizer(item["prompt"], truncation=True,
                                   max_length=max_length, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = enc["input_ids"].clone()
            labels[0, :prompt_len] = -100  # don't train on prompt
            self.examples.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": labels.squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def run_sft(model, tokenizer, sft_data: list[dict[str, str]],
            max_steps: int = 500, lr: float = 5e-5,
            batch_size: int = 8, max_length: int = 128,
            device: str = "cuda") -> dict:
    """Run SFT training and return timing + loss info.

    Uses a simple training loop rather than TRL to minimise dependencies.
    """
    dataset = SFTDataset(sft_data, tokenizer, max_length=max_length)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    t0 = time.time()
    step = 0
    losses = []

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1

    wall_seconds = time.time() - t0
    model.eval()

    return {
        "method": "sft",
        "wall_seconds": wall_seconds,
        "steps": step,
        "final_loss": losses[-1] if losses else 0,
        "mean_loss": sum(losses) / len(losses) if losses else 0,
    }
