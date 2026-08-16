"""GRPO baseline — Group Relative Policy Optimization.

Implements GRPO (DeepSeek-style) for countdown post-training. For each prompt,
generates G completions, scores them with the verifier, and updates the policy
using the relative advantage within the group.

This is a minimal self-contained implementation rather than a TRL dependency,
so the friend can run it without installing the full TRL stack.

Wall-clock is matched to the ES methods: GRPO gets the same total training time.

Fair-budget accounting (from design.md): GRPO does G rollouts per prompt plus a
backward pass (~2x forward). ES does N forward-only rollouts. At G=8 and N=32,
ES uses ~4x the rollouts and no backward. Report FLOPs alongside wall-clock.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F


def generate_completions(model, tokenizer, prompts: list[str],
                         max_new_tokens: int, temperature: float = 0.7,
                         n_samples: int = 1, device: str = "cuda") -> list[str]:
    """Generate completions for a batch of prompts."""
    all_completions = []

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                           truncation=True).to(device)

        for _ in range(n_samples):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            # Decode only the generated part
            gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            all_completions.append(completion)

    return all_completions


def run_grpo(model, tokenizer, task, problems, max_new_tokens: int = 24,
             group_size: int = 8, lr: float = 1e-5, kl_coeff: float = 0.01,
             max_steps: int = 200, batch_size: int = 4,
             device: str = "cuda") -> dict:
    """Run GRPO training.

    For each step:
    1. Sample a batch of problems
    2. Generate G completions per problem
    3. Score with verifier
    4. Compute group-relative advantage
    5. Policy gradient update with KL penalty
    """
    import copy
    import random

    # Freeze a reference model for KL
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    t0 = time.time()
    rng = random.Random(42)
    losses = []
    rewards_history = []

    for step in range(max_steps):
        # Sample problems
        batch_problems = rng.sample(problems, min(batch_size, len(problems)))
        prompts = task.format_prompts(batch_problems)

        # Generate G completions per problem
        all_completions = []
        for prompt in prompts:
            comps = generate_completions(
                model, tokenizer, [prompt], max_new_tokens,
                n_samples=group_size, device=device)
            all_completions.append(comps)

        # Score each completion
        all_rewards = []
        for prob, comps in zip(batch_problems, all_completions):
            rewards = task.score([prob] * len(comps), comps)
            all_rewards.append(rewards)

        # GRPO update: for each problem, compute advantage within the group
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_terms = 0

        for prompt, comps, rewards in zip(prompts, all_completions, all_rewards):
            # Group-relative advantage
            mean_r = rewards.mean()
            std_r = rewards.std().clamp_min(1e-8)
            advantages = (rewards - mean_r) / std_r

            for comp, adv in zip(comps, advantages):
                if abs(adv.item()) < 1e-8:
                    continue

                text = prompt + comp
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=256).to(device)
                prompt_inputs = tokenizer(prompt, return_tensors="pt",
                                          truncation=True).to(device)
                prompt_len = prompt_inputs["input_ids"].shape[1]

                if inputs["input_ids"].shape[1] <= prompt_len:
                    continue

                # Log prob under current policy
                outputs = model(**inputs)
                logits = outputs.logits[:, prompt_len - 1:-1]
                targets = inputs["input_ids"][:, prompt_len:]
                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                seq_log_prob = token_log_probs.sum()

                # KL penalty vs reference
                with torch.no_grad():
                    ref_outputs = ref_model(**inputs)
                    ref_logits = ref_outputs.logits[:, prompt_len - 1:-1]
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                    ref_token_log_probs = ref_log_probs.gather(
                        2, targets.unsqueeze(-1)).squeeze(-1)
                    kl = (token_log_probs.detach() - ref_token_log_probs).sum()

                # GRPO loss: -advantage * log_prob + kl_coeff * KL
                term = -adv.to(device) * seq_log_prob + kl_coeff * kl
                total_loss = total_loss + term
                n_terms += 1

        if n_terms > 0:
            loss = total_loss / n_terms
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())

            mean_reward = torch.stack([r.mean() for r in all_rewards]).mean().item()
            rewards_history.append(mean_reward)

    wall_seconds = time.time() - t0
    model.eval()

    # Clean up reference model
    del ref_model

    return {
        "method": "grpo",
        "wall_seconds": wall_seconds,
        "steps": max_steps,
        "final_loss": losses[-1] if losses else 0,
        "mean_reward": sum(rewards_history) / len(rewards_history) if rewards_history else 0,
    }
