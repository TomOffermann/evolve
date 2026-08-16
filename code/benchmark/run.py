"""GPU benchmark — ES vs GRPO/SFT on a real pretrained model.

Single entry point. Clone-and-run:
    pip install -r code/benchmark/requirements.txt
    python code/benchmark/run.py --model HuggingFaceTB/SmolLM2-135M --seeds 0,1,2

Methods compared:
    sft                  supervised fine-tuning on countdown solutions
    grpo                 Group Relative Policy Optimization (gradient-based RL)
    eggroll_vanilla      EGGROLL ES: global IID perturbation + ResolutionRule
    eggroll_partitioned  EGGROLL + partitioned perturbation (E8-validated)
    eggroll_selective    EGGROLL + partitioned + utility-gated update (DiPEC-inspired)

Metrics:
    pass@1, pass@4, pass@16   diversity axis
    perplexity delta          forgetting probe (lower = better preserved)
    wall-clock time           practical comparison
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

# Ensure benchmark/ and code/ are importable
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))           # for config, eggroll, tasks, baselines
sys.path.insert(0, str(_here.parent))    # for evolve package

import torch

from config import BenchmarkConfig
from eggroll.hooks import HookManager
from eggroll.trainer import ESConfig, ESTrainer, PartitionedESTrainer, SelectiveESTrainer
from tasks.countdown_lm import CountdownLMTask
from tasks.forgetting import ForgettingTracker
from results.logger import RunResult


def load_model(config: BenchmarkConfig):
    """Load model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(config.dtype, torch.float32)

    print(f"Loading {config.model_name} ({config.dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        device_map=config.device if torch.cuda.is_available() else "cpu",
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.1f}M parameters, device={next(model.parameters()).device}")
    return model, tokenizer


def make_evaluate_fn(task: CountdownLMTask, config: BenchmarkConfig):
    """Create the evaluation function for ES trainers.

    Returns a callable(model, tokenizer, gen_seed, greedy) -> (fitness, per_example).
    """
    def evaluate_fn(model, tokenizer, gen_seed, greedy=True):
        problems = task.train_batch(gen_seed, config.n_problems_train)
        prompts = task.format_prompts(problems)

        # Generate completions
        completions = []
        device = next(model.parameters()).device
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=not greedy,
                    temperature=1.0 if greedy else config.temperature,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            completions.append(completion)

        rewards = task.score(problems, completions)
        return float(rewards.mean()), rewards

    return evaluate_fn


def make_parent_eval_fn(task: CountdownLMTask, config: BenchmarkConfig):
    """Create parent fitness function (unperturbed model, same CRN seed)."""
    def parent_eval_fn(model, tokenizer, gen_seed):
        problems = task.train_batch(gen_seed, config.n_problems_train)
        prompts = task.format_prompts(problems)
        device = next(model.parameters()).device

        completions = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            completions.append(completion)

        rewards = task.score(problems, completions)
        return float(rewards.mean())

    return parent_eval_fn


def evaluate_pass_at_k(model, tokenizer, task: CountdownLMTask,
                       config: BenchmarkConfig) -> dict:
    """Evaluate pass@k on held-out problems."""
    problems = task.eval_batch()
    prompts = task.format_prompts(problems)
    device = next(model.parameters()).device

    results = {}
    for k in config.eval_ks:
        solved = 0
        for prompt, prob in zip(prompts, problems):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            any_correct = False
            for _ in range(k):
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=config.max_new_tokens,
                        do_sample=(k > 1),
                        temperature=config.temperature if k > 1 else 1.0,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )
                gen_tokens = outputs[0, inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                if task.score([prob], [completion]).item() > 0:
                    any_correct = True
                    break
            if any_correct:
                solved += 1
        results[f"pass@{k}"] = solved / len(problems)
    return results


# ---------------------------------------------------------------- method runners

def run_es_method(method: str, model, tokenizer, task, config, seed):
    """Run an ES method (vanilla, partitioned, or selective)."""
    from evolve.operators.sigma import ResolutionRule

    # Deep copy model so each method starts from the same base
    model_copy = copy.deepcopy(model)
    device = next(model_copy.parameters()).device

    hook_mgr = HookManager(model_copy, sigma=config.sigma, rank=config.rank)
    print(f"  {hook_mgr.n_layers} perturbable layers, "
          f"{hook_mgr.n_params / 1e6:.1f}M perturbable params")

    evaluate_fn = make_evaluate_fn(task, config)
    parent_eval_fn = make_parent_eval_fn(task, config)
    sigma_rule = ResolutionRule(config.sigma)

    es_config = ESConfig(
        n_pop=config.n_pop, rank=config.rank, sigma=config.sigma,
        alpha=config.alpha, generations=config.generations, seed=seed,
    )

    if method == "eggroll_partitioned":
        partitions = hook_mgr.auto_partitions(config.partition_groups)
        trainer = PartitionedESTrainer(
            model_copy, tokenizer, hook_mgr, evaluate_fn, es_config,
            sigma_rule=sigma_rule, parent_eval_fn=parent_eval_fn,
            partitions=partitions, n_active=config.n_active,
        )
    elif method == "eggroll_selective":
        partitions = hook_mgr.auto_partitions(config.partition_groups)
        trainer = SelectiveESTrainer(
            model_copy, tokenizer, hook_mgr, evaluate_fn, es_config,
            sigma_rule=sigma_rule, parent_eval_fn=parent_eval_fn,
            partitions=partitions, n_active=config.n_active,
        )
    else:
        trainer = ESTrainer(
            model_copy, tokenizer, hook_mgr, evaluate_fn, es_config,
            sigma_rule=sigma_rule, parent_eval_fn=parent_eval_fn,
        )

    # Forgetting tracker
    forgetting = ForgettingTracker(model_copy, tokenizer)
    forgetting.measure_baseline()

    # Run with checkpoint evaluation
    checkpoints = []

    def on_gen(rec):
        gen = rec.generation
        if gen % config.eval_every == 0 or gen == config.generations - 1:
            hook_mgr.deactivate()
            ev = evaluate_pass_at_k(model_copy, tokenizer, task, config)
            cp = {"gen": gen, "sigma": rec.sigma, "fitness_mean": rec.fitness_mean}
            cp.update(ev)
            checkpoints.append(cp)
            print(f"    gen {gen:4d}  F={rec.fitness_mean:.4f}  "
                  f"pass@1={ev.get('pass@1', 0):.4f}  "
                  f"sigma={rec.sigma:.5f}  {rec.seconds:.1f}s")

    t0 = time.time()
    trainer.run(callback=on_gen)
    wall_seconds = time.time() - t0

    # Final evaluation
    hook_mgr.deactivate()
    final = evaluate_pass_at_k(model_copy, tokenizer, task, config)
    forgetting.measure_current(config.generations)

    hook_mgr.remove()

    return RunResult(
        method=method,
        model=config.model_name,
        task=config.task,
        seed=seed,
        config={"n_pop": config.n_pop, "rank": config.rank,
                "sigma": config.sigma, "alpha": config.alpha,
                "generations": config.generations},
        checkpoints=checkpoints,
        final=final,
        forgetting=forgetting.report(),
        history=[{"gen": r.generation, "fitness_mean": r.fitness_mean,
                  "sigma": r.sigma} for r in trainer.history],
        wall_seconds=wall_seconds,
    )


def run_sft_method(model, tokenizer, task, config, seed):
    """Run SFT baseline."""
    from baselines.sft import run_sft

    model_copy = copy.deepcopy(model)

    forgetting = ForgettingTracker(model_copy, tokenizer)
    forgetting.measure_baseline()

    sft_data = task.sft_data(1000, seed=seed)
    device = next(model_copy.parameters()).device
    info = run_sft(model_copy, tokenizer, sft_data, max_steps=500,
                   device=str(device))

    final = evaluate_pass_at_k(model_copy, tokenizer, task, config)
    forgetting.measure_current(0)

    return RunResult(
        method="sft",
        model=config.model_name,
        task=config.task,
        seed=seed,
        config={"lr": 5e-5, "steps": 500},
        final=final,
        forgetting=forgetting.report(),
        wall_seconds=info["wall_seconds"],
        method_info=info,
    )


def run_grpo_method(model, tokenizer, task, config, seed):
    """Run GRPO baseline."""
    from baselines.grpo import run_grpo

    model_copy = copy.deepcopy(model)

    forgetting = ForgettingTracker(model_copy, tokenizer)
    forgetting.measure_baseline()

    problems = task.train_problems
    device = next(model_copy.parameters()).device
    info = run_grpo(model_copy, tokenizer, task, problems,
                    max_new_tokens=config.max_new_tokens,
                    max_steps=config.generations,
                    device=str(device))

    final = evaluate_pass_at_k(model_copy, tokenizer, task, config)
    forgetting.measure_current(0)

    return RunResult(
        method="grpo",
        model=config.model_name,
        task=config.task,
        seed=seed,
        config={"group_size": 8, "lr": 1e-5, "kl_coeff": 0.01,
                "steps": config.generations},
        final=final,
        forgetting=forgetting.report(),
        wall_seconds=info["wall_seconds"],
        method_info=info,
    )


# -------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="GPU benchmark: ES vs GRPO/SFT on real models")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--task", default="countdown")
    parser.add_argument("--methods", default=None,
                        help="Comma-separated methods, or 'all'")
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma-separated seeds")
    parser.add_argument("--generations", type=int, default=200)
    parser.add_argument("--n-pop", type=int, default=32)
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--output", default="code/benchmark/results")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: 30 gens, 1 seed, fewer problems")
    args = parser.parse_args()

    config = BenchmarkConfig(
        model_name=args.model,
        task=args.task,
        dtype=args.dtype,
        n_pop=args.n_pop,
        sigma=args.sigma,
        generations=args.generations,
        output_dir=args.output,
    )

    if args.quick:
        config.generations = 30
        config.n_problems_train = 16
        config.n_problems_eval = 64
        config.eval_every = 10

    seeds = [int(s) for s in args.seeds.split(",")]
    config.seeds = seeds

    if args.methods:
        methods = args.methods.split(",") if args.methods != "all" else config.methods
    else:
        methods = config.methods

    config.methods = methods

    print(f"GPU Benchmark")
    print(f"  Model: {config.model_name}")
    print(f"  Task: {config.task}")
    print(f"  Methods: {methods}")
    print(f"  Seeds: {seeds}")
    print(f"  N={config.n_pop}, σ={config.sigma}, α={config.alpha}, "
          f"gens={config.generations}")
    print()

    # Add benchmark dir to path for imports
    sys.path.insert(0, str(Path(__file__).parent))

    # Load model once
    model, tokenizer = load_model(config)
    task = CountdownLMTask(
        n_train=config.n_problems_train,
        n_eval=config.n_problems_eval,
    )

    # Baseline evaluation
    print("Baseline evaluation...")
    baseline = evaluate_pass_at_k(model, tokenizer, task, config)
    print(f"  Base model: {baseline}")
    print()

    # Run each method × seed
    for method in methods:
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Running: {method}, seed={seed}")
            print(f"{'='*60}")

            try:
                if method == "sft":
                    result = run_sft_method(model, tokenizer, task, config, seed)
                elif method == "grpo":
                    result = run_grpo_method(model, tokenizer, task, config, seed)
                elif method.startswith("eggroll"):
                    result = run_es_method(method, model, tokenizer, task,
                                           config, seed)
                else:
                    print(f"  Unknown method: {method}, skipping")
                    continue

                fname = result.save(config.output_dir)
                print(f"\n  Result: pass@1={result.final.get('pass@1', 0):.4f}  "
                      f"pass@16={result.final.get('pass@16', 0):.4f}  "
                      f"ppl_Δ={result.forgetting.get('perplexity_delta', 0):+.1f}  "
                      f"wall={result.wall_seconds:.0f}s")
                print(f"  Saved: {fname}")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback
                traceback.print_exc()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    from results.summary import print_comparison
    print_comparison(config.output_dir)


if __name__ == "__main__":
    main()
