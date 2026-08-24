"""ES-based router optimization for real MoE models (OLMoE, Phi-MoE).

Loads a pretrained MoE model, freezes everything except gate/router weights,
and optimizes routing with ES vs gradient. Tests whether ES's direct hard-
routing optimization beats gradient's soft-routing proxy.

Setup (Colab / any GPU):
    pip install torch transformers datasets accelerate bitsandbytes

Usage:
    # Quick smoke test (~5 min on T4)
    python run_olmoe.py --quick

    # Full experiment (~1 hour on T4)
    python run_olmoe.py --model allenai/OLMoE-1B-7B-0924 --generations 100

    # Smaller model for tighter GPUs
    python run_olmoe.py --model microsoft/Phi-tiny-MoE-instruct --dtype float16

Arms:
    pretrained   -- original router, no training (upper bound if routing is good)
    random       -- random router (lower bound)
    gradient     -- Adam on cross-entropy with soft routing, STE for hard layers
    es_global    -- EGGROLL low-rank ES on all gate weights jointly
    es_partitioned -- each member perturbs routing to one expert (per-expert credit)
"""

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# EGGROLL seed utilities
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evolve.core.noise import chunk_seed, member_seed


# ========================================================================
# Router discovery — works for OLMoE, Phi-MoE, Mixtral, DeepSeek, Qwen-MoE
# ========================================================================

def _is_linear(module) -> bool:
    """Check if module is a linear layer (including quantized variants)."""
    if isinstance(module, nn.Linear):
        return True
    # bitsandbytes quantized linear layers
    typ = type(module).__name__
    if "Linear4bit" in typ or "Linear8bit" in typ:
        return True
    return hasattr(module, "weight") and hasattr(module, "in_features")


def find_gate_modules(model) -> dict[str, nn.Module]:
    """Auto-discover MoE gate/router linear layers."""
    # Common gate/router names across MoE architectures
    gate_names = {"gate", "gate_proj", "router", "w_gate"}
    gates = {}
    for name, module in model.named_modules():
        short = name.split(".")[-1]
        if short in gate_names and _is_linear(module):
            gates[name] = module
    if not gates:
        # Debug: print all module names to help identify the gate
        print("\nDEBUG: No gate layers found. All module names:")
        for name, module in model.named_modules():
            if _is_linear(module):
                short = name.split(".")[-1]
                print(f"  {name}  ({type(module).__name__}, "
                      f"out={getattr(module, 'out_features', '?')}, "
                      f"in={getattr(module, 'in_features', '?')})")
        raise RuntimeError(
            "No gate layers found. See module list above and update "
            "gate_names in find_gate_modules().")
    return gates


def get_gate_info(gates: dict[str, nn.Linear]) -> list[dict]:
    """Extract shapes and metadata from discovered gates."""
    info = []
    for name, module in gates.items():
        info.append({
            "name": name,
            "out_features": module.out_features,   # n_experts
            "in_features": module.in_features,      # hidden_dim
            "n_params": module.weight.numel(),
        })
    return info


def freeze_except_gates(model, gate_names: list[str]):
    """Freeze all parameters except named gate layers."""
    for param in model.parameters():
        param.requires_grad = False
    for name in gate_names:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        for param in module.parameters():
            param.requires_grad = True


def get_gate_param(model, gate_name: str) -> nn.Parameter:
    module = model
    for part in gate_name.split("."):
        module = getattr(module, part)
    return module.weight


# ========================================================================
# Factor generation (deterministic, per-gate, per-member)
# ========================================================================

def draw_gate_factors(base_seed, member_idx, gate_idx,
                      out_feat, in_feat, rank, dtype=torch.float32):
    """Generate (A, B) for one member at one gate. CPU only, move later."""
    seed = chunk_seed(member_seed(base_seed, member_idx), gate_idx)
    g = torch.Generator(device="cpu").manual_seed(seed)
    total = out_feat * rank + in_feat * rank
    flat = torch.randn(total, generator=g, dtype=dtype)
    A = flat[:out_feat * rank].reshape(out_feat, rank)
    B = flat[out_feat * rank:].reshape(in_feat, rank)
    return A, B


def rank_normalise(F_vec: torch.Tensor) -> torch.Tensor:
    """Centred rank transform with tied ranks averaged."""
    N = F_vec.numel()
    order = F_vec.argsort()
    ranks = torch.empty_like(F_vec)
    ranks[order] = torch.arange(N, dtype=F_vec.dtype, device=F_vec.device)
    sortedF = F_vec[order]
    uniq, inv, counts = torch.unique(sortedF, return_inverse=True, return_counts=True)
    sums = torch.zeros(uniq.numel(), dtype=F_vec.dtype, device=F_vec.device)
    sums.index_add_(0, inv, torch.arange(N, dtype=F_vec.dtype, device=F_vec.device))
    ranks[order] = (sums / counts)[inv]
    return ranks / max(N - 1, 1) - 0.5


# ========================================================================
# Evaluation tasks
# ========================================================================

def load_eval_data(task: str, n_train: int, n_eval: int, seed: int = 42):
    """Load evaluation data. Returns (train_texts, eval_texts).

    For router optimization, we use text chunks as fitness signal
    (lower perplexity = better routing for this data distribution).
    """
    from datasets import load_dataset

    if task == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        rng = random.Random(seed)
        rng.shuffle(texts)
        return texts[:n_train], texts[n_train:n_train + n_eval]

    elif task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        items = list(ds)
        rng = random.Random(seed)
        rng.shuffle(items)
        train_texts = [f"Q: {it['question']}\nA: {it['answer']}"
                       for it in items[:n_train]]
        eval_texts = [f"Q: {it['question']}\nA: {it['answer']}"
                      for it in items[n_train:n_train + n_eval]]
        return train_texts, eval_texts

    elif task == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="train")
        items = list(ds)
        rng = random.Random(seed)
        rng.shuffle(items)
        def format_arc(item):
            q = item["question"]
            choices = item["choices"]
            labels = choices["label"]
            texts = choices["text"]
            opts = "\n".join(f"  {l}. {t}" for l, t in zip(labels, texts))
            ans_key = item["answerKey"]
            ans_idx = labels.index(ans_key) if ans_key in labels else 0
            return f"Question: {q}\n{opts}\nAnswer: {labels[ans_idx]}. {texts[ans_idx]}"
        train_texts = [format_arc(it) for it in items[:n_train]]
        eval_texts = [format_arc(it) for it in items[n_train:n_train + n_eval]]
        return train_texts, eval_texts

    else:
        raise ValueError(f"Unknown task: {task}")


def compute_perplexity(model, tokenizer, texts, device, max_length=256):
    """Mean perplexity over a list of texts."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].shape[1] - 1
            if n > 0:
                total_loss += outputs.loss.item() * n
                total_tokens += n
    return math.exp(total_loss / max(total_tokens, 1))


def compute_routing_metrics(model, gate_names, tokenizer, texts, device,
                            max_length=256):
    """Expert utilization, load balance, collapse detection."""
    # Collect routing decisions via hooks
    routing_counts = defaultdict(lambda: torch.zeros(1))
    hooks_registered = []

    def make_hook(gate_name):
        def hook_fn(module, args, output):
            # output is the gate logits: (batch, seq_len, n_experts) or (batch, n_experts)
            with torch.no_grad():
                if output.dim() == 3:
                    chosen = output.argmax(dim=-1).reshape(-1)
                else:
                    chosen = output.argmax(dim=-1).reshape(-1)
                n_exp = output.shape[-1]
                counts = torch.bincount(chosen.cpu(), minlength=n_exp).float()
                routing_counts[gate_name] = routing_counts.get(
                    gate_name, torch.zeros(n_exp)) + counts
            return output
        return hook_fn

    for gate_name in gate_names:
        module = model
        for part in gate_name.split("."):
            module = getattr(module, part)
        h = module.register_forward_hook(make_hook(gate_name))
        hooks_registered.append(h)

    # Run forward passes
    model.eval()
    with torch.no_grad():
        for text in texts[:50]:  # limit for speed
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            model(**inputs)

    for h in hooks_registered:
        h.remove()

    # Aggregate
    total_counts = None
    for name, counts in routing_counts.items():
        if total_counts is None:
            total_counts = counts.clone()
        else:
            if total_counts.shape == counts.shape:
                total_counts += counts
            # else skip mismatched shapes

    if total_counts is None:
        return {"balance": 0, "collapsed": 0, "n_experts": 0}

    p = total_counts / total_counts.sum().clamp_min(1)
    p = p.clamp_min(1e-8)
    entropy = float(-(p * p.log()).sum())
    max_entropy = math.log(len(p))
    balance = entropy / max_entropy if max_entropy > 0 else 0

    return {
        "balance": round(balance, 4),
        "collapsed": int((p < 0.001).sum()),
        "n_experts": len(p),
        "utilization": [round(x, 4) for x in p.tolist()[:16]],  # first 16
        "entropy": round(entropy, 4),
    }


# ========================================================================
# ES Router Optimization
# ========================================================================

def es_optimize_routing(model, tokenizer, gate_info, train_texts, device,
                        n_pop=32, rank=1, sigma=0.01, alpha=0.005,
                        generations=50, seed=0, partitioned=False,
                        max_length=256):
    """EGGROLL ES on gate weights only.

    Returns (model, history).
    """
    _sigma = sigma
    history = []

    for gen in range(generations):
        t0 = time.time()
        gen_seed = chunk_seed(seed * 1_000_003, gen)

        # Select CRN batch
        rng = random.Random(gen_seed)
        batch = rng.sample(train_texts, min(len(train_texts), 20))

        # ---- pass 1: evaluate each member -----------------------------------
        all_fitness = []
        # Save original gate weights
        orig_weights = {}
        for gi, ginfo in enumerate(gate_info):
            orig_weights[gi] = get_gate_param(model, ginfo["name"]).data.clone()

        for member_idx in range(n_pop):
            # Apply perturbation to gate weights
            for gi, ginfo in enumerate(gate_info):
                param = get_gate_param(model, ginfo["name"])

                if partitioned:
                    # Each member perturbs one expert row in one gate
                    # Deterministic assignment: which gate and which expert
                    assign_g = torch.Generator(device="cpu").manual_seed(
                        (gen_seed ^ 0x5EED) + 0x9E37 * member_idx)
                    n_experts = ginfo["out_features"]
                    n_gates = len(gate_info)
                    # Pick one gate
                    chosen_gate = int(torch.randint(0, n_gates, (1,),
                                                    generator=assign_g).item())
                    if gi != chosen_gate:
                        continue
                    # Pick one expert row
                    chosen_expert = int(torch.randint(0, n_experts, (1,),
                                                      generator=assign_g).item())
                    # Perturb only that row
                    A, B = draw_gate_factors(
                        gen_seed, member_idx, gi,
                        1, ginfo["in_features"], rank)
                    scale = sigma / (rank ** 0.5)
                    delta = (A @ B.T) * scale  # (1, in_features)
                    with torch.no_grad():
                        param.data[chosen_expert:chosen_expert+1] += delta.to(param)
                else:
                    # Global: perturb all weights
                    A, B = draw_gate_factors(
                        gen_seed, member_idx, gi,
                        ginfo["out_features"], ginfo["in_features"], rank)
                    scale = sigma / (rank ** 0.5)
                    delta = (A @ B.T) * scale
                    with torch.no_grad():
                        param.data += delta.to(param)

            # Evaluate: negative perplexity as fitness (higher = better)
            total_loss = 0.0
            total_tokens = 0
            model.eval()
            with torch.no_grad():
                for text in batch:
                    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                       max_length=max_length).to(device)
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    n = inputs["input_ids"].shape[1] - 1
                    if n > 0:
                        total_loss += outputs.loss.item() * n
                        total_tokens += n
            fitness = -total_loss / max(total_tokens, 1)  # negative loss
            all_fitness.append(fitness)

            # Restore original weights
            for gi, ginfo in enumerate(gate_info):
                param = get_gate_param(model, ginfo["name"])
                param.data.copy_(orig_weights[gi])

        fitness_tensor = torch.tensor(all_fitness, dtype=torch.float32)

        # ---- pass 2: accumulate update --------------------------------------
        weights = rank_normalise(fitness_tensor)

        n_parts = len(gate_info)
        importance = n_parts if partitioned else 1.0
        update_scale = alpha / (n_pop * _sigma) * importance
        r_scale = rank ** -0.5

        for gi, ginfo in enumerate(gate_info):
            param = get_gate_param(model, ginfo["name"])
            out_f, in_f = ginfo["out_features"], ginfo["in_features"]

            if partitioned:
                # For each member, check if they perturbed this gate
                for member_idx in range(n_pop):
                    assign_g = torch.Generator(device="cpu").manual_seed(
                        (gen_seed ^ 0x5EED) + 0x9E37 * member_idx)
                    chosen_gate = int(torch.randint(0, n_parts, (1,),
                                                     generator=assign_g).item())
                    if gi != chosen_gate:
                        continue
                    chosen_expert = int(torch.randint(0, out_f, (1,),
                                                      generator=assign_g).item())
                    A, B = draw_gate_factors(gen_seed, member_idx, gi,
                                            1, in_f, rank)
                    dW = (A @ B.T) * r_scale * weights[member_idx].item()
                    with torch.no_grad():
                        param.data[chosen_expert:chosen_expert+1] += (
                            update_scale * dW.to(param))
            else:
                # Global: accumulate all members
                A_all = torch.empty(n_pop, out_f, rank)
                B_all = torch.empty(n_pop, in_f, rank)
                for mi in range(n_pop):
                    A_all[mi], B_all[mi] = draw_gate_factors(
                        gen_seed, mi, gi, out_f, in_f, rank)
                w = weights.to(A_all.dtype)
                dW = torch.einsum("i,imr,inr->mn", w, A_all, B_all) * r_scale
                with torch.no_grad():
                    param.data += (update_scale * dW.to(param))

        elapsed = time.time() - t0
        rec = {
            "gen": gen,
            "fitness_mean": float(fitness_tensor.mean()),
            "fitness_max": float(fitness_tensor.max()),
            "fitness_std": float(fitness_tensor.std()),
            "sigma": _sigma,
            "seconds": elapsed,
        }
        history.append(rec)

        if gen % 10 == 0 or gen == generations - 1:
            print(f"  gen {gen:4d}  fitness={rec['fitness_mean']:.4f} "
                  f"(max={rec['fitness_max']:.4f})  {elapsed:.1f}s")

    return model, history


# ========================================================================
# Gradient Router Optimization
# ========================================================================

def gradient_optimize_routing(model, tokenizer, gate_names, train_texts, device,
                              lr=1e-3, steps=200, max_length=256):
    """Adam on cross-entropy with standard forward pass (soft routing via model)."""
    freeze_except_gates(model, gate_names)

    # Collect trainable params
    trainable = []
    for name in gate_names:
        module = model
        for part in name.split("."):
            module = getattr(module, part)
        trainable.extend(module.parameters())

    optimizer = torch.optim.Adam(trainable, lr=lr)
    history = []

    t0 = time.time()
    for step in range(steps):
        # Random batch
        rng = random.Random(step)
        batch = rng.sample(train_texts, min(len(train_texts), 8))

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        n_texts = 0
        for text in batch:
            inputs = tokenizer(text, return_tensors="pt", truncation=True,
                               max_length=max_length).to(device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss = total_loss + outputs.loss
            n_texts += 1

        loss = total_loss / max(n_texts, 1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0 or step == steps - 1:
            print(f"  step {step:4d}  loss={loss.item():.4f}  "
                  f"{time.time() - t0:.1f}s")

        history.append({"step": step, "loss": loss.item()})

    wall_seconds = time.time() - t0
    return model, history, wall_seconds


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ES-based router optimization for MoE models")
    parser.add_argument("--model", default="allenai/OLMoE-1B-7B-0924",
                        help="HuggingFace model ID")
    parser.add_argument("--task", default="arc",
                        choices=["wikitext", "gsm8k", "arc"])
    parser.add_argument("--dtype", default="auto",
                        help="float16, bfloat16, float32, or auto")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit (requires bitsandbytes)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit (requires bitsandbytes)")
    parser.add_argument("--n-pop", type=int, default=32)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test: 10 gens, small eval")
    parser.add_argument("--arms", default="all",
                        help="Comma-separated: pretrained,random,gradient,"
                             "es_global,es_partitioned, or 'all'")
    parser.add_argument("--output", default=None,
                        help="JSON output file")
    args = parser.parse_args()

    if args.quick:
        args.generations = 10
        args.n_pop = 16
        n_train, n_eval = 50, 20
    else:
        n_train, n_eval = 200, 100

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load model ----
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {}
    if args.dtype == "auto":
        load_kwargs["torch_dtype"] = "auto"
    elif args.dtype == "float16":
        load_kwargs["torch_dtype"] = torch.float16
    elif args.dtype == "bfloat16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:
        load_kwargs["torch_dtype"] = torch.float32

    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        load_kwargs["device_map"] = {"": 0}  # force all on GPU 0
    elif args.load_in_8bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        load_kwargs["device_map"] = {"": 0}
    elif device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.0f}M total params")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    # ---- Discover gates ----
    gates = find_gate_modules(model)
    gate_info = get_gate_info(gates)
    gate_names = list(gates.keys())
    total_gate_params = sum(g["n_params"] for g in gate_info)

    print(f"\n  Found {len(gates)} gate layers, {total_gate_params:,} gate params "
          f"({100 * total_gate_params / n_params:.2f}% of model)")
    for g in gate_info[:3]:
        print(f"    {g['name']}: ({g['out_features']}, {g['in_features']})")
    if len(gate_info) > 3:
        print(f"    ... and {len(gate_info) - 3} more")

    # ---- Load data ----
    print(f"\nLoading {args.task} data (train={n_train}, eval={n_eval})...")
    train_texts, eval_texts = load_eval_data(args.task, n_train, n_eval,
                                              seed=args.seed)
    print(f"  Loaded {len(train_texts)} train, {len(eval_texts)} eval texts")

    # ---- Select arms ----
    if args.arms == "all":
        arms = ["pretrained", "random", "gradient", "es_global", "es_partitioned"]
    else:
        arms = args.arms.split(",")

    results = {}

    # ---- Baseline: pretrained router ----
    if "pretrained" in arms:
        print(f"\n{'='*60}")
        print("Arm: pretrained (original router, no training)")
        print(f"{'='*60}")
        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        routing = compute_routing_metrics(model, gate_names, tokenizer,
                                          eval_texts, device)
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Load balance: {routing['balance']:.4f}  "
              f"Collapsed: {routing['collapsed']}/{routing['n_experts']}")
        results["pretrained"] = {
            "perplexity": ppl, "routing": routing, "wall_seconds": 0}

    # Save base model state for resetting between arms
    base_state = {}
    for gi, ginfo in enumerate(gate_info):
        base_state[gi] = get_gate_param(model, ginfo["name"]).data.clone()

    def reset_gates():
        for gi, ginfo in enumerate(gate_info):
            get_gate_param(model, ginfo["name"]).data.copy_(base_state[gi])

    # ---- Arm: random router ----
    if "random" in arms:
        print(f"\n{'='*60}")
        print("Arm: random (randomly initialized router)")
        print(f"{'='*60}")
        torch.manual_seed(args.seed + 7)
        for ginfo in gate_info:
            param = get_gate_param(model, ginfo["name"])
            param.data.normal_(0, 0.01)

        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        routing = compute_routing_metrics(model, gate_names, tokenizer,
                                          eval_texts, device)
        print(f"  Perplexity: {ppl:.2f}")
        print(f"  Load balance: {routing['balance']:.4f}  "
              f"Collapsed: {routing['collapsed']}/{routing['n_experts']}")
        results["random"] = {
            "perplexity": ppl, "routing": routing, "wall_seconds": 0}
        reset_gates()

    # ---- Arm: gradient router ----
    if "gradient" in arms:
        print(f"\n{'='*60}")
        print("Arm: gradient (Adam on cross-entropy, standard forward)")
        print(f"{'='*60}")
        model_grad = copy.deepcopy(model)
        n_steps = args.generations * 2  # roughly match wall-clock
        model_grad, grad_hist, grad_time = gradient_optimize_routing(
            model_grad, tokenizer, gate_names, train_texts, device,
            lr=1e-3, steps=n_steps)

        ppl = compute_perplexity(model_grad, tokenizer, eval_texts, device)
        routing = compute_routing_metrics(model_grad, gate_names, tokenizer,
                                          eval_texts, device)
        print(f"  Perplexity: {ppl:.2f}  ({grad_time:.0f}s)")
        print(f"  Load balance: {routing['balance']:.4f}  "
              f"Collapsed: {routing['collapsed']}/{routing['n_experts']}")
        results["gradient"] = {
            "perplexity": ppl, "routing": routing,
            "wall_seconds": grad_time, "history": grad_hist}
        del model_grad
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Arm: ES global ----
    if "es_global" in arms:
        print(f"\n{'='*60}")
        print("Arm: ES global (EGGROLL on all gate weights)")
        print(f"{'='*60}")
        reset_gates()
        t0 = time.time()
        model, es_hist = es_optimize_routing(
            model, tokenizer, gate_info, train_texts, device,
            n_pop=args.n_pop, rank=args.rank, sigma=args.sigma,
            alpha=args.alpha, generations=args.generations,
            seed=args.seed, partitioned=False)
        es_time = time.time() - t0

        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        routing = compute_routing_metrics(model, gate_names, tokenizer,
                                          eval_texts, device)
        print(f"  Perplexity: {ppl:.2f}  ({es_time:.0f}s)")
        print(f"  Load balance: {routing['balance']:.4f}  "
              f"Collapsed: {routing['collapsed']}/{routing['n_experts']}")
        results["es_global"] = {
            "perplexity": ppl, "routing": routing,
            "wall_seconds": es_time, "history": es_hist}
        reset_gates()

    # ---- Arm: ES partitioned ----
    if "es_partitioned" in arms:
        print(f"\n{'='*60}")
        print("Arm: ES partitioned (per-expert perturbation)")
        print(f"{'='*60}")
        reset_gates()
        t0 = time.time()
        model, esp_hist = es_optimize_routing(
            model, tokenizer, gate_info, train_texts, device,
            n_pop=args.n_pop, rank=args.rank, sigma=args.sigma,
            alpha=args.alpha, generations=args.generations,
            seed=args.seed, partitioned=True)
        esp_time = time.time() - t0

        ppl = compute_perplexity(model, tokenizer, eval_texts, device)
        routing = compute_routing_metrics(model, gate_names, tokenizer,
                                          eval_texts, device)
        print(f"  Perplexity: {ppl:.2f}  ({esp_time:.0f}s)")
        print(f"  Load balance: {routing['balance']:.4f}  "
              f"Collapsed: {routing['collapsed']}/{routing['n_experts']}")
        results["es_partitioned"] = {
            "perplexity": ppl, "routing": routing,
            "wall_seconds": esp_time, "history": esp_hist}

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<20s} {'Perplexity':>12s} {'Balance':>10s} "
          f"{'Collapsed':>10s} {'Time':>8s}")
    print("-" * 65)
    for method in ["pretrained", "random", "gradient", "es_global", "es_partitioned"]:
        if method not in results:
            continue
        r = results[method]
        wt = f"{r['wall_seconds']:.0f}s" if r['wall_seconds'] else "  --"
        print(f"{method:<20s} {r['perplexity']:>12.2f} "
              f"{r['routing']['balance']:>10.4f} "
              f"{r['routing']['collapsed']:>10d} {wt:>8s}")

    # ---- Save results ----
    output_path = args.output or f"moe_routing_{args.model.split('/')[-1]}.json"
    with open(output_path, "w") as f:
        json.dump({
            "model": args.model,
            "task": args.task,
            "config": {
                "n_pop": args.n_pop, "rank": args.rank,
                "sigma": args.sigma, "alpha": args.alpha,
                "generations": args.generations, "seed": args.seed,
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
