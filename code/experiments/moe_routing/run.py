"""MoE routing experiment: can ES optimise discrete routing better than SGD?

Routing in Mixture-of-Experts is inherently discrete (argmax over experts).
Gradient methods require softmax relaxation; ES evaluates the hard routing
directly. This experiment tests whether that matters.

Three arms:
    1. Gradient-based router  -- Adam on cross-entropy with soft routing
    2. ES router (global)     -- EGGROLL perturbation of all router weights
    3. ES router (partitioned) -- each member perturbs one expert's routing row

All three start from the same pre-trained model (experts trained via SGD).
Only the router weights are optimised; everything else is frozen.

Run:  python code/experiments/moe_routing/run.py
"""

import sys
import time
import math
import copy
from pathlib import Path

import torch
import torch.nn.functional as F

# Make evolve and local modules importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evolve.core.trainer import TrainConfig, Trainer
from evolve.core.parallel import SerialExecutor
from evolve.operators.sampling import IIDSampler, PartitionedSampler
from evolve.operators.weighting import RankWeighting

from model import ToyMoE, make_modular_arithmetic_data, make_split_data
from objective import MoERouterObjective

# ---------------------------------------------------------------- hyperparams

HIDDEN = 128
N_EXPERTS = 4
VOCAB_SIZE = 16
SEQ_LEN = 6
N_CLASSES = 8
SEED = 42

# Pre-training
PRETRAIN_EPOCHS = 80
PRETRAIN_LR = 3e-3
PRETRAIN_BATCH = 256
N_TRAIN = 16000
N_VAL = 2000

# Router optimization
GRAD_STEPS = 300
GRAD_LR = 3e-3
ES_GENS = 150
ES_POP = 64
ES_SIGMA = 0.05
ES_ALPHA = 0.02
ES_RANK = 1

DEVICE = "cpu"


# ------------------------------------------------------------ pre-training

def pretrain_experts(model: ToyMoE, x_train, y_train, x_val, y_val):
    """Train the full model (all params) with soft routing so experts learn."""
    print("Pre-training experts with SGD (soft routing)...")
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    best_acc = 0.0
    for epoch in range(PRETRAIN_EPOCHS):
        # Shuffle
        perm = torch.randperm(x_train.shape[0])
        x_shuf, y_shuf = x_train[perm], y_train[perm]

        total_loss = 0.0
        n_batches = 0
        for i in range(0, x_train.shape[0], PRETRAIN_BATCH):
            xb = x_shuf[i:i+PRETRAIN_BATCH]
            yb = y_shuf[i:i+PRETRAIN_BATCH]
            logits = model(xb, hard=False)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        # Validation accuracy with hard routing
        with torch.no_grad():
            val_logits = model(x_val, hard=True)
            val_acc = float((val_logits.argmax(-1) == y_val).float().mean())
        if val_acc > best_acc:
            best_acc = val_acc
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  loss {total_loss/n_batches:.4f}  "
                  f"val_acc {val_acc:.4f}")

    print(f"  pre-training done, best val_acc = {best_acc:.4f}\n")
    return model


# ---------------------------------------------------------- gradient router

def train_router_gradient(model: ToyMoE, x_train, y_train, x_val, y_val):
    """Optimise the router with Adam on cross-entropy (soft routing for grads)."""
    model = copy.deepcopy(model)
    model.freeze_experts()
    opt = torch.optim.Adam(model.router.parameters(), lr=GRAD_LR)

    t0 = time.time()
    for step in range(GRAD_STEPS):
        # Random batch
        idx = torch.randint(0, x_train.shape[0], (PRETRAIN_BATCH,))
        xb, yb = x_train[idx], y_train[idx]
        logits = model(xb, hard=False)  # soft routing for gradient flow
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    wall_time = time.time() - t0
    return model, wall_time


# --------------------------------------------------------------- ES router

def train_router_es(model: ToyMoE, x_val, y_val, partitioned: bool = False):
    """Optimise the router with EGGROLL ES."""
    model = copy.deepcopy(model)
    obj = MoERouterObjective(model, x_val, y_val, n_eval=N_VAL, device=DEVICE,
                             seed=SEED)

    if partitioned:
        # One part per expert row in the router weight matrix.
        # Router weight is (K, H). We define K parts, one per expert.
        # The PartitionedSampler works on matrix names; since we have one
        # matrix "router", we need a custom approach: we wrap the objective
        # to expose K separate matrices (one row each).
        obj_p = PartitionedRouterObjective(model, x_val, y_val, n_eval=N_VAL,
                                           device=DEVICE, seed=SEED)
        sampler = PartitionedSampler(n_active=1)
    else:
        obj_p = obj
        sampler = IIDSampler()

    cfg = TrainConfig(
        n_pop=ES_POP, rank=ES_RANK, sigma=ES_SIGMA, alpha=ES_ALPHA,
        generations=ES_GENS, seed=SEED, device=DEVICE,
    )
    tr = Trainer(obj_p, sampler, RankWeighting(), cfg,
                 executor=SerialExecutor())

    t0 = time.time()
    tr.run()
    wall_time = time.time() - t0
    tr.close()

    # For partitioned, reassemble the router weight from parts
    if partitioned:
        obj_p.sync_router_to_model()

    return obj_p.model if partitioned else obj.model, wall_time


# ---------------------------------------- partitioned objective (per-expert)

class PartitionedRouterObjective:
    """Wraps MoE router as K separate matrices for PartitionedSampler.

    Each matrix "router_k" is a single row of the router weight (1, H),
    so each ES member perturbs routing to a single expert.
    """

    def __init__(self, model: ToyMoE, x_val, y_val, n_eval=512,
                 device="cpu", seed=0):
        self.model = model.to(device)
        self.model.freeze_experts()
        self.device = device
        self.seed = seed
        self.n_eval = n_eval

        self.x_val = x_val[:n_eval].to(device)
        self.y_val = y_val[:n_eval].to(device)

        self._router_w = self.model.router.weight.data  # (K, H)
        self.K, self.H = self._router_w.shape

    @property
    def shapes(self):
        """One matrix per expert row: router_0 .. router_{K-1}, each (1, H)."""
        return {f"router_{k}": (1, self.H) for k in range(self.K)}

    def evaluate(self, req):
        from evolve.core.types import Evaluation
        N = req.size

        g = torch.Generator().manual_seed(req.crn_seed)
        M = min(self.n_eval, self.x_val.shape[0])
        idx = torch.randperm(self.x_val.shape[0], generator=g)[:M]
        x = self.x_val[idx]
        y = self.y_val[idx]

        per_example = torch.zeros(N, M, device=self.device)

        with torch.no_grad():
            one_hot = F.one_hot(x.long(), self.model.vocab_size).float()
            h = self.model.embedding(one_hot).mean(dim=1)

            for i in range(N):
                # Build perturbed router from per-expert rows
                w_pert = self._router_w.clone()
                s = req.rank ** -0.5
                for k in range(self.K):
                    A_k, B_k = req.factors[f"router_{k}"]
                    delta = torch.einsum("mr,nr->mn", A_k[i], B_k[i]) * s * req.sigma
                    w_pert[k:k+1] += delta

                routing_logits = h @ w_pert.T
                chosen = routing_logits.argmax(dim=-1)

                out = torch.zeros_like(h)
                for k in range(self.K):
                    mask = chosen == k
                    if mask.any():
                        out[mask] = self.model.experts[k](h[mask])

                logits = self.model.output_head(out + h)
                preds = logits.argmax(dim=-1)
                per_example[i] = (preds == y).float()

        fitness = per_example.mean(dim=1)
        return Evaluation(fitness=fitness, per_example=per_example)

    def parent_fitness(self, req):
        g = torch.Generator().manual_seed(req.crn_seed)
        M = min(self.n_eval, self.x_val.shape[0])
        idx = torch.randperm(self.x_val.shape[0], generator=g)[:M]
        x = self.x_val[idx]
        y = self.y_val[idx]

        with torch.no_grad():
            logits = self.model(x, hard=True)
            preds = logits.argmax(dim=-1)
            return float((preds == y).float().mean())

    def apply_update(self, delta):
        for k in range(self.K):
            key = f"router_{k}"
            if key in delta:
                self._router_w[k:k+1] += delta[key].to(self._router_w.device)

    def sync_router_to_model(self):
        """No-op: _router_w is already model.router.weight.data."""
        pass

    def compute_metrics(self, x, y):
        """Delegate to the same logic as MoERouterObjective."""
        with torch.no_grad():
            logits, routing_logits = self.model(x, hard=True, return_routing=True)
            preds = logits.argmax(dim=-1)
            accuracy = float((preds == y).float().mean())

            chosen = routing_logits.argmax(dim=-1)
            K = self.model.n_experts
            counts = torch.bincount(chosen, minlength=K).float()
            utilization = counts / counts.sum()

            p = utilization.clamp_min(1e-8)
            entropy = float(-(p * p.log()).sum())
            max_entropy = math.log(K)
            balance = entropy / max_entropy

            collapsed = int((utilization < 0.01).sum())

        return {
            "accuracy": accuracy,
            "utilization": utilization.tolist(),
            "entropy": entropy,
            "balance": balance,
            "collapsed_experts": collapsed,
        }


# ------------------------------------------------------------ evaluation

def evaluate_model(model: ToyMoE, x, y, label: str) -> dict:
    """Compute all metrics for a model."""
    with torch.no_grad():
        logits, routing_logits = model(x, hard=True, return_routing=True)
        preds = logits.argmax(dim=-1)
        accuracy = float((preds == y).float().mean())

        chosen = routing_logits.argmax(dim=-1)
        K = model.n_experts
        counts = torch.bincount(chosen, minlength=K).float()
        utilization = counts / counts.sum()

        p = utilization.clamp_min(1e-8)
        entropy = float(-(p * p.log()).sum())
        max_entropy = math.log(K)
        balance = entropy / max_entropy

        collapsed = int((utilization < 0.01).sum())

    return {
        "label": label,
        "accuracy": accuracy,
        "utilization": [f"{u:.3f}" for u in utilization.tolist()],
        "entropy": entropy,
        "balance": balance,
        "collapsed_experts": collapsed,
    }


# ------------------------------------------------------------------- main

def main():
    print("=" * 72)
    print("MoE Routing Experiment: ES vs Gradient-Based Router Optimization")
    print("=" * 72)
    print(f"\nModel: {N_EXPERTS} experts, hidden={HIDDEN}, vocab={VOCAB_SIZE}, "
          f"classes={N_CLASSES}")
    print(f"Task: modular arithmetic (sum of tokens mod {N_CLASSES})")
    print(f"ES: pop={ES_POP}, rank={ES_RANK}, sigma={ES_SIGMA}, "
          f"alpha={ES_ALPHA}, gens={ES_GENS}")
    print(f"Gradient: Adam lr={GRAD_LR}, steps={GRAD_STEPS}")
    print()

    # ---- data
    x_train, y_train, x_val, y_val = make_split_data(
        n_train=N_TRAIN, n_val=N_VAL, seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE, n_classes=N_CLASSES, seed=SEED
    )

    # ---- build and pre-train
    model = ToyMoE(vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN, hidden=HIDDEN,
                    n_experts=N_EXPERTS, n_classes=N_CLASSES, seed=SEED)
    print(f"Total parameters:  {model.n_params():,}")
    print(f"Router parameters: {model.router_params():,}")
    print()

    model = pretrain_experts(model, x_train, y_train, x_val, y_val)

    # Baseline: pre-trained model before router optimization
    baseline = evaluate_model(model, x_val, y_val, "Pre-trained (baseline)")
    print(f"Baseline accuracy: {baseline['accuracy']:.4f}")
    print(f"Baseline balance:  {baseline['balance']:.4f}")
    print(f"Baseline routing:  {baseline['utilization']}")
    print()

    # Reset router to near-uniform so all three arms start from the same point.
    # This makes the comparison fair: we are testing who optimises routing best,
    # not who benefits most from the pre-trained router.
    base_model = copy.deepcopy(model)
    torch.manual_seed(SEED + 7)
    nn_init_val = 0.01
    base_model.router.weight.data.normal_(0, nn_init_val)

    reset_baseline = evaluate_model(base_model, x_val, y_val,
                                     "After router reset")
    print(f"After router reset: accuracy={reset_baseline['accuracy']:.4f}, "
          f"balance={reset_baseline['balance']:.4f}")
    print()

    # ---- Arm 1: Gradient-based router
    print("-" * 72)
    print("Arm 1: Gradient-based router (Adam, soft routing)")
    print("-" * 72)
    grad_model, grad_time = train_router_gradient(
        base_model, x_train, y_train, x_val, y_val)
    grad_result = evaluate_model(grad_model, x_val, y_val, "Gradient (Adam)")
    grad_result["wall_time"] = grad_time
    print(f"  accuracy={grad_result['accuracy']:.4f}  "
          f"balance={grad_result['balance']:.4f}  "
          f"time={grad_time:.1f}s")
    print()

    # ---- Arm 2: ES router (global)
    print("-" * 72)
    print("Arm 2: ES router (global perturbation)")
    print("-" * 72)
    es_global_model, es_global_time = train_router_es(
        base_model, x_val, y_val, partitioned=False)
    es_global_result = evaluate_model(es_global_model, x_val, y_val,
                                       "ES (global)")
    es_global_result["wall_time"] = es_global_time
    print(f"  accuracy={es_global_result['accuracy']:.4f}  "
          f"balance={es_global_result['balance']:.4f}  "
          f"time={es_global_time:.1f}s")
    print()

    # ---- Arm 3: ES router (partitioned, per-expert)
    print("-" * 72)
    print("Arm 3: ES router (partitioned, per-expert perturbation)")
    print("-" * 72)
    es_part_model, es_part_time = train_router_es(
        base_model, x_val, y_val, partitioned=True)
    es_part_result = evaluate_model(es_part_model, x_val, y_val,
                                     "ES (partitioned)")
    es_part_result["wall_time"] = es_part_time
    print(f"  accuracy={es_part_result['accuracy']:.4f}  "
          f"balance={es_part_result['balance']:.4f}  "
          f"time={es_part_time:.1f}s")
    print()

    # ---- comparison table
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    results = [
        reset_baseline,
        baseline,
        grad_result,
        es_global_result,
        es_part_result,
    ]

    hdr = f"{'Method':<30s} {'Acc':>6s} {'Balance':>8s} {'Collapsed':>10s} " \
          f"{'Time':>7s}  {'Expert utilization'}"
    print(hdr)
    print("-" * len(hdr) + "-" * 30)
    for r in results:
        wt = f"{r.get('wall_time', 0):.1f}s" if r.get('wall_time') else "  --  "
        print(f"{r['label']:<30s} {r['accuracy']:>6.4f} {r['balance']:>8.4f} "
              f"{r['collapsed_experts']:>10d} {wt:>7s}  {r['utilization']}")

    print()
    print("Balance: 1.0 = perfectly uniform routing, 0.0 = all tokens to one expert")
    print("Collapsed: number of experts receiving < 1% of tokens")
    print()

    # ---- verdict
    best = max(results[2:], key=lambda r: r["accuracy"])
    print(f"Best router optimizer: {best['label']} "
          f"(accuracy={best['accuracy']:.4f})")

    if es_part_result["balance"] > grad_result["balance"]:
        print("Partitioned ES achieved better load balance than gradient-based routing.")
    if es_part_result["collapsed_experts"] < grad_result["collapsed_experts"]:
        print("Partitioned ES collapsed fewer experts than gradient-based routing.")


if __name__ == "__main__":
    main()
