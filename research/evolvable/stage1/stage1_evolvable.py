#!/usr/bin/env python3
"""
Stage 1 -- Evolvable Representations: synthetic task family with known ground truth.

Tests, on a world whose true generative factors are known:
  RQ1  Do landscape statistics (epistasis eps, task distance k) predict how many reward
       evaluations an EA needs to adapt?
  RQ2  Does evolution-in-the-loop (EIL) training shape the encoder towards lower epistasis /
       smaller task distance than generic training (multi-task, autoencoder) at similar capacity?
  RQ3  Does that translate into faster REWARD-ONLY adaptation (fresh tasks and task
       transitions), compared with EAs on other representations and with CMA-ES on a
       continuous head?

Everything runs on CPU (numpy + small torch MLPs). Designed for Colab / Euler.

  python stage1_evolvable.py --mode quick            # ~10-20 min on a Colab CPU, 2 seeds
  python stage1_evolvable.py --mode full --seeds 0 1 2 3 4
  python stage1_evolvable.py --sanity-only           # only the correctness checks

Outputs (in --out): results.json (everything), summary.csv, summary.txt, plots (*.png).

Conventions (match the proposal, research/evolvable/evolvable_representations.tex):
  features f(x) in {0,1}^n, genome g in {-1,0,1}^n, m_g(x) = <g, f(x)> - THETA, THETA = 0.5,
  prediction sign(m), reward = batch accuracy. The EA only ever sees the scalar batch reward.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
except ImportError as e:  # pragma: no cover
    raise SystemExit("PyTorch is required: pip install torch") from e

try:
    import cma
except ImportError as e:  # pragma: no cover
    raise SystemExit("pycma is required: pip install cma") from e

import warnings
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

THETA = 0.5          # read-out threshold (non-integer => no ties, lemma applies)
BAND = 4.0           # epistatic band |m| < 4 from the lemma


# =====================================================================================
# Configuration
# =====================================================================================
@dataclass
class Config:
    # world
    D: int = 16                  # latent binary factors
    dx: int = 64                 # observation dim
    noise: float = 0.05          # observation noise
    task_size: int = 3           # |S_tau|
    train_frac: float = 0.7      # fraction of tasks used for (meta-)training
    # representation
    n: int = 64                  # number of binary features = genome length
    hidden: int = 128
    # training of encoders
    train_steps: int = 3000
    train_batch: int = 256
    tasks_per_step: int = 64     # EIL: tasks per outer step (multitask uses ALL training tasks)
    lr: float = 2e-3
    head_lr: float = 1e-2        # multitask heads (strong, well-tuned baseline)
    # evolution-in-the-loop
    inner_iters: int = 30        # inner (1+1)-EA iterations per task visit (forced mutation)
    gamma_plain: float = 1.0     # hinge margin for plain EIL
    gamma_margin: float = 4.5    # hinge margin that pushes examples out of the band
    lam_co: float = 0.3
    lam_band: float = 0.3
    lam_co_multitask: float = 0.3
    lam_rate: float = 10.0       # guard against dead units when penalising co-activation
    min_rate: float = 0.05       # minimum firing rate per unit enforced by the guard
    # deployment / evaluation
    B_dep: int = 512             # batch on which the EA sees its reward (fixed per task = CRN)
    B_test: int = 4096           # held-out evaluation batch
    B_meas: int = 2048           # batch for landscape statistics
    budget: int = 3000           # reward evaluations per adaptation run
    target: float = 0.95         # absolute success threshold on the deployment reward
    rel_target: float = 0.95     # relative threshold: fraction of the run's own final reward
    n_eval_tasks: int = 24       # fresh held-out tasks per (seed, representation)
    lam_pop: int = 16            # lambda of the (1+lambda) EA
    cma_sigma0: float = 0.5
    eps_pairs: int = 256         # random (i,j) pairs per genome snapshot for epistasis
    snap_evals: tuple = (64, 256, 1024)   # + final genome
    reps: tuple = ("ideal", "random", "autoencoder", "multitask", "multitask_co",
                   "eil", "eil_full")
    run_cma: bool = True


def make_config(mode: str) -> Config:
    if mode == "quick":
        return Config()
    if mode == "full":
        return Config(train_steps=4000, budget=6000, n_eval_tasks=48,
                      reps=("ideal", "random", "autoencoder", "multitask", "multitask_co",
                            "eil", "eil_margin", "eil_co", "eil_full"))
    if mode == "smoke":   # for testing the pipeline end-to-end in ~1-2 minutes
        return Config(train_steps=150, budget=600, n_eval_tasks=6, B_test=1024, B_meas=512,
                      eps_pairs=64)
    raise ValueError(mode)


# =====================================================================================
# World: latent factors -> entangled observations; task family = sparse threshold rules
# =====================================================================================
class World:
    def __init__(self, cfg: Config, seed: int):
        rng = np.random.default_rng(10_000 + seed)
        self.cfg = cfg
        D, dx = cfg.D, cfg.dx
        self.M1 = (rng.normal(size=(dx, D)) * (1.5 / math.sqrt(D))).astype(np.float32)
        self.b1 = (rng.normal(size=dx) * 0.3).astype(np.float32)
        self.M2 = (rng.normal(size=(dx, dx)) / math.sqrt(dx)).astype(np.float32)
        # task family: all (S, a) with |S| = task_size, a in {-1,+1}^|S|
        self.tasks = []
        for S in itertools.combinations(range(D), cfg.task_size):
            for a in itertools.product((-1, 1), repeat=cfg.task_size):
                self.tasks.append((S, a))
        T = len(self.tasks)
        A = np.zeros((T, D), dtype=np.float32)
        for t, (S, a) in enumerate(self.tasks):
            A[t, list(S)] = a
        self.A = A                                         # (T, D)
        self.index = {task: t for t, task in enumerate(self.tasks)}
        perm = rng.permutation(T)
        n_train = int(cfg.train_frac * T)
        self.train_ids = np.sort(perm[:n_train])
        self.test_ids = np.sort(perm[n_train:])
        self.is_test = np.zeros(T, dtype=bool)
        self.is_test[self.test_ids] = True

    def sample(self, B: int, rng: np.random.Generator):
        s = rng.integers(0, 2, size=(B, self.cfg.D)).astype(np.float32)
        pre = (2 * s - 1) @ self.M1.T + self.b1 + self.cfg.noise * rng.normal(size=(B, self.cfg.dx))
        x = np.tanh(pre).astype(np.float32) @ self.M2.T
        return s, x.astype(np.float32)

    def labels(self, s: np.ndarray, task_ids) -> np.ndarray:
        """y in {-1,+1}, shape (B, len(task_ids)). Sums are odd => never zero."""
        z = (2 * s - 1) @ self.A[np.asarray(task_ids)].T
        return np.sign(z).astype(np.float32)

    def successors(self, t: int):
        """Related tasks: replace one factor of S (either sign) or flip one sign."""
        S, a = self.tasks[t]
        out = []
        for pos in range(len(S)):
            # flip sign
            a2 = list(a); a2[pos] = -a2[pos]
            out.append((S, tuple(a2)))
            # replace factor
            for j in range(self.cfg.D):
                if j in S:
                    continue
                for sg in (-1, 1):
                    S2 = list(S); a3 = list(a)
                    S2[pos] = j; a3[pos] = sg
                    order = np.argsort(S2)
                    out.append((tuple(np.array(S2)[order].tolist()), tuple(np.array(a3)[order].tolist())))
        return sorted({self.index[u] for u in out} - {t})


def ideal_features(s: np.ndarray, n: int) -> np.ndarray:
    """Oracle: [s, 1-s] tiled to n columns (n must be a multiple of 2D)."""
    base = np.concatenate([s, 1 - s], axis=1)
    reps = int(math.ceil(n / base.shape[1]))
    return np.tile(base, (1, reps))[:, :n].astype(np.float32)


def ideal_genome(world: World, t: int, n: int, copies: int = 1) -> np.ndarray:
    D = world.cfg.D
    g = np.zeros(n, dtype=np.int8)
    S, a = world.tasks[t]
    for c in range(copies):
        for j, aj in zip(S, a):
            g[c * 2 * D + j] = aj
            g[c * 2 * D + D + j] = -aj
    return g


# =====================================================================================
# Fitness and evolutionary algorithms (vectorised over many tasks / candidates)
# =====================================================================================
def fitness(feats: np.ndarray, Y: np.ndarray, G: np.ndarray) -> np.ndarray:
    """feats (B,n) in {0,1}; Y (B,K) or (B,1) in {-1,+1}; G (K,n) in {-1,0,1}. Returns (K,)."""
    m = feats @ G.T.astype(np.float32) - THETA
    return (Y * m > 0).mean(axis=0)


def mutate(G: np.ndarray, rng: np.random.Generator, rate: float, force: bool):
    """Ternary mutation: each position w.p. rate moves to a uniformly random OTHER value."""
    K, n = G.shape
    mask = rng.random((K, n)) < rate
    if force:
        none = ~mask.any(axis=1)
        if none.any():
            rows = np.flatnonzero(none)
            mask[rows, rng.integers(0, n, size=rows.size)] = True
    shift = rng.integers(1, 3, size=(K, n))
    Gm = ((G.astype(np.int16) + 1 + shift) % 3 - 1).astype(np.int8)
    return np.where(mask, Gm, G), mask.any(axis=1)


def one_plus_one(feats, Y, G0, budget, target, rng, snap_evals=()):
    """Standard (1+1) EA (mutation rate 1/n), vectorised over K independent runs.
    Only offspring that differ from the parent are evaluated/counted (identical offspring
    carry no information). Acceptance: offspring >= parent (elitist, accepts ties).
    Returns dict with final genomes, fitness, evals-to-target (-1 = never), AUC, snapshots."""
    G = G0.copy()
    K, n = G.shape
    f = fitness(feats, Y, G)
    evals = np.zeros(K, dtype=np.int64)
    hit = np.where(f >= target, 0, -1)
    auc = np.zeros(K)
    snaps = np.repeat(G[None], len(snap_evals), axis=0)
    rec = np.zeros((len(snap_evals), K), dtype=bool)
    traj = np.zeros((K, budget), dtype=np.float32)       # best-so-far reward after each counted eval
    G_hit = G.copy()                                      # genome when the target was first reached
    while (evals < budget).any():
        active = evals < budget
        Gc, changed = mutate(G, rng, 1.0 / n, force=False)
        counted = changed & active
        if not counted.any():
            continue
        fc = fitness(feats, Y, Gc)
        evals += counted
        acc = counted & (fc >= f)
        G[acc] = Gc[acc]
        f[acc] = fc[acc]
        auc += np.where(counted, f, 0.0)
        rows = np.flatnonzero(counted)
        traj[rows, evals[rows] - 1] = f[rows]
        newly = (hit < 0) & (f >= target) & counted
        hit[newly] = evals[newly]
        G_hit[newly] = G[newly]
        for si, se in enumerate(snap_evals):
            nw = (~rec[si]) & (evals >= se)
            if nw.any():
                snaps[si, nw] = G[nw]
                rec[si, nw] = True
    for si in range(len(snap_evals)):
        snaps[si, ~rec[si]] = G[~rec[si]]
    G_hit[hit < 0] = G[hit < 0]
    return dict(G=G, f=f, hit=hit, auc=auc / budget, snaps=snaps, traj=traj, G_hit=G_hit)


def hits_from_traj(traj, f0, thresholds):
    """First counted evaluation at which best-so-far reward >= threshold (per run); 0 if the
    start already satisfies it; -1 if never. thresholds: (K,) array."""
    K, T = traj.shape
    out = np.full(K, -1, dtype=np.int64)
    for k in range(K):
        if f0[k] >= thresholds[k]:
            out[k] = 0
            continue
        idx = np.flatnonzero(traj[k] >= thresholds[k])
        if idx.size:
            out[k] = idx[0] + 1
    return out


def prune_towards(feats, Y, G_start, G_final, rng, sweeps=2):
    """Remove neutral drift: revert differing positions of G_final back to G_start whenever the
    reward does not drop. Returns pruned genomes and k_essential = #positions still differing.
    k_essential is an upper bound on the number of changes actually needed to move from
    G_start to a solution as good as G_final."""
    G = G_final.copy()
    K, n = G.shape
    f_ref = fitness(feats, Y, G)
    for _ in range(sweeps):
        for pos in rng.permutation(n):
            diff = G[:, pos] != G_start[:, pos]
            if not diff.any():
                continue
            Gc = G.copy()
            Gc[diff, pos] = G_start[diff, pos]
            fc = fitness(feats, Y, Gc)
            ok = diff & (fc >= f_ref)
            G[ok] = Gc[ok]
    return G, (G != G_start).sum(1)


def one_plus_lambda(feats, Y, G0, budget, target, rng, lam):
    """(1+lambda) EA with forced mutation (>=1 position changed), rate 1/n.
    Counts lam evaluations per generation (parallel workers)."""
    G = G0.copy()
    K, n = G.shape
    f = fitness(feats, Y, G)
    evals = 0
    hit = np.where(f >= target, 0, -1)
    auc = np.zeros(K)
    while evals < budget:
        Gc, _ = mutate(np.repeat(G, lam, axis=0), rng, 1.0 / n, force=True)   # (K*lam, n)
        Yr = np.repeat(Y, lam, axis=1)
        fc = fitness(feats, Yr, Gc).reshape(K, lam)
        best = fc.argmax(axis=1)
        fb = fc[np.arange(K), best]
        take = fb >= f
        Gb = Gc.reshape(K, lam, n)[np.arange(K), best]
        G[take] = Gb[take]
        f[take] = fb[take]
        # within the generation, the best-so-far can only be credited at its end (parallel)
        auc += f * lam
        evals += lam
        newly = (hit < 0) & (f >= target)
        hit[newly] = evals
    return dict(G=G, f=f, hit=hit, auc=auc / evals)


def cma_head(feats, y, w0, budget, target, seed, sigma0):
    """CMA-ES on a continuous linear head (n weights + bias), reward = batch accuracy.
    Restarts from the incumbent if pycma stops early. Returns best head and stats."""
    n1 = feats.shape[1] + 1
    Xb = np.concatenate([feats, np.ones((feats.shape[0], 1), np.float32)], axis=1)
    evals, best_f, best_w, hit, auc = 0, -1.0, w0.copy(), -1, 0.0
    hit90 = -1
    start, restart = w0.copy(), 0
    while evals < budget:
        es = cma.CMAEvolutionStrategy(start, sigma0, {
            "verbose": -9, "seed": seed * 1000 + restart + 1,
            "maxfevals": budget - evals, "tolflatfitness": 50})
        while not es.stop() and evals < budget:
            W = np.asarray(es.ask())
            fv = ((Xb @ W.T) * y[:, None] > 0).mean(axis=0)
            for i, fi in enumerate(fv):
                if evals >= budget:
                    break
                evals += 1
                if fi > best_f:
                    best_f, best_w = float(fi), W[i].copy()
                auc += best_f
                if hit < 0 and best_f >= target:
                    hit = evals
                if hit90 < 0 and best_f >= 0.90:
                    hit90 = evals
            es.tell(list(W[:len(fv)]), list(-fv))
        start, restart = best_w.copy(), restart + 1
    return dict(w=best_w, f=best_f, hit=hit, hit90=hit90, auc=auc / budget)


def head_accuracy(feats, y, w):
    Xb = np.concatenate([feats, np.ones((feats.shape[0], 1), np.float32)], axis=1)
    return float(((Xb @ w) * y > 0).mean())


# =====================================================================================
# Encoders and their training
# =====================================================================================
class Encoder(nn.Module):
    def __init__(self, dx, hidden, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dx, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n))

    def forward(self, x):
        h = self.net(x)
        soft = torch.sigmoid(h)
        hard = (h > 0).float()
        ste = hard + soft - soft.detach()       # forward: hard bits, backward: sigmoid grad
        return ste, soft, hard

    @torch.no_grad()
    def features(self, x_np: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x_np), 8192):
            h = self.net(torch.from_numpy(x_np[i:i + 8192]))
            out.append((h > 0).float().numpy())
        return np.concatenate(out).astype(np.float32)


def coactivation_penalty(ste: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal P[f_i = f_j = 1] of the HARD bits (gradient via the STE)."""
    B, n = ste.shape
    C = ste.T @ ste / B
    return (C.sum() - torch.diagonal(C).sum()) / (n * (n - 1))


def rate_guard(ste: torch.Tensor, min_rate: float) -> torch.Tensor:
    """Penalise units whose (hard) firing rate drops below min_rate (co-activation penalties are
    otherwise trivially minimised by switching all features off). Gradient via the STE."""
    rate = ste.mean(dim=0)
    return (Fnn.relu(min_rate - rate) ** 2).sum()


def band_penalty(ste: torch.Tensor, m: torch.Tensor, temp: float = 0.5) -> torch.Tensor:
    """Smooth version of the lemma bound: E[c(c-1)/(n(n-1)) * kappa(m)], kappa ~ 1[|m| < 4]."""
    n = ste.shape[1]
    c = ste.sum(dim=1, keepdim=True)
    kappa = torch.sigmoid((BAND - m.abs()) / temp)
    return (c * (c - 1) / (n * (n - 1)) * kappa).mean()


def train_encoder(kind: str, world: World, cfg: Config, seed: int, log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(20_000 + seed)
    enc = Encoder(cfg.dx, cfg.hidden, cfg.n)
    if kind == "random":
        return enc, {}
    params = list(enc.parameters())
    head_params = []
    train_ids = world.train_ids
    Tn = len(train_ids)
    extra = {}
    if kind in ("multitask", "multitask_co"):
        heads = nn.Parameter(torch.randn(Tn, cfg.n) * 0.05)
        bias = nn.Parameter(torch.zeros(Tn))
        head_params = [heads, bias]
    elif kind == "autoencoder":
        dec = nn.Sequential(nn.Linear(cfg.n, cfg.hidden), nn.GELU(), nn.Linear(cfg.hidden, cfg.dx))
        params += list(dec.parameters())
    elif kind.startswith("eil"):
        archive = np.zeros((Tn, cfg.n), dtype=np.int8)       # persistent genome per training task
        gamma = cfg.gamma_margin if kind in ("eil_margin", "eil_full") else cfg.gamma_plain
        lam_co = cfg.lam_co if kind in ("eil_co", "eil_full") else 0.0
        lam_band = cfg.lam_band if kind == "eil_full" else 0.0
    else:
        raise ValueError(kind)
    groups = [dict(params=params, lr=cfg.lr)]
    if head_params:
        groups.append(dict(params=head_params, lr=cfg.head_lr))
    opt = torch.optim.Adam(groups)
    hist = []
    t0 = time.time()
    for step in range(cfg.train_steps):
        s, x = world.sample(cfg.train_batch, rng)
        xt = torch.from_numpy(x)
        ste, soft, hard = enc(xt)
        if kind in ("multitask", "multitask_co"):
            loc = np.arange(Tn)                                           # all training tasks
        else:
            loc = rng.choice(Tn, size=min(cfg.tasks_per_step, Tn), replace=False)
        tids = train_ids[loc]
        Y = world.labels(s, tids)                                         # (B, M)
        if kind in ("multitask", "multitask_co"):
            logits = ste @ heads[loc].T + bias[loc]
            loss = Fnn.binary_cross_entropy_with_logits(logits, torch.from_numpy((Y + 1) / 2))
            if kind == "multitask_co":
                loss = loss + cfg.lam_co_multitask * coactivation_penalty(ste) \
                    + cfg.lam_rate * rate_guard(ste, cfg.min_rate)
        elif kind == "autoencoder":
            loss = Fnn.mse_loss(dec(ste), xt)
        else:
            # ---- inner loop: (1+1) EA with forced mutation, reward only, on hard features ----
            feats = hard.numpy()
            G = archive[loc].copy()
            f = fitness(feats, Y, G)
            for _ in range(cfg.inner_iters):
                Gc, _ = mutate(G, rng, 1.0 / cfg.n, force=True)
                fc = fitness(feats, Y, Gc)
                acc = fc >= f
                G[acc] = Gc[acc]
                f[acc] = fc[acc]
            archive[loc] = G
            # ---- outer loop: gradient at the genomes the EA found ----
            Gt = torch.from_numpy(G.astype(np.float32))
            m = ste @ Gt.T - THETA                                        # (B, M)
            Yt = torch.from_numpy(Y)
            loss = Fnn.relu(gamma - Yt * m).mean()
            if lam_co > 0:
                loss = loss + lam_co * coactivation_penalty(ste) + cfg.lam_rate * rate_guard(ste, cfg.min_rate)
            if lam_band > 0:
                loss = loss + lam_band * band_penalty(ste, m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, cfg.train_steps // 10) == 0 or step == cfg.train_steps - 1:
            rec = dict(step=step, loss=float(loss.detach()))
            if kind.startswith("eil"):
                rec["inner_fit"] = float(f.mean())
            hist.append(rec)
    extra["history"] = hist
    extra["train_time_s"] = time.time() - t0
    log(f"    trained {kind:13s} in {extra['train_time_s']:.1f}s, final loss {hist[-1]['loss']:.4f}"
        + (f", inner fitness {hist[-1]['inner_fit']:.3f}" if kind.startswith('eil') else ""))
    return enc, extra


# =====================================================================================
# Measurements
# =====================================================================================
def landscape_stats(feats, Y, G_snaps, rng, pairs):
    """Relative epistasis per task (pooled over genome snapshots) + exact check of the lemma.
    feats (B,n); Y (B,K); G_snaps (S,K,n)."""
    S, K, n = G_snaps.shape
    num = np.zeros(K); den = np.zeros(K); nz = np.zeros(K)
    viol = 0; checked = 0; max_ratio = 0.0
    for si in range(S):
        for k in range(K):
            g = G_snaps[si, k]
            i = rng.integers(0, n, size=pairs)
            j = (i + rng.integers(1, n, size=pairs)) % n                       # j != i
            vi = ((g[i].astype(np.int16) + 1 + rng.integers(1, 3, size=pairs)) % 3 - 1).astype(np.int8)
            vj = ((g[j].astype(np.int16) + 1 + rng.integers(1, 3, size=pairs)) % 3 - 1).astype(np.int8)
            Gi = np.repeat(g[None], pairs, 0); Gi[np.arange(pairs), i] = vi
            Gj = np.repeat(g[None], pairs, 0); Gj[np.arange(pairs), j] = vj
            Gij = Gi.copy(); Gij[np.arange(pairs), j] = vj
            f0 = fitness(feats, Y[:, k:k + 1], g[None])[0]
            fi = fitness(feats, Y[:, k:k + 1], Gi)
            fj = fitness(feats, Y[:, k:k + 1], Gj)
            fij = fitness(feats, Y[:, k:k + 1], Gij)
            d1 = np.abs(np.concatenate([fi - f0, fj - f0]))
            d2 = np.abs(fij - fi - fj + f0)
            num[k] += d2.mean(); den[k] += d1.mean(); nz[k] += (d2 > 1e-12).mean()
            # lemma: |d2| <= 2 P[f_i = f_j = 1, |m_g| < 4]
            mg = feats @ g.astype(np.float32) - THETA
            inband = (np.abs(mg) < BAND).astype(np.float32)
            bound = 2 * ((feats[:, i] * feats[:, j]) * inband[:, None]).mean(axis=0)
            viol += int((d2 > bound + 1e-9).sum()); checked += pairs
            ok = bound > 0
            if ok.any():
                max_ratio = max(max_ratio, float((d2[ok] / bound[ok]).max()))
    eps = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return dict(eps=eps, frac_nonzero_d2=nz / S, lemma_violations=viol, lemma_checked=checked,
                lemma_max_ratio=max_ratio)


def feature_stats(feats, G_final):
    B, n = feats.shape
    rate = feats.mean(0)
    C = feats.T @ feats / B
    off = (C.sum() - np.trace(C)) / (n * (n - 1))
    c = feats.sum(1)
    band = []
    for g in G_final:
        mg = feats @ g.astype(np.float32) - THETA
        band.append(float((np.abs(mg) < BAND).mean()))
    return dict(firing_rate_mean=float(rate.mean()), dead_frac=float((rate < 0.01).mean()),
                saturated_frac=float((rate > 0.99).mean()), coact_offdiag=float(off),
                active_mean=float(c.mean()), band_mass_at_solution=float(np.mean(band)),
                genome_nonzero_mean=float((G_final != 0).sum(1).mean()))


def factor_decodability(feats, s):
    """How linearly decodable is each ground-truth factor from the binary features
    (logistic regression, 50/50 split). Mean accuracy over factors."""
    half = len(feats) // 2
    accs = []
    for d in range(s.shape[1]):
        clf = LogisticRegression(C=10.0, max_iter=2000)
        clf.fit(feats[:half], s[:half, d])
        accs.append(clf.score(feats[half:], s[half:, d]))
    return float(np.mean(accs))


# =====================================================================================
# Sanity checks (these must pass before any result is trusted)
# =====================================================================================
def sanity_checks(cfg: Config, log=print):
    log("=" * 80)
    log("SANITY CHECKS")
    rng = np.random.default_rng(0)
    world = World(cfg, seed=0)
    ok = True

    # 1. vectorised fitness == naive loop
    s, x = world.sample(300, rng)
    feats = (rng.random((300, cfg.n)) < 0.5).astype(np.float32)
    tids = rng.choice(len(world.tasks), 5, replace=False)
    Y = world.labels(s, tids)
    G = rng.integers(-1, 2, size=(5, cfg.n)).astype(np.int8)
    fv = fitness(feats, Y, G)
    naive = np.array([np.mean([1.0 if Y[b, k] * (np.dot(G[k], feats[b]) - THETA) > 0 else 0.0
                               for b in range(300)]) for k in range(5)])
    c1 = np.allclose(fv, naive)
    log(f"  [{'PASS' if c1 else 'FAIL'}] vectorised fitness equals naive loop"); ok &= c1

    # 2. labels balanced and never zero
    s, x = world.sample(20000, rng)
    Yall = world.labels(s, np.arange(0, len(world.tasks), 97))
    c2 = (np.abs(Yall) == 1).all() and abs(Yall.mean()) < 0.03
    log(f"  [{'PASS' if c2 else 'FAIL'}] labels in {{-1,+1}}, mean {Yall.mean():+.4f}"); ok &= c2

    # 3. oracle features + analytic genome give accuracy 1 (world/labels/readout consistent)
    fi = ideal_features(s, cfg.n)
    accs = [fitness(fi, world.labels(s, [t]), ideal_genome(world, t, cfg.n)[None])[0]
            for t in rng.choice(len(world.tasks), 50, replace=False)]
    c3 = np.min(accs) == 1.0
    log(f"  [{'PASS' if c3 else 'FAIL'}] ideal features + analytic genome: min acc {np.min(accs):.4f}"); ok &= c3

    # 4. ternary mutation always moves a mutated position to a DIFFERENT value, uniformly
    G0 = rng.integers(-1, 2, size=(20000, 1)).astype(np.int8)
    Gm, ch = mutate(G0, rng, 1.0, force=True)
    diff = (Gm != G0).all()
    trans = np.mean(Gm[G0 == 0] == 1)
    c4 = diff and abs(trans - 0.5) < 0.02
    log(f"  [{'PASS' if c4 else 'FAIL'}] mutation changes value (P(0->1)={trans:.3f})"); ok &= c4

    # 5. STE forward = hard bits
    enc = Encoder(cfg.dx, cfg.hidden, cfg.n)
    ste, soft, hard = enc(torch.from_numpy(x[:100]))
    c5 = torch.allclose(ste.detach(), hard, atol=1e-6) and np.array_equal(enc.features(x[:100]), hard.numpy())
    log(f"  [{'PASS' if c5 else 'FAIL'}] straight-through forward equals hard features"); ok &= c5

    # 6. Lemma: no violation on random encoders and random + EA genomes
    s, x = world.sample(1024, rng)
    tids = rng.choice(len(world.tasks), 6, replace=False)
    Y = world.labels(s, tids)
    for name, F in [("random encoder", enc.features(x)), ("ideal", ideal_features(s, cfg.n))]:
        G0 = rng.integers(-1, 2, size=(6, cfg.n)).astype(np.int8)
        run = one_plus_one(F, Y, np.zeros((6, cfg.n), np.int8), 300, 2.0, rng, snap_evals=(50,))
        snaps = np.stack([G0, run["snaps"][0], run["G"]])
        st = landscape_stats(F, Y, snaps, rng, 128)
        c6 = st["lemma_violations"] == 0
        log(f"  [{'PASS' if c6 else 'FAIL'}] lemma holds on {name}: {st['lemma_violations']} violations "
            f"in {st['lemma_checked']} pairs (max |d2|/bound = {st['lemma_max_ratio']:.3f})")
        ok &= c6

    # 7. Proposition: (1+1) EA on OneMax-like ternary function from distance k: E[T] <= 2 e n H_k
    n = 64
    for k in (1, 4, 16):
        R = 1000
        gstar = rng.integers(-1, 2, size=(R, n)).astype(np.int8)
        G = gstar.copy()
        for r in range(R):
            pos = rng.choice(n, k, replace=False)
            G[r, pos] = ((G[r, pos].astype(np.int16) + 1 + rng.integers(1, 3, size=k)) % 3 - 1)
        dist = (G != gstar).sum(1)
        T = np.zeros(R, dtype=np.int64)
        while (dist > 0).any():
            Gc, _ = mutate(G, rng, 1.0 / n, force=False)
            dc = (Gc != gstar).sum(1)
            act = dist > 0
            T += act
            acc = act & (dc <= dist)
            G[acc] = Gc[acc]; dist[acc] = dc[acc]
        Hk = sum(1.0 / i for i in range(1, k + 1))
        bound = 2 * math.e * n * Hk
        sem = T.std(ddof=1) / math.sqrt(R)
        c7 = T.mean() <= bound + 3 * sem          # the bound is on the expectation; allow MC error
        log(f"  [{'PASS' if c7 else 'FAIL'}] (1+1) EA from distance k={k:2d}: mean {T.mean():7.1f} "
            f"(+-{sem:4.1f}) <= bound 2enH_k = {bound:7.1f}  (ratio {T.mean() / bound:.2f})")
        ok &= c7

    # 8. EA on ideal features solves fresh tasks
    s, x = world.sample(cfg.B_dep, rng)
    tt = world.test_ids[:4]
    F = ideal_features(s, cfg.n)
    run = one_plus_one(F, world.labels(s, tt), np.zeros((len(tt), cfg.n), np.int8), 5000, 1.0, rng)
    c8 = (run["f"] >= 0.99).all()
    log(f"  [{'PASS' if c8 else 'FAIL'}] (1+1) EA on ideal features reaches reward >= 0.99 on 4 tasks "
        f"(evals to 1.0: {run['hit'].tolist()})")
    ok &= c8
    log(f"SANITY: {'ALL PASSED' if ok else 'SOME FAILED -- do not trust results'}")
    log("=" * 80)
    return bool(ok)


# =====================================================================================
# Evaluation of one representation
# =====================================================================================
def censored(hit, budget):
    h = np.asarray(hit, dtype=float)
    return np.where(h < 0, budget + 1, h)


def evaluate_representation(name, featurize, world, cfg, seed, log=print):
    """featurize(s, x) -> binary features. Returns metrics dict."""
    rng = np.random.default_rng(30_000 + seed)
    K = cfg.n_eval_tasks
    fresh = rng.choice(world.test_ids, size=K, replace=False)
    # related successors that are also held-out tasks
    succ = []
    for t in fresh:
        cand = [u for u in world.successors(int(t)) if world.is_test[u]]
        succ.append(int(rng.choice(cand)))
    succ = np.array(succ)

    # fixed batches (common random numbers): deployment reward, test, measurement
    s_dep, x_dep = world.sample(cfg.B_dep, rng)
    s_te, x_te = world.sample(cfg.B_test, rng)
    s_me, x_me = world.sample(cfg.B_meas, rng)
    F_dep, F_te, F_me = featurize(s_dep, x_dep), featurize(s_te, x_te), featurize(s_me, x_me)
    Y_dep, Y_te, Y_me = world.labels(s_dep, fresh), world.labels(s_te, fresh), world.labels(s_me, fresh)
    Y2_dep, Y2_te, Y2_me = world.labels(s_dep, succ), world.labels(s_te, succ), world.labels(s_me, succ)
    out = dict(name=name)
    t0 = time.time()

    # ---- capacity references ----
    lr_acc = []
    for k in range(min(K, 12)):
        clf = LogisticRegression(C=10.0, max_iter=3000)
        yk = Y_dep[:, k]
        if len(np.unique(yk)) < 2:
            continue
        clf.fit(F_dep, yk)
        lr_acc.append(clf.score(F_te, Y_te[:, k]))
    out["capacity_logreg_labels"] = float(np.mean(lr_acc))
    out["factor_decodability"] = factor_decodability(F_me, s_me)

    # ---- fresh tasks: (1+1) EA from the zero genome ----
    Z = np.zeros((K, cfg.n), np.int8)
    f0_fresh = fitness(F_dep, Y_dep, Z)
    r1 = one_plus_one(F_dep, Y_dep, Z, cfg.budget, cfg.target, rng, snap_evals=cfg.snap_evals)
    te1 = np.array([fitness(F_te, Y_te[:, k:k + 1], r1["G"][k:k + 1])[0] for k in range(K)])
    rel1 = hits_from_traj(r1["traj"], f0_fresh, f0_fresh + cfg.rel_target * (r1["f"] - f0_fresh))
    h90_1 = hits_from_traj(r1["traj"], f0_fresh, np.full(K, 0.90))
    Gp1, kess1 = prune_towards(F_dep, Y_dep, Z, r1["G"], rng)
    out["ea11_fresh"] = dict(hit=r1["hit"].tolist(), auc=r1["auc"].tolist(),
                             final_dep=r1["f"].tolist(), final_test=te1.tolist(),
                             hit_rel=rel1.tolist(), hit90=h90_1.tolist(), k_essential=kess1.tolist(),
                             nonzero_final=(r1["G"] != 0).sum(1).tolist())
    # ---- fresh tasks: (1+lambda) EA ----
    rl = one_plus_lambda(F_dep, Y_dep, Z, cfg.budget, cfg.target, rng, cfg.lam_pop)
    tel = np.array([fitness(F_te, Y_te[:, k:k + 1], rl["G"][k:k + 1])[0] for k in range(K)])
    out["ea1l_fresh"] = dict(hit=rl["hit"].tolist(), auc=rl["auc"].tolist(),
                             final_dep=rl["f"].tolist(), final_test=tel.tolist())

    # ---- transitions: start the (1+1) EA at the (pruned) genome found for tau, adapt to tau' ----
    # Starting from the pruned genome removes neutral drift accumulated on tau.
    G_start = Gp1
    f0_tr = fitness(F_dep, Y2_dep, G_start)
    r2 = one_plus_one(F_dep, Y2_dep, G_start, cfg.budget, cfg.target, rng, snap_evals=cfg.snap_evals)
    te2 = np.array([fitness(F_te, Y2_te[:, k:k + 1], r2["G"][k:k + 1])[0] for k in range(K)])
    rel2 = hits_from_traj(r2["traj"], f0_tr, f0_tr + cfg.rel_target * (r2["f"] - f0_tr))
    h90_2 = hits_from_traj(r2["traj"], f0_tr, np.full(K, 0.90))
    k_trav = (r2["G"] != G_start).sum(1)
    _, kess2 = prune_towards(F_dep, Y2_dep, G_start, r2["G"], rng)
    out["ea11_transition"] = dict(hit=r2["hit"].tolist(), auc=r2["auc"].tolist(),
                                  final_dep=r2["f"].tolist(), final_test=te2.tolist(),
                                  hit_rel=rel2.tolist(), hit90=h90_2.tolist(), k_travelled=k_trav.tolist(),
                                  k_essential=kess2.tolist(), start_reward=f0_tr.tolist())

    # ---- landscape statistics (fresh tasks, along the (1+1) trajectory) ----
    snaps = np.concatenate([r1["snaps"], r1["G"][None]], axis=0)
    ls = landscape_stats(F_me, Y_me, snaps, rng, cfg.eps_pairs)
    snaps2 = np.concatenate([r2["snaps"], r2["G"][None]], axis=0)
    ls2 = landscape_stats(F_me, Y2_me, snaps2, rng, cfg.eps_pairs)
    out["landscape_fresh"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ls.items()}
    out["landscape_transition"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ls2.items()}
    out["features"] = feature_stats(F_me, r1["G"])

    # ---- CMA-ES on a continuous head (fresh + transitions) ----
    if cfg.run_cma:
        cf, ct = [], []
        for k in range(K):
            a = cma_head(F_dep, Y_dep[:, k], np.zeros(cfg.n + 1), cfg.budget, cfg.target,
                         seed * 100 + k, cfg.cma_sigma0)
            b = cma_head(F_dep, Y2_dep[:, k], a["w"], cfg.budget, cfg.target,
                         seed * 100 + k + 50, cfg.cma_sigma0)
            cf.append((a["hit"], a["auc"], a["f"], head_accuracy(F_te, Y_te[:, k], a["w"]), a["hit90"]))
            ct.append((b["hit"], b["auc"], b["f"], head_accuracy(F_te, Y2_te[:, k], b["w"]), b["hit90"]))
        for key, arr in (("cma_fresh", cf), ("cma_transition", ct)):
            out[key] = dict(hit=[int(a[0]) for a in arr], auc=[float(a[1]) for a in arr],
                            final_dep=[float(a[2]) for a in arr], final_test=[float(a[3]) for a in arr],
                            hit90=[int(a[4]) for a in arr])

    out["eval_time_s"] = time.time() - t0
    b = cfg.budget
    msg = (f"    {name:13s} cap(LR)={out['capacity_logreg_labels']:.3f} "
           f"dec={out['factor_decodability']:.3f} | "
           f"(1+1) fresh: succ={np.mean(r1['hit'] >= 0):.2f} med={np.median(censored(r1['hit'], b)):6.0f} "
           f"test={te1.mean():.3f} | trans: succ={np.mean(r2['hit'] >= 0):.2f} "
           f"med={np.median(censored(r2['hit'], b)):6.0f} k_ess={np.median(kess2):4.1f} | "
           f"eps={np.nanmean(ls['eps']):.3f} lemma_viol={ls['lemma_violations'] + ls2['lemma_violations']}")
    if cfg.run_cma:
        msg += (f" | CMA fresh succ={np.mean(np.array(out['cma_fresh']['hit']) >= 0):.2f} "
                f"test={np.mean(out['cma_fresh']['final_test']):.3f}")
    log(msg + f"  [{out['eval_time_s']:.0f}s]")
    return out


# =====================================================================================
# Driver, aggregation, plots
# =====================================================================================
def run_seed(cfg: Config, seed: int, log=print):
    log(f"\n--- seed {seed} ---")
    world = World(cfg, seed)
    log(f"  world: {len(world.tasks)} tasks ({len(world.train_ids)} train / {len(world.test_ids)} held out)")
    res = {}
    for rep in cfg.reps:
        if rep == "ideal":
            featurize = lambda s, x: ideal_features(s, cfg.n)
            extra = {}
        else:
            enc, extra = train_encoder(rep, world, cfg, seed, log)
            enc.eval()
            featurize = (lambda e: (lambda s, x: e.features(x)))(enc)
        r = evaluate_representation(rep, featurize, world, cfg, seed, log)
        r["train"] = extra
        res[rep] = r
    return res


def aggregate(all_res, cfg):
    """Per representation: mean over seeds of per-seed summaries (+ sem)."""
    b = cfg.budget
    rows = []
    reps = list(cfg.reps)
    algos = ["ea11_fresh", "ea1l_fresh", "ea11_transition"] + (["cma_fresh", "cma_transition"] if cfg.run_cma else [])
    for rep in reps:
        per_seed = []
        for seed, res in all_res.items():
            r = res[rep]
            d = dict(capacity_lr=r["capacity_logreg_labels"], decodability=r["factor_decodability"],
                     eps_fresh=float(np.nanmean(r["landscape_fresh"]["eps"])),
                     eps_trans=float(np.nanmean(r["landscape_transition"]["eps"])),
                     k_trav=float(np.median(r["ea11_transition"]["k_travelled"])),
                     k_ess_trans=float(np.median(r["ea11_transition"]["k_essential"])),
                     k_ess_fresh=float(np.median(r["ea11_fresh"]["k_essential"])),
                     ea11_fresh_relmed=float(np.median(censored(r["ea11_fresh"]["hit_rel"], b))),
                     ea11_transition_relmed=float(np.median(censored(r["ea11_transition"]["hit_rel"], b))),

                     coact=r["features"]["coact_offdiag"], band=r["features"]["band_mass_at_solution"],
                     active=r["features"]["active_mean"], dead=r["features"]["dead_frac"],
                     lemma_viol=r["landscape_fresh"]["lemma_violations"] + r["landscape_transition"]["lemma_violations"])
            for a in algos:
                if "hit90" in r[a]:
                    h9 = np.array(r[a]["hit90"])
                    d[f"{a}_succ90"] = float(np.mean(h9 >= 0))
                    d[f"{a}_med90"] = float(np.median(censored(h9, b)))
                h = np.array(r[a]["hit"])
                d[f"{a}_succ"] = float(np.mean(h >= 0))
                d[f"{a}_med"] = float(np.median(censored(h, b)))
                d[f"{a}_auc"] = float(np.mean(r[a]["auc"]))
                d[f"{a}_test"] = float(np.mean(r[a]["final_test"]))
            per_seed.append(d)
        keys = per_seed[0].keys()
        row = dict(rep=rep, n_seeds=len(per_seed))
        for k in keys:
            v = np.array([p[k] for p in per_seed], dtype=float)
            row[k] = float(np.nanmean(v))
            row[k + "_sem"] = float(np.nanstd(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        rows.append(row)
    return rows


def rq1_correlations(all_res, cfg):
    """Pooled over seeds and representations (excluding the oracle): do epistasis / task
    distance predict adaptation cost? Uses censored log evals-to-0.90, the relative hit time
    and AUC (higher AUC = faster)."""
    b = cfg.budget
    acc = {k: [] for k in ("eps", "auc", "c90", "crel", "k", "c90_t", "crel_t", "eps_t", "auc_t")}
    for seed, res in all_res.items():
        for rep, r in res.items():
            if rep == "ideal":
                continue
            fr, tr = r["ea11_fresh"], r["ea11_transition"]
            acc["eps"] += r["landscape_fresh"]["eps"]
            acc["auc"] += fr["auc"]
            acc["c90"] += np.log(censored(fr["hit90"], b) + 1).tolist()
            acc["crel"] += np.log(censored(fr["hit_rel"], b) + 1).tolist()
            acc["eps_t"] += r["landscape_transition"]["eps"]
            acc["auc_t"] += tr["auc"]
            acc["k"] += np.log(np.array(tr["k_essential"]) + 1).tolist()
            acc["c90_t"] += np.log(censored(tr["hit90"], b) + 1).tolist()
            acc["crel_t"] += np.log(censored(tr["hit_rel"], b) + 1).tolist()
    a = {k: np.array(v, dtype=float) for k, v in acc.items()}
    out = {}

    def corr(name, x, y):
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() > 5 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
            rho, p = spearmanr(x[ok], y[ok])
            out[name] = (float(rho), float(p))

    corr("eps vs log evals-to-0.90 (fresh)", a["eps"], a["c90"])
    corr("eps vs log evals-to-95%-of-gain (fresh)", a["eps"], a["crel"])
    corr("eps vs AUC (fresh; expect negative)", a["eps"], a["auc"])
    corr("eps vs AUC (transition; expect negative)", a["eps_t"], a["auc_t"])
    corr("log k vs log evals-to-0.90 (transition)", a["k"], a["c90_t"])
    corr("log k vs log evals-to-95%-of-gain (trans.)", a["k"], a["crel_t"])
    return out


def make_plots(all_res, rows, cfg, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    reps = list(cfg.reps)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    b = cfg.budget
    # 1. adaptation cost per representation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, key, title in ((axes[0], "ea11_fresh", "fresh held-out tasks"),
                           (axes[1], "ea11_transition", "related task transitions")):
        data = [np.concatenate([censored(res[r][key]["hit90"], b) for res in all_res.values()]) for r in reps]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(reps) + 1)); ax.set_xticklabels(reps, rotation=30, ha="right")
        ax.set_yscale("log"); ax.axhline(b + 1, color="grey", ls=":", lw=1)
        ax.set_ylabel("reward evals to 0.90 (censored at budget+1)")
        ax.set_title(f"(1+1) EA, {title}")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "adaptation_cost.png"), dpi=140); plt.close(fig)
    # 2. RQ1 scatter: epistasis vs adaptation cost
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ci, r in enumerate(reps):
        e = np.concatenate([np.array(res[r]["landscape_fresh"]["eps"]) for res in all_res.values()])
        c = np.concatenate([np.array(res[r]["ea11_fresh"]["auc"]) for res in all_res.values()])
        kk = np.concatenate([np.array(res[r]["ea11_transition"]["k_essential"]) for res in all_res.values()])
        ct = np.concatenate([censored(res[r]["ea11_transition"]["hit_rel"], b) for res in all_res.values()])
        axes[0].scatter(e, c, s=12, alpha=0.6, color=colors[ci % 10], label=r)
        axes[1].scatter(kk + 1, ct, s=12, alpha=0.6, color=colors[ci % 10], label=r)
    axes[0].set_xlabel("relative epistasis eps (fresh task)"); axes[0].set_ylabel("AUC of best reward (higher = faster)")
    axes[1].set_xlabel("essential genome changes k (+1)"); axes[1].set_ylabel("evals to 95% of own gain (transition)")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    xs = np.logspace(0, 2, 50)
    axes[1].plot(xs, 2 * math.e * cfg.n * np.array([sum(1 / i for i in range(1, int(max(1, x)) + 1)) for x in xs]),
                 color="k", lw=1, ls="--", label="2enH_k (additive bound)")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "rq1_landscape_vs_cost.png"), dpi=140); plt.close(fig)
    # 3. capacity vs evolvability
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for ci, row in enumerate(rows):
        ax.errorbar(row["ea11_fresh_test"], row["ea11_fresh_auc"], xerr=row.get("ea11_fresh_test_sem", 0),
                    yerr=row.get("ea11_fresh_auc_sem", 0), fmt="o", color=colors[ci % 10], label=row["rep"])
    ax.set_xlabel("final test accuracy of (1+1) EA genome (capacity reached)")
    ax.set_ylabel("AUC of best reward over budget (speed)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "capacity_vs_speed.png"), dpi=140); plt.close(fig)


def write_summary(rows, corr, cfg, outdir, log=print):
    keys = [k for k in rows[0].keys() if k not in ("rep", "n_seeds") and not k.endswith("_sem")]
    with open(os.path.join(outdir, "summary.csv"), "w") as fh:
        fh.write(",".join(["rep", "n_seeds"] + keys + [k + "_sem" for k in keys]) + "\n")
        for r in rows:
            fh.write(",".join([r["rep"], str(r["n_seeds"])] + [f"{r[k]:.5g}" for k in keys]
                              + [f"{r[k + '_sem']:.5g}" for k in keys]) + "\n")

    def table(title, spec):
        out = [title, f"{'rep':14s}" + "".join(f"{h:>9s}" for h, _, _ in spec)]
        for r in rows:
            out.append(f"{r['rep']:14s}" + "".join(
                (f"{r[k]:9{fmt}}" if k in r and not (isinstance(r[k], float) and math.isnan(r[k])) else f"{'-':>9s}")
                for _, k, fmt in spec))
        return out

    lines = [f"Stage 1 summary: budget={cfg.budget} reward evaluations, n={cfg.n}, "
             f"seeds={rows[0]['n_seeds']} (means over seeds; medians censored at budget+1)", ""]
    lines += table("A. Representation and landscape", [
        ("capLR", "capacity_lr", ".3f"), ("decod", "decodability", ".3f"), ("eps", "eps_fresh", ".3f"),
        ("epsTr", "eps_trans", ".3f"), ("kFresh", "k_ess_fresh", ".1f"), ("kTrans", "k_ess_trans", ".1f"),
        ("coact", "coact", ".3f"), ("band", "band", ".2f"), ("active", "active", ".1f"), ("dead", "dead", ".2f"),
        ("lemmaV", "lemma_viol", ".0f")])
    lines.append("")
    lines += table("B. Fresh held-out tasks, reward-only adaptation from scratch", [
        ("11s90", "ea11_fresh_succ90", ".2f"), ("11m90", "ea11_fresh_med90", ".0f"),
        ("11s95", "ea11_fresh_succ", ".2f"), ("11rel", "ea11_fresh_relmed", ".0f"),
        ("11auc", "ea11_fresh_auc", ".3f"), ("11test", "ea11_fresh_test", ".3f"),
        ("1Ls95", "ea1l_fresh_succ", ".2f"), ("1Ltest", "ea1l_fresh_test", ".3f"),
        ("CMAs90", "cma_fresh_succ90", ".2f"), ("CMAm90", "cma_fresh_med90", ".0f"),
        ("CMAauc", "cma_fresh_auc", ".3f"), ("CMAtest", "cma_fresh_test", ".3f")])
    lines.append("")
    lines += table("C. Related-task transitions, start from the previous task's solution", [
        ("11s90", "ea11_transition_succ90", ".2f"), ("11m90", "ea11_transition_med90", ".0f"),
        ("11rel", "ea11_transition_relmed", ".0f"), ("11auc", "ea11_transition_auc", ".3f"),
        ("11test", "ea11_transition_test", ".3f"),
        ("CMAs90", "cma_transition_succ90", ".2f"), ("CMAm90", "cma_transition_med90", ".0f"),
        ("CMAauc", "cma_transition_auc", ".3f"), ("CMAtest", "cma_transition_test", ".3f")])
    lines.append("")
    lines.append("RQ1 Spearman correlations (pooled over seeds and non-oracle representations, per task):")
    for k, (rho, p) in corr.items():
        lines.append(f"  {k:46s} rho={rho:+.3f}  p={p:.2g}")
    lines += ["",
              "Legend: capLR = test acc of a logistic head trained WITH labels (capacity reference, not reward-only);",
              "decod = mean linear decodability of the true factors; eps = relative epistasis along the (1+1) EA",
              "trajectory; kFresh/kTrans = genome positions still needed after pruning neutral drift; coact = mean",
              "off-diagonal P[f_i=f_j=1]; band = P(|m|<4) at the found solution; s90/s95 = fraction of runs reaching",
              "reward 0.90/0.95; m90 = median evals to 0.90; rel = median evals to realise 95% of the run's own",
              "improvement; auc = mean best-so-far reward over the budget; test = held-out accuracy of the final",
              "solution. 11 = (1+1) EA on the ternary genome, 1L = (1+16) EA, CMA = CMA-ES on a continuous head",
              "over the SAME features. ideal = true factors (oracle), random = untrained encoder (floor).",
              "RQ2: compare eps/k/coact of eil* with multitask/autoencoder at similar capLR/test.",
              "RQ3: compare Table B/C speed columns across representations, and EA vs CMA on each representation."]
    txt = "\n".join(lines)
    with open(os.path.join(outdir, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")
    log("\n" + txt)


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", message=".*constant.*")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="quick", choices=["smoke", "quick", "full"])
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--out", default="results_stage1")
    ap.add_argument("--sanity-only", action="store_true")
    ap.add_argument("--no-cma", action="store_true")
    ap.add_argument("--reps", nargs="+", default=None, help="subset of representations")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config fields, e.g. --set train_frac=0.2 budget=5000 lam_co=1.0")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    cfg = make_config(args.mode)
    if args.no_cma:
        cfg.run_cma = False
    if args.reps:
        cfg.reps = tuple(args.reps)
    for kv in args.set:
        key, val = kv.split("=", 1)
        if not hasattr(cfg, key):
            raise SystemExit(f"unknown config field {key}")
        cur = getattr(cfg, key)
        setattr(cfg, key, type(cur)(val) if not isinstance(cur, bool) else val.lower() in ("1", "true", "yes"))
    seeds = args.seeds if args.seeds is not None else ([0, 1] if args.mode != "full" else [0, 1, 2, 3, 4])
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    log(f"config ({args.mode}): {json.dumps(to_jsonable(asdict(cfg)))}")
    passed = sanity_checks(cfg, log)
    if args.sanity_only:
        return
    if not passed:
        log("Sanity checks failed -- continuing, but results are NOT trustworthy.")
    all_res = {}
    t0 = time.time()
    for seed in seeds:
        all_res[seed] = run_seed(cfg, seed, log)
        rows = aggregate(all_res, cfg)
        corr = rq1_correlations(all_res, cfg)
        with open(os.path.join(args.out, "results.json"), "w") as fh:
            json.dump(to_jsonable(dict(config=asdict(cfg), sanity_passed=passed, seeds=all_res,
                                       summary=rows, rq1=corr)), fh)
        write_summary(rows, corr, cfg, args.out, log)
        make_plots(all_res, rows, cfg, args.out)
        log(f"[elapsed {time.time() - t0:.0f}s after seed {seed}]")
    log(f"done. results in {args.out}/")


if __name__ == "__main__":
    main()
