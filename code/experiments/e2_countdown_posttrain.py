"""E2 -- ES post-training on countdown, with sparse verifiable reward.

This is the benchmark E0/E1 should have used. The grammar task had dense
log-likelihood reward and could not exhibit the failure the whole programme is about.
Here reward is sparse and binary, most problems admit several correct answers, and
**pass@k** measures exactly the thing diversity is supposed to protect.

Two questions:

  1. Does ES post-training show the documented RLVR failure -- pass@1 up, pass@k
     down? If yes, the collapse detector should fire, which is the prerequisite
     decision 0002 set for diversity mechanisms being worth anything.
  2. With sigma self-adapted by the 1/5th rule, do mechanisms earn their place?
     E1c's result was partly "the fixed baseline was untuned". Self-adapting sigma
     removes that confound, so this is the fair test.

Run:  python code/experiments/e2_countdown_posttrain.py
"""

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.diagnostics.collapse import CollapseDetector
from evolve.diversity import mechanisms as MECH
from evolve.es import noise
from evolve.es.policy import TinyPolicy, pass_at_k, rewards_for
from evolve.es.sft import sft
from evolve.es.sigma import OneFifthRule, ResolutionRule
from evolve.tasks.countdown import Countdown

DEV = "cpu"
N_POP, RANK = 128, 1
B_PROB = 96
SIGMA0, ALPHA = 0.005, 0.001   # from the sigma/alpha sweep; see E2-results.md
GENERATIONS = 250
SEEDS = [0, 1, 2]
EVAL_K = 16
LOG_EVERY = 50


def build_base(seed):
    """SFT once per seed: format only, no target semantics."""
    task = Countdown()
    pol = TinyPolicy(task.vocab_size, task.ctx, device=DEV, seed=seed)
    X, Y = task.sft_data(4000, seed)
    sft(pol, torch.from_numpy(X), torch.from_numpy(Y), steps=1200, seed=seed)
    return task, pol


def run(label, mech, seed, adapt_sigma="resolution", trace=False):
    task, pol = build_base(seed)
    state, det = {}, CollapseDetector(warmup=30, window=40)  # sparse reward: signature is degenerate early
    rule = ({"onefifth": OneFifthRule, "resolution": ResolutionRule}
            [adapt_sigma](SIGMA0) if adapt_sigma else None)
    sigma = SIGMA0
    g = torch.Generator().manual_seed(7000 + seed)
    prob_rng = np.random.default_rng(seed)

    ev_P = task.problems(256, 99_999)                       # held out, fixed
    ev_C = torch.from_numpy(task.context(ev_P))
    hist = []

    for gen in range(GENERATIONS):
        P = task.problems(B_PROB, int(prob_rng.integers(1 << 30)))
        C = torch.from_numpy(task.context(P))
        fac = noise.factors(seed * 100_000 + gen, pol.shapes, N_POP, RANK, DEV)
        # Paired evaluation: population and parent see the SAME uniform draws, so
        # the 1/5th rule's success test reflects weight differences rather than
        # sampling noise. Unpaired, the success rate is pure noise and sigma
        # random-walks upward until the policy is destroyed.
        gseed = int(torch.randint(1 << 30, (1,), generator=g))
        gp = torch.Generator().manual_seed(gseed)
        out = pol.rollout(C, task.out_len, fac=fac, sigma=sigma, rank=RANK,
                          n_pop=N_POP, generator=gp)
        f = rewards_for(task, P, out)                        # (N, B) binary
        F = f.mean(dim=1)

        det.observe(f, F)
        if rule is not None:
            gp2 = torch.Generator().manual_seed(gseed)
            parent = float(rewards_for(
                task, P, pol.rollout(C, task.out_len, n_pop=1, generator=gp2)).mean())
            sigma = rule.observe(F, parent)

        w = mech(F, f, state=state)
        scale = ALPHA / (N_POP * sigma * RANK**0.5)
        for nm, (A, B) in fac.items():
            pol.W[nm] += scale * torch.einsum("i,imr,ikr->mk", w, A, B)

        if trace and gen % LOG_EVERY == 0:
            p1, _ = pass_at_k(task, ev_P, pol, 1, ev_C, generator=g)
            pk, _ = pass_at_k(task, ev_P, pol, EVAL_K, ev_C, generator=g)
            h = det.history[-1]
            hist.append((gen, p1, pk, sigma, h["eff_rank"], float(F.mean())))

    p1, _ = pass_at_k(task, ev_P, pol, 1, ev_C, generator=g)
    pk, _ = pass_at_k(task, ev_P, pol, EVAL_K, ev_C, generator=g)
    return p1, pk, sigma, det, hist


ARMS = [
    ("fixed σ (swept)",       MECH.baseline,                            False),
    ("classical 1/5-rule",    MECH.baseline,                            "onefifth"),
    ("resolution rule",       MECH.baseline,                            "resolution"),
    ("res + novelty λ=.2",    partial(MECH.novelty_bonus, lam=0.2),     "resolution"),
    ("res + RANDOM λ=.2",     partial(MECH.random_bonus, lam=0.2),      "resolution"),
    ("res + niche k=8",       partial(MECH.niche_balanced, n_niches=8), "resolution"),
]


def main():
    task, pol = build_base(0)
    print(f"E2  countdown post-training   params={pol.n_params()}  "
          f"vocab={task.vocab_size}")
    print(f"N={N_POP} B={B_PROB} gens={GENERATIONS} seeds={len(SEEDS)} k={EVAL_K}\n")

    g = torch.Generator().manual_seed(0)
    ev_P = task.problems(256, 99_999)
    ev_C = torch.from_numpy(task.context(ev_P))
    b1, _ = pass_at_k(task, ev_P, pol, 1, ev_C, generator=g)
    bk, _ = pass_at_k(task, ev_P, pol, EVAL_K, ev_C, generator=g)
    print(f"SFT base model:  pass@1 {b1:.4f}   pass@{EVAL_K} {bk:.4f}\n")

    print("--- trace: resolution rule, seed 0 ---")
    print(f"{'gen':>5s} {'pass@1':>8s} {'pass@k':>8s} {'sigma':>8s} "
          f"{'eff_rank':>9s} {'train_R':>8s}")
    _, _, _, det, hist = run("trace", MECH.baseline, 0, adapt_sigma="resolution", trace=True)
    for gen, p1, pk, sg, er, tr in hist:
        print(f"{gen:>5d} {p1:>8.4f} {pk:>8.4f} {sg:>8.4f} {er:>9.1f} {tr:>8.4f}")
    print(f"detector: {det.summary()}\n")

    rows = []
    for label, mech, adapt in ARMS:
        t0 = time.time()
        res = [run(label, mech, s, adapt_sigma=adapt) for s in SEEDS]
        p1 = np.array([r[0] for r in res])
        pk = np.array([r[1] for r in res])
        fired = sum(r[3].fires() for r in res)
        rows.append((label, p1.mean(), p1.std(ddof=1) / len(p1) ** 0.5,
                     pk.mean(), pk.std(ddof=1) / len(pk) ** 0.5, fired))
        print(f"  {label:22s} {time.time()-t0:5.1f}s  "
              f"pass@1 {p1.mean():.4f}  pass@{EVAL_K} {pk.mean():.4f}")

    print(f"\n{'arm':22s} {'pass@1':>8s} {'sem':>6s} {'pass@k':>8s} {'sem':>6s} "
          f"{'gap':>7s} {'fired':>6s}")
    for label, m1, s1, mk, sk, fired in rows:
        print(f"{label:22s} {m1:>8.4f} {s1:>6.3f} {mk:>8.4f} {sk:>6.3f} "
              f"{mk-m1:>7.4f} {fired:>4d}/{len(SEEDS)}")
    print(f"\nSFT base: pass@1 {b1:.4f}  pass@{EVAL_K} {bk:.4f}  gap {bk-b1:.4f}")
    print("gap = pass@k - pass@1. A shrinking gap IS diversity collapse:")
    print("the policy gets more reliable at one answer and loses the others.")
    print("'fired' = seeds where the collapse detector triggered.")


if __name__ == "__main__":
    main()
