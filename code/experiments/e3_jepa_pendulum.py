"""E3 -- can ES train a JEPA without the anti-collapse hacks?

P3's claim: stop-gradients and EMA target encoders exist to stop *gradient descent*
finding the collapse shortcut. With ES the fitness is a black box, so anti-collapse
can be stated directly -- including non-differentiable terms. If the claim holds,
removing the asymmetry machinery and simply saying "predict, and don't collapse"
should work.

Arms (all ES, no stop-gradient, no EMA target anywhere):
    prediction only      -- should collapse. This is the control that proves the
                            task can fail; without it a success means nothing.
    + variance hinge     -- VICReg's variance term in the fitness
    + variance + cov     -- both
    + rank bonus         -- effective rank of the latent batch, added to fitness.
                            NON-DIFFERENTIABLE. This is the term you cannot
                            backprop through, and the reason to use ES at all.

Scored on things a collapsed encoder cannot fake: latent effective rank, and a
linear-probe R^2 for recovering the true pendulum state. Never on the fitness --
a collapsed encoder achieves a perfect prediction loss.

Run:  python code/experiments/e3_jepa_pendulum.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.es import noise
from evolve.es.eggroll import rank_normalise
from evolve.es.sigma import ResolutionRule
from evolve.tasks.pendulum_jepa import JEPA, Pendulum, evaluate, fitness

DEV = "cpu"
N_POP, RANK = 96, 1
SIGMA0, ALPHA = 0.02, 0.005   # swept; alpha=0.02 diverges at this sigma
GENERATIONS = 600
BATCH = 256
SEEDS = [0, 1, 2]


def rank_bonus(z):
    """Effective rank of the latent batch. Non-differentiable by construction --
    it goes through an SVD and an entropy. ES does not care."""
    zc = z - z.mean(dim=1, keepdim=True)
    sv = torch.linalg.svdvals(zc)                       # (N, D)
    p = sv / sv.sum(-1, keepdim=True).clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum(-1))


def run(arm, seed, adapt=True):
    env = Pendulum()
    model = JEPA(device=DEV, seed=seed)
    rule = ResolutionRule(SIGMA0, target_tie=0.3) if adapt else None
    sigma = SIGMA0
    rng = np.random.default_rng(seed)

    O, A, S = env.trajectories(BATCH, 2, seed=seed)
    obs_t = torch.from_numpy(O[:, 0]); obs_t1 = torch.from_numpy(O[:, 1])
    act = torch.from_numpy(A[:, 0])
    Oe, Ae, Se = env.trajectories(512, 1, seed=12345)
    # Probe the TRUE state, not the observation -- the encoder trivially preserves
    # its own input, so probing Oe would report R^2 ~ 1 for every arm including a
    # collapsed one. Note also that lstsq is scale-invariant, so probe R^2 does not
    # see *scale* collapse; latent std does. The two measure different failures and
    # they dissociate, which is worth reporting rather than hiding.
    ev_obs = torch.from_numpy(Oe[:, 0])
    ev_state = torch.from_numpy(Se[:, 0])

    vw = 1.0 if "var" in arm else 0.0
    cw = 0.5 if "cov" in arm else 0.0

    for gen in range(GENERATIONS):
        fac = noise.factors(seed * 100_000 + gen, model.shapes, N_POP, RANK, DEV)
        ot = obs_t.unsqueeze(0).expand(N_POP, -1, -1)
        ot1 = obs_t1.unsqueeze(0).expand(N_POP, -1, -1)
        ac = act.unsqueeze(0).expand(N_POP, -1, -1)

        F, per_ex, z = fitness(model, ot, ot1, ac, fac, sigma, RANK,
                               var_weight=vw, cov_weight=cw)
        if "rank" in arm:
            F = F + 0.3 * rank_bonus(z)

        if rule is not None:
            F0, _, z0 = fitness(model, ot[:1], ot1[:1], ac[:1], None, 0.0, RANK,
                                var_weight=vw, cov_weight=cw)
            if "rank" in arm:
                F0 = F0 + 0.3 * rank_bonus(z0)
            sigma = rule.observe(F, float(F0[0]))

        w = rank_normalise(F)
        scale = ALPHA / (N_POP * sigma * RANK**0.5)
        for nm, (Af, Bf) in fac.items():
            model.W[nm] += scale * torch.einsum("i,imr,ikr->mk", w, Af, Bf)

    er, r2, std = evaluate(model, ev_obs, ev_state)
    return er, r2, std


ARMS = ["prediction only", "var", "var+cov", "var+cov+rank"]


def main():
    m = JEPA()
    print(f"E3  JEPA on pendulum   params={m.n_params()}  latent={m.latent}  "
          f"N={N_POP} gens={GENERATIONS} seeds={len(SEEDS)}")
    print("No stop-gradient, no EMA target encoder in ANY arm.\n")
    print("eff_rank = effective rank of the latent batch (max = latent dim 8)")
    print("probe R² = linear recovery of true pendulum state from the latent")
    print("Neither can be faked by collapsing; the fitness can.\n")

    rows = []
    for arm in ARMS:
        t0 = time.time()
        res = [run(arm, s) for s in SEEDS]
        er = np.array([r[0] for r in res])
        r2 = np.array([r[1] for r in res])
        sd = np.array([r[2] for r in res])
        rows.append((arm, er.mean(), er.std(ddof=1) / len(er) ** 0.5,
                     r2.mean(), r2.std(ddof=1) / len(r2) ** 0.5, sd.mean()))
        print(f"  {arm:18s} {time.time()-t0:5.1f}s  eff_rank {er.mean():.2f}  "
              f"probe R² {r2.mean():.3f}")

    print(f"\n{'arm':18s} {'eff_rank':>9s} {'sem':>6s} {'probe R²':>9s} {'sem':>6s} "
          f"{'latent std':>11s}")
    for arm, e, es, r, rs, sd in rows:
        print(f"{arm:18s} {e:>9.2f} {es:>6.3f} {r:>9.3f} {rs:>6.3f} {sd:>11.4f}")
    print("\nA collapsed encoder shows eff_rank -> 1, probe R² -> 0, latent std -> 0.")


if __name__ == "__main__":
    main()
