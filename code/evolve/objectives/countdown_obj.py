"""Countdown post-training as an `Objective`. Sparse verifiable reward.

The reference benchmark: 94k-parameter policy, binary reward, most problems admit
several correct answers so pass@k is meaningful, and it **reproduces RLVR diversity
collapse** (pass@1 up, pass@k down). See research/experiments/E2-results.md.

Note how `crn_seed` is used. Every member samples its tokens from the *same* uniform
draws, so fitness differences reflect weight differences rather than sampling noise.
Without this the ES gradient is estimated almost entirely from noise and training
runs backwards -- not a subtle effect, it was the difference between learning and
actively unlearning.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.noise import chunk_seed
from ..core.types import EvalRequest, Evaluation
from ..es.policy import TinyPolicy, pass_at_k, rewards_for
from ..es.sft import sft
from ..tasks.countdown import Countdown


class CountdownObjective:
    def __init__(self, n_problems=96, device="cpu", seed=0, sft_steps=1200,
                 emb=32, hidden=192, temperature=1.0):
        self.task = Countdown()
        self.policy = TinyPolicy(self.task.vocab_size, self.task.ctx, emb=emb,
                                 hidden=hidden, device=device, seed=seed)
        self.device, self.temperature = device, temperature
        self.n_problems = n_problems
        self.base_seed = seed
        self._gen_cache: tuple[int, list, torch.Tensor] | None = None

        if sft_steps:
            X, Y = self.task.sft_data(4000, seed)
            sft(self.policy, torch.from_numpy(X), torch.from_numpy(Y),
                steps=sft_steps, seed=seed)

        self.eval_problems = self.task.problems(256, 99_999)
        self.eval_ctx = torch.from_numpy(self.task.context(self.eval_problems))

    @property
    def shapes(self):
        return self.policy.shapes

    def n_params(self):
        return self.policy.n_params()

    # ------------------------------------------------------------------ internals

    def _problems_for(self, generation: int):
        """One problem set per generation, derived **statelessly** from
        (base_seed, generation).

        The obvious implementation -- draw from a stateful RNG on cache miss -- is a
        race under the ThreadExecutor: two chunks can miss simultaneously, both
        advance the RNG, and end up evaluating *different problem sets*. Their
        fitnesses are then not comparable, and the ES update is computed across
        populations that were never scored on the same task. Measured cost of that
        bug: pass@1 0.117 -> 0.020.

        Deriving from the generation number removes the state entirely, so
        concurrent misses compute the same thing and the cache is a pure
        optimisation."""
        cached = self._gen_cache
        if cached is not None and cached[0] == generation:
            return cached[1], cached[2]
        P = self.task.problems(self.n_problems,
                               chunk_seed(self.base_seed, generation))
        C = torch.from_numpy(self.task.context(P)).to(self.device)
        self._gen_cache = (generation, P, C)     # benign race: same value either way
        return P, C

    # ------------------------------------------------------------------- protocol

    def evaluate(self, req: EvalRequest) -> Evaluation:
        P, C = self._problems_for(req.generation)
        g = torch.Generator().manual_seed(req.crn_seed)      # common random numbers
        out = self.policy.rollout(C, self.task.out_len, fac=req.factors,
                                  sigma=req.sigma, rank=req.rank,
                                  temperature=self.temperature, n_pop=req.size,
                                  generator=g, common_randoms=True)
        f = rewards_for(self.task, P, out)                   # (N, B) binary
        return Evaluation(fitness=f.mean(dim=1), per_example=f)

    def parent_fitness(self, req: EvalRequest) -> float:
        P, C = self._problems_for(req.generation)
        g = torch.Generator().manual_seed(req.crn_seed)      # SAME draws -> paired
        out = self.policy.rollout(C, self.task.out_len, n_pop=1, generator=g,
                                  temperature=self.temperature, common_randoms=True)
        return float(rewards_for(self.task, P, out).mean())

    def apply_update(self, delta):
        for name, d in delta.items():
            self.policy.W[name] += d

    # ----------------------------------------------------------------- evaluation

    def report(self, ks=(1, 16), seed=0):
        """pass@k on held-out problems, with INDEPENDENT samples -- pass@k is
        meaningless under common random numbers."""
        g = torch.Generator().manual_seed(seed)
        return {f"pass@{k}": pass_at_k(self.task, self.eval_problems, self.policy,
                                       k, self.eval_ctx, self.temperature, g)[0]
                for k in ks}
