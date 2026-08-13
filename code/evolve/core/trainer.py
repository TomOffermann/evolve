"""The ES loop. Owns the evolutionary logic; never sees a dense parameter delta.

Two-pass structure per generation:

    pass 1   evaluate every chunk           -> fitness (N,), per_example (N, M)
    ----     compute weights over the WHOLE population (novelty, niching, ranks
             all need global context, not per-chunk context)
    pass 2   regenerate the same factors    -> accumulate dW chunk by chunk

Regenerating in pass 2 rather than storing pass 1's factors is what keeps memory at
O(chunk) instead of O(N). It costs one extra RNG draw per member and buys an
unbounded population. This is the payoff of EGGROLL's seed-based design, and the
reason `noise.draw_factors` has a strict determinism contract.

Everything evolutionary is a plugged-in operator. Swapping the sampler, the
weighting or the sigma rule requires no change here -- which is the point, since
E0-E4 showed that which operator helps depends entirely on the regime.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from . import noise
from .parallel import Executor, auto_executor, plan_chunks
from .types import (Diagnostic, EvalRequest, Objective, Sampler, SigmaRule,
                    Weighting)


@dataclass
class TrainConfig:
    n_pop: int = 128
    rank: int = 1
    sigma: float = 0.005
    alpha: float = 0.001
    generations: int = 250
    chunk_size: int | None = None      # None = one chunk; set to bound peak memory
    workers: int = 4
    seed: int = 0
    device: str = "cpu"
    log_every: int = 0                 # 0 = never


@dataclass
class GenerationRecord:
    generation: int
    fitness_mean: float
    fitness_max: float
    fitness_std: float
    sigma: float
    seconds: float
    diagnostics: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, objective: Objective, sampler: Sampler,
                 weighting: Weighting, config: TrainConfig,
                 sigma_rule: SigmaRule | None = None,
                 diagnostics: dict[str, Diagnostic] | None = None,
                 executor: Executor | None = None):
        self.obj = objective
        self.sampler = sampler
        self.weighting = weighting
        self.cfg = config
        self.sigma_rule = sigma_rule
        self.diagnostics = diagnostics or {}
        # Stochastic operators derive their RNG from this, so their noise varies
        # across seeds like everything else. A control whose noise is constant
        # across seeds reports error bars that exclude its own variability.
        self.state: dict = {"run_seed": config.seed}
        self.history: list[GenerationRecord] = []
        self._sigma = config.sigma
        self.executor = executor or auto_executor(
            config.n_pop, config.chunk_size, config.workers)

    # ------------------------------------------------------------------ helpers

    def _request(self, spec, factors) -> EvalRequest:
        return EvalRequest(factors=factors, sigma=self._sigma, rank=self.cfg.rank,
                           crn_seed=spec.crn_seed, generation=spec.generation,
                           chunk_id=spec.chunk_id, n_chunks=spec.n_chunks)

    def _draw(self, spec):
        """Factors for one chunk, keyed on GLOBAL member index (spec.lo), not on
        chunk id -- so chunking cannot change results. Pass 2 regenerates exactly
        what pass 1 evaluated."""
        return self.sampler.draw(spec.seed, self.obj.shapes, spec.size,
                                 self.cfg.rank, self.cfg.device,
                                 index_offset=spec.lo)

    # -------------------------------------------------------------------- steps

    def step(self, generation: int) -> GenerationRecord:
        t0 = time.time()
        cfg = self.cfg
        gen_seed = noise.chunk_seed(cfg.seed * 1_000_003, generation)
        crn_seed = noise.chunk_seed(gen_seed, 0xC12)

        specs = plan_chunks(cfg.n_pop, cfg.chunk_size, gen_seed, crn_seed,
                            self._sigma, cfg.rank, generation)

        # ---- pass 1: evaluate -------------------------------------------------
        def _eval(spec):
            return self.obj.evaluate(self._request(spec, self._draw(spec)))

        evals = self.executor.map(_eval, specs)
        fitness = torch.cat([e.fitness for e in evals])
        per_example = torch.cat([e.per_example for e in evals])

        # ---- sigma adaptation (paired parent -- see types.py) ------------------
        if self.sigma_rule is not None:
            parent = self.obj.parent_fitness(self._request(specs[0], self._draw(specs[0])))
            self._sigma = self.sigma_rule.update(fitness, parent)

        # ---- weights over the WHOLE population --------------------------------
        w = self.weighting.weights(fitness, per_example, self.state)

        # ---- pass 2: accumulate the update ------------------------------------
        delta: dict[str, torch.Tensor] = {}
        scale = cfg.alpha / (cfg.n_pop * self._sigma)
        for spec in specs:
            factors = self._draw(spec)
            wc = w[spec.lo:spec.hi]
            imp = self.sampler.importance(factors, cfg.n_pop)
            if imp is not None:
                wc = wc * imp          # 1/pi_i keeps a non-uniform sampler unbiased
            noise.accumulate_update(factors, wc, cfg.rank, into=delta)
        self.obj.apply_update({k: v * scale for k, v in delta.items()})

        # ---- diagnostics (read-only) ------------------------------------------
        diag = {}
        for name, d in self.diagnostics.items():
            try:
                diag[name] = d.observe(generation, fitness, per_example,
                                       self._sigma, self.obj)
            except Exception as exc:                      # never break training
                diag[name] = {"error": repr(exc)}

        rec = GenerationRecord(
            generation=generation,
            fitness_mean=float(fitness.mean()),
            fitness_max=float(fitness.max()),
            fitness_std=float(fitness.std()),
            sigma=self._sigma,
            seconds=time.time() - t0,
            diagnostics=diag,
        )
        self.history.append(rec)
        return rec

    def run(self, generations: int | None = None, callback=None):
        n = generations if generations is not None else self.cfg.generations
        for gen in range(n):
            rec = self.step(gen)
            if callback is not None:
                callback(rec)
            elif self.cfg.log_every and gen % self.cfg.log_every == 0:
                print(f"gen {rec.generation:5d}  F {rec.fitness_mean:+.4f}  "
                      f"sigma {rec.sigma:.5f}  {rec.seconds:.2f}s")
        return self.history

    def close(self):
        self.executor.shutdown()
