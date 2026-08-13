"""Framework invariants. Run: python3 code/tests/test_framework.py

The load-bearing test is `test_chunking_is_exact`. The trainer's two-pass loop
regenerates factors in pass 2 rather than storing them, and chunk boundaries derive
their own seeds -- so if determinism were broken anywhere, updates would silently
differ from the unchunked reference and every large-N result would be wrong in a way
no fitness curve would reveal.
"""

import sys
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evolve.core import noise
from evolve.core.parallel import SerialExecutor, ThreadExecutor, plan_chunks
from evolve.core.trainer import TrainConfig, Trainer
from evolve.core.types import EvalRequest, Evaluation
from evolve.operators import sampling, sigma, weighting

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


class QuadraticObjective:
    """Deterministic, dense, analytic. Fitness peaks at W = target."""

    def __init__(self, m=6, n=5, device="cpu", seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = {"a": torch.zeros(m, n), "b": torch.zeros(n, m)}
        self.target = {k: torch.randn(v.shape, generator=g) for k, v in self.W.items()}
        self.updates = []

    @property
    def shapes(self):
        return {k: tuple(v.shape) for k, v in self.W.items()}

    def _score(self, W):
        return -sum(((W[k] - self.target[k]) ** 2).sum() for k in W)

    def evaluate(self, req: EvalRequest) -> Evaluation:
        N = req.size
        s = req.sigma / req.rank**0.5
        per = []
        for i in range(N):
            W = {k: self.W[k] + s * (req.factors[k][0][i] @ req.factors[k][1][i].T)
                 for k in self.W}
            per.append(self._score(W))
        F = torch.stack(per)
        return Evaluation(fitness=F, per_example=F[:, None].expand(N, 4).contiguous())

    def parent_fitness(self, req):
        return float(self._score(self.W))

    def apply_update(self, delta):
        self.updates.append({k: v.clone() for k, v in delta.items()})
        for k, d in delta.items():
            self.W[k] += d


def run(chunk_size=None, executor=None, gens=6, n_pop=32):
    obj = QuadraticObjective()
    cfg = TrainConfig(n_pop=n_pop, rank=1, sigma=0.05, alpha=0.05,
                      generations=gens, chunk_size=chunk_size, seed=7)
    t = Trainer(obj, sampling.IIDSampler(), weighting.RankWeighting(), cfg,
                executor=executor)
    t.run()
    return obj, t


# --------------------------------------------------------------------------- tests

def test_noise_determinism():
    shapes = {"a": (4, 3)}
    f1 = noise.draw_factors(11, shapes, 5, 1, "cpu")
    f2 = noise.draw_factors(11, shapes, 5, 1, "cpu")
    check("noise is deterministic in seed",
          torch.equal(f1["a"][0], f2["a"][0]) and torch.equal(f1["a"][1], f2["a"][1]))
    f3 = noise.draw_factors(12, shapes, 5, 1, "cpu")
    check("different seeds differ", not torch.equal(f1["a"][0], f3["a"][0]))


def test_chunking_is_exact():
    """Chunking must not change the mathematics.

    Two levels, deliberately:

    * **Factors must be bit-identical.** Member noise is keyed on global index, so
      chunk boundaries cannot change which perturbation a member draws. Any drift
      here is a real bug, and it would be invisible in fitness curves.
    * **Parameters must agree to float32 rounding.** Bit-equality is *not*
      achievable: splitting the update sum into chunks regroups a float addition,
      and float addition is not associative. Measured drift is ~1e-7 relative, which
      is epsilon. Do not "fix" this by forcing exact equality -- that would only be
      satisfiable by summing in one chunk, i.e. by deleting the feature.
    """
    shapes = {"a": (6, 5), "b": (5, 6)}
    full = noise.draw_factors(99, shapes, 32, 1, "cpu")
    piece = noise.draw_factors(99, shapes, 8, 1, "cpu", index_offset=8)
    check("factors are bit-identical across chunk boundaries",
          torch.equal(full["a"][0][8:16], piece["a"][0])
          and torch.equal(full["b"][1][8:16], piece["b"][1]))

    ref, _ = run(chunk_size=None)
    for cs in (4, 8, 13, 32):
        got, _ = run(chunk_size=cs)
        rel = max(float((ref.W[k] - got.W[k]).abs().max())
                  / max(float(ref.W[k].abs().max()), 1e-12) for k in ref.W)
        check(f"chunk_size={cs} matches unchunked to float32 epsilon", rel < 1e-5,
              f"rel={rel:.1e}")


class StatefulObjective(QuadraticObjective):
    """Objective whose per-generation data comes from a stateful RNG.

    This is the shape of a real bug: under the ThreadExecutor two chunks can miss
    the cache simultaneously, both advance the RNG, and end up scored on *different*
    data -- so the ES update is computed across populations that were never
    evaluated on the same task. It cost pass@1 0.117 -> 0.020 in the countdown
    objective, and the simple stateless test objective could never have caught it.

    Here the fix is asserted rather than the bug: data must derive from the
    generation number, so concurrent misses compute the same thing."""

    def __init__(self, stateless=True, **kw):
        super().__init__(**kw)
        self.stateless = stateless
        self._counter = 0
        self._cache = None

    def _data(self, generation):
        if self._cache is not None and self._cache[0] == generation:
            return self._cache[1]
        if self.stateless:
            seed = noise.chunk_seed(1234, generation)
        else:
            self._counter += 1
            seed = self._counter * 7919
        val = torch.randn(3, generator=torch.Generator().manual_seed(seed))
        self._cache = (generation, val)
        return val

    def evaluate(self, req):
        ev = super().evaluate(req)
        shift = self._data(req.generation).sum()
        return Evaluation(fitness=ev.fitness + shift, per_example=ev.per_example)


def _run_stateful(stateless, chunk_size, executor):
    obj = StatefulObjective(stateless=stateless)
    cfg = TrainConfig(n_pop=32, rank=1, sigma=0.05, alpha=0.05, generations=6,
                      chunk_size=chunk_size, seed=7)
    Trainer(obj, sampling.IIDSampler(), weighting.RankWeighting(), cfg,
            executor=executor).run()
    return obj


def test_stateless_data_survives_concurrency():
    ref = _run_stateful(True, None, SerialExecutor())
    ex = ThreadExecutor(4)
    got = _run_stateful(True, 4, ex)
    ex.shutdown()
    rel = max(float((ref.W[k] - got.W[k]).abs().max())
              / max(float(ref.W[k].abs().max()), 1e-12) for k in ref.W)
    check("generation-derived data is chunk/thread invariant", rel < 1e-5,
          f"rel={rel:.1e}")


def test_thread_executor_matches_serial():
    ref, _ = run(chunk_size=8, executor=SerialExecutor())
    ex = ThreadExecutor(4)
    got, _ = run(chunk_size=8, executor=ex)
    ex.shutdown()
    check("ThreadExecutor == SerialExecutor",
          all(torch.allclose(ref.W[k], got.W[k]) for k in ref.W))


def test_optimisation_actually_works():
    obj, t = run(gens=120, n_pop=64)
    first, last = t.history[0].fitness_mean, t.history[-1].fitness_mean
    check("fitness improves on a convex objective", last > first,
          f"({first:.3f} -> {last:.3f})")


def test_tied_ranks_averaged():
    F = torch.tensor([1.0, 1.0, 1.0, 2.0])
    w = weighting.centred_ranks(F)
    check("tied fitness gets equal weight", torch.allclose(w[:3], w[:3].mean().expand(3)),
          f"{w.tolist()}")


def test_double_centring():
    f = torch.randn(6, 9)
    z = weighting.double_centre(f)
    check("double-centred: rows and cols both ~0",
          z.mean(0).abs().max() < 1e-5 and z.mean(1).abs().max() < 1e-5)


def test_per_example_required():
    try:
        Evaluation(fitness=torch.zeros(3), per_example=None)
        check("Evaluation rejects missing per_example", False)
    except ValueError:
        check("Evaluation rejects missing per_example", True)


def test_partitioned_importance():
    s = sampling.PartitionedSampler()
    shapes = {"a": (4, 3), "b": (3, 4), "c": (2, 2)}
    fac = s.draw(5, shapes, 60, 1, "cpu")
    imp = s.importance(fac, 60)
    check("partitioned sampler returns an importance correction", imp is not None)
    check("importance ~ 1 for a uniform 3-part split",
          imp is not None and abs(float(imp.mean()) - 1.0) < 1e-4,
          f"mean={float(imp.mean()):.4f}")
    # each member perturbs exactly one part -> two of three A blocks are zero
    zeros = sum(int((fac[k][0][0].abs().sum() == 0)) for k in shapes)
    check("each member perturbs exactly one part", zeros == 2, f"zero blocks={zeros}")
    ham = s.mask_hamming()
    check("mask Hamming is a valid distance matrix",
          ham is not None and ham.shape == (60, 60) and float(ham.diagonal().max()) == 0.0)


def test_sigma_rules():
    r = sigma.ResolutionRule(0.01, window=3, dead_band=0.02)
    dense = torch.randn(64)
    for _ in range(10):
        out = r.update(dense, float(dense.mean()))
    check("resolution rule holds sigma on dense fitness (dead band)",
          abs(out - 0.01) < 1e-12, f"sigma={out}")

    r2 = sigma.ResolutionRule(0.01, window=3, dead_band=0.02, target_tie=0.5)
    tied = torch.zeros(64)
    for _ in range(10):
        out2 = r2.update(tied, 0.0)
    check("resolution rule grows sigma when everything ties", out2 > 0.01,
          f"sigma={out2:.5f}")


def test_registry():
    from evolve import registry
    w = registry.build("weighting", "novelty", lam=0.3)
    check("registry builds a novelty operator", isinstance(w, weighting.NoveltyBonus))
    try:
        registry.build("sampling", "nope")
        check("registry rejects unknown names", False)
    except KeyError:
        check("registry rejects unknown names", True)


def test_plan_chunks():
    specs = plan_chunks(10, 4, seed=1, crn_seed=2, sigma=0.1, rank=1, generation=0)
    check("chunk plan covers the population exactly",
          [s.size for s in specs] == [4, 4, 2] and specs[-1].hi == 10)
    check("all chunks share one crn_seed", len({s.crn_seed for s in specs}) == 1)


if __name__ == "__main__":
    print("framework tests\n")
    for fn in [test_noise_determinism, test_chunking_is_exact,
               test_stateless_data_survives_concurrency,
               test_thread_executor_matches_serial, test_optimisation_actually_works,
               test_tied_ranks_averaged, test_double_centring,
               test_per_example_required, test_partitioned_importance,
               test_sigma_rules, test_registry, test_plan_chunks]:
        fn()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failures:", ", ".join(FAIL))
        sys.exit(1)
