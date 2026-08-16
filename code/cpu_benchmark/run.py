"""CPU benchmark — all ES arms on the countdown task, one (arm, seed) per invocation.

Designed for SLURM array jobs on Euler: each job runs one arm at one seed, writes
one JSON file. `aggregate.py` collects them into a comparison table.

Arms:
    0  iid + fixed sigma (swept)      baseline, no adaptation
    1  iid + resolution rule          self-adapting baseline
    2  partitioned + resolution       E8-validated: structural effect on pass@16
    3  partitioned + selective update  DiPEC-inspired: utility-gated accumulation
    4  novelty + resolution            E7 reference: directionally positive, not established
    5  orthogonal + resolution         untested, carries a lower-MSE theorem

Usage:
    python code/cpu_benchmark/run.py --arm 2 --seed 0 --output results/
    python code/cpu_benchmark/run.py --arm all --seed 0   # run all arms sequentially
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Ensure the code/ directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from evolve.core.trainer import TrainConfig, Trainer
from evolve.objectives.countdown_obj import CountdownObjective
from evolve.operators import sampling, sigma, weighting

# ---------------------------------------------------------------- configuration

N_POP = 128
RANK = 1
SIGMA = 0.005
ALPHA = 0.0005      # halved from 0.001 — smoke test showed both arms degrading
GENERATIONS = 300
N_PROBLEMS = 96
EVAL_KS = (1, 4, 16)
EVAL_EVERY = 25     # finer checkpoints to see peaks


# ------------------------------------------------------------- selective update

class SelectiveObjective:
    """Wraps an Objective to gate updates by per-part utility.

    After each generation the trainer accumulates a delta dict. Before applying,
    this wrapper checks which parts had above-median utility and zeros out the
    rest. The effect: parts of the model that are not useful for the target task
    stay frozen, reducing cumulative parameter drift.

    This is the DiPEC-inspired idea adapted for ES: NOT (2+1), NOT population
    subsampling (E4 showed that fails). Instead, proper population (N=128) with
    partitioned sampling, and the selection acts on *which parameters to update*,
    not on which members to keep.

    On the 94k-param countdown model the partition is just {emb, h, out} (P=3),
    which is coarse. The point is to test the mechanism, not to produce a
    definitive result. The GPU benchmark tests it at P=7 on a real model.
    """

    def __init__(self, inner: CountdownObjective, threshold: str = "median"):
        self.inner = inner
        self.threshold = threshold
        self._part_weights: dict[str, float] = {}
        self._sampler = None

    @property
    def shapes(self):
        return self.inner.shapes

    def evaluate(self, req):
        return self.inner.evaluate(req)

    def parent_fitness(self, req):
        return self.inner.parent_fitness(req)

    def apply_update(self, delta):
        if not self._part_weights:
            self.inner.apply_update(delta)
            return

        # Gate: only apply update to parts above threshold
        utilities = self._part_weights
        if not utilities:
            self.inner.apply_update(delta)
            return

        vals = list(utilities.values())
        if self.threshold == "median":
            import statistics
            thresh = statistics.median(vals) if vals else 0.0
        else:
            thresh = float(self.threshold)

        filtered = {}
        for name, d in delta.items():
            if utilities.get(name, 0.0) >= thresh:
                filtered[name] = d
        self.inner.apply_update(filtered)

    def set_part_weights(self, weights: dict[str, float]):
        """Called between pass 1 and pass 2 of the trainer loop."""
        self._part_weights = weights

    # Forward everything else
    def __getattr__(self, name):
        return getattr(self.inner, name)


class SelectiveTrainer(Trainer):
    """Trainer subclass that computes per-part utility and gates the update.

    Overrides step() to insert the utility computation between pass 1 and pass 2.
    The objective must be a SelectiveObjective.
    """

    def step(self, generation):
        from evolve.core import noise
        from evolve.core.parallel import plan_chunks

        t0 = time.time()
        cfg = self.cfg
        gen_seed = noise.chunk_seed(cfg.seed * 1_000_003, generation)
        crn_seed = noise.chunk_seed(gen_seed, 0xC12)

        specs = plan_chunks(cfg.n_pop, cfg.chunk_size, gen_seed, crn_seed,
                            self._sigma, cfg.rank, generation)

        # ---- pass 1: evaluate -----------------------------------------------
        def _eval(spec):
            return self.obj.evaluate(self._request(spec, self._draw(spec)))

        evals = self.executor.map(_eval, specs)
        fitness = torch.cat([e.fitness for e in evals])
        per_example = torch.cat([e.per_example for e in evals])

        # ---- sigma adaptation ------------------------------------------------
        if self.sigma_rule is not None:
            parent = self.obj.parent_fitness(
                self._request(specs[0], self._draw(specs[0])))
            self._sigma = self.sigma_rule.update(fitness, parent)

        # ---- weights over the WHOLE population --------------------------------
        w = self.weighting.weights(fitness, per_example, self.state)

        # ---- per-part utility (the selective mechanism) -----------------------
        mask = getattr(self.sampler, '_last_mask', None)
        if mask is not None and hasattr(self.obj, 'set_part_weights'):
            part_names = list(self.obj.shapes.keys())
            utilities = {}
            for p_idx, pname in enumerate(part_names):
                sel = mask[:, p_idx] > 0
                if sel.any():
                    utilities[pname] = float(w[sel].abs().mean())
                else:
                    utilities[pname] = 0.0
            self.obj.set_part_weights(utilities)

        # ---- pass 2: accumulate the update -----------------------------------
        delta: dict[str, torch.Tensor] = {}
        scale = cfg.alpha / (cfg.n_pop * self._sigma)
        for spec in specs:
            factors = self._draw(spec)
            wc = w[spec.lo:spec.hi]
            imp = self.sampler.importance(factors, cfg.n_pop)
            if imp is not None:
                wc = wc * imp
            noise.accumulate_update(factors, wc, cfg.rank, into=delta)
        self.obj.apply_update({k: v * scale for k, v in delta.items()})

        # ---- diagnostics (read-only) -----------------------------------------
        diag = {}
        for name, d in self.diagnostics.items():
            try:
                diag[name] = d.observe(generation, fitness, per_example,
                                       self._sigma, self.obj)
            except Exception as exc:
                diag[name] = {"error": repr(exc)}

        from evolve.core.trainer import GenerationRecord
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


# ----------------------------------------------------------------- arm builders

def build_arm(arm_idx, seed):
    """Returns (name, objective, sampler, weighting, sigma_rule, trainer_cls)."""

    arms = {
        0: "iid_fixed_swept",
        1: "iid_resolution",
        2: "partitioned_resolution",
        3: "partitioned_selective",
        4: "novelty_resolution",
        5: "orthogonal_resolution",
    }
    name = arms[arm_idx]

    if name == "partitioned_selective":
        obj = SelectiveObjective(
            CountdownObjective(n_problems=N_PROBLEMS, seed=seed))
    else:
        obj = CountdownObjective(n_problems=N_PROBLEMS, seed=seed)

    if name.startswith("partitioned"):
        smp = sampling.PartitionedSampler(n_active=1)
    elif name.startswith("orthogonal"):
        smp = sampling.OrthogonalSampler()
    else:
        smp = sampling.IIDSampler()

    if name.startswith("novelty"):
        wgt = weighting.NoveltyBonus(lam=0.2)
    else:
        wgt = weighting.RankWeighting()

    if "fixed" in name:
        sig = sigma.FixedSigma(SIGMA)
    else:
        sig = sigma.ResolutionRule(SIGMA)

    trainer_cls = SelectiveTrainer if name == "partitioned_selective" else Trainer

    return name, obj, smp, wgt, sig, trainer_cls


# -------------------------------------------------------------------- run logic

def run_one(arm_idx, seed, output_dir):
    name, obj, smp, wgt, sig, trainer_cls = build_arm(arm_idx, seed)

    cfg = TrainConfig(
        n_pop=N_POP, rank=RANK, sigma=SIGMA, alpha=ALPHA,
        generations=GENERATIONS, seed=seed,
    )

    tr = trainer_cls(obj, smp, wgt, cfg, sigma_rule=sig)

    # Cosine alpha decay: full alpha at gen 0, 10% of alpha at final gen
    alpha_min = ALPHA * 0.1

    # Checkpoints: evaluate pass@k at regular intervals
    checkpoints = []

    def on_gen(rec):
        gen = rec.generation
        # Cosine decay on alpha
        progress = gen / max(GENERATIONS - 1, 1)
        tr.cfg.alpha = alpha_min + 0.5 * (ALPHA - alpha_min) * (1 + math.cos(math.pi * progress))

        if gen % EVAL_EVERY == 0 or gen == GENERATIONS - 1:
            rep = obj.report(ks=EVAL_KS, seed=seed) if not isinstance(obj, SelectiveObjective) \
                else obj.inner.report(ks=EVAL_KS, seed=seed)
            cp = {"gen": gen, "sigma": rec.sigma, "alpha": tr.cfg.alpha}
            cp.update(rep)
            checkpoints.append(cp)

    t0 = time.time()
    tr.run(callback=on_gen)
    wall_seconds = time.time() - t0
    tr.close()

    # Final evaluation
    final_obj = obj.inner if isinstance(obj, SelectiveObjective) else obj
    final = final_obj.report(ks=EVAL_KS, seed=seed)

    # History (compact: just fitness and sigma per generation)
    history = [{"gen": r.generation, "fitness_mean": r.fitness_mean,
                "fitness_max": r.fitness_max, "sigma": r.sigma}
               for r in tr.history]

    result = {
        "arm": name,
        "arm_idx": arm_idx,
        "seed": seed,
        "config": {
            "n_pop": N_POP, "rank": RANK, "sigma_init": SIGMA,
            "alpha": ALPHA, "generations": GENERATIONS,
            "n_problems": N_PROBLEMS,
        },
        "checkpoints": checkpoints,
        "final": final,
        "history": history,
        "wall_seconds": wall_seconds,
    }

    # Write JSON
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fname = out_path / f"{name}_seed{seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {name} seed={seed}  {wall_seconds:.1f}s  "
          f"pass@1={final['pass@1']:.4f}  pass@16={final['pass@16']:.4f}  -> {fname}")
    return result


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(
        description="CPU benchmark: ES arms on countdown (94k-param policy)")
    parser.add_argument("--arm", default="all",
                        help="Arm index (0-5) or 'all'")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="code/cpu_benchmark/results",
                        help="Output directory for JSON results")
    args = parser.parse_args()

    if args.arm == "all":
        arms = list(range(6))
    else:
        arms = [int(args.arm)]

    print(f"CPU benchmark  N={N_POP}  gens={GENERATIONS}  seed={args.seed}")
    print(f"Arms: {arms}\n")

    for arm_idx in arms:
        run_one(arm_idx, args.seed, args.output)


if __name__ == "__main__":
    main()
