# code/

## The framework

```python
from evolve import Trainer, TrainConfig, default_diagnostics, registry
from evolve.objectives.countdown_obj import CountdownObjective

obj = CountdownObjective(seed=0)
tr = Trainer(
    obj,
    sampler    = registry.build("sampling",  "iid"),
    weighting  = registry.build("weighting", "novelty", lam=0.2),
    config     = TrainConfig(n_pop=128, sigma=0.005, alpha=0.001,
                             generations=250, chunk_size=64, workers=4),
    sigma_rule = registry.build("sigma", "resolution", sigma=0.005),
    diagnostics= default_diagnostics(128),
)
tr.run(); print(obj.report())
```

`python3 -m evolve.registry` prints every operator with its **measured verdict**. Read it before
picking one — E0–E4 established that whether an operator helps depends on the regime, not on the
operator.

### Three seams, deliberately separate

| seam | protocol | operators |
|---|---|---|
| how perturbations are drawn | `Sampler` | iid, antithetic, orthogonal, **partitioned** (Proposal B) |
| fitness → update weight | `Weighting` | rank, novelty, random-control, niche, whitened, dipec |
| step size | `SigmaRule` | fixed, **resolution**, one-fifth, cosine |

A non-uniform sampler **must** return an importance correction (`1/π_i`) or the ES estimator is
silently biased — enforced by the protocol and tested.

### Parallelism, three levels

1. **Within a chunk** — the population is a batch dimension in every einsum. Always on.
2. **Across chunks** — `Executor.map`; also bounds peak memory, which is what allows N ≫ what
   fits at once. `auto_executor` picks threads only when there's enough work to amortise them.
3. **Across processes/nodes** — not implemented, but the wire format is fixed: `ChunkSpec` is a
   tiny picklable record, and workers return only fitness. Exchanging `(seed, fitness)` instead
   of gradients is ES's structural advantage on weak, loosely-coupled hardware (P3).

**Chunking cannot change results.** Member noise is keyed on *global index*, not chunk id, so
`chunk_size` is purely a memory/parallelism knob. Verified by `tests/test_framework.py` — factors
are bit-identical, parameters agree to float32 epsilon (exact equality is impossible: chunking
regroups a float sum).

```bash
python3 code/tests/test_framework.py                  # 22 invariants
python3 code/experiments/e5_framework_regression.py   # reproduces E2/E4
```

---

## Legacy: the E0–E4 harnesses

The hand-rolled code the framework was distilled from. Frozen as the experimental record.

```bash
python3 code/experiments/e2_countdown_posttrain.py   # ⭐ the real benchmark: sparse reward
python3 code/experiments/e3_jepa_pendulum.py         # ⭐ JEPA, where collapse is the failure
python3 code/experiments/e0_metric_sanity.py         # which diversity metric predicts behaviour
python3 code/experiments/e1e_collapse_diagnostic.py  # is there any collapse to fix?
```

## Benchmarks

**Countdown post-training** (`tasks/countdown.py`, 94k-param policy) — sparse binary verifiable
reward, most problems have several correct answers, so **pass@k** is meaningful. Reproduces
RLVR diversity collapse: post-training doubles pass@1 while pass@16 falls.

**JEPA on pendulum** (`tasks/pendulum_jepa.py`, 2.3k params) — dense reward, but representation
collapse is the failure. Anti-collapse terms go straight into the fitness, including a
**non-differentiable** effective-rank bonus.

The old synthetic grammar (`tasks/grammar.py`) is kept only because E0/E1 reference it. It has
dense log-likelihood reward and **cannot exhibit diversity collapse** — that was the flaw in the
original benchmark, and the reason E1's conclusions had to be revisited.

## What ships

Vanilla EGGROLL plus:

- `es/sigma.py::ResolutionRule` — self-adapting σ. Matches a hand-swept σ without the sweep.
  The classical 1/5th rule **inverts** under sparse binary reward; see E2 Finding 3.
- `diagnostics/collapse.py` — `CollapseDetector` (ES population) **and**
  `PolicyDiversityDetector` (the policy's own output distribution). E2 showed these are
  different objects and only the second one collapses. Monitor both.

`diversity/mechanisms.py` holds the benchmarked mechanisms; see
[decision 0003](../claude/decisions/0003-benchmark-rebuilt-mechanisms-reopened.md) for which, if
any, is currently recommended.

Everything else below is still the plan.

## Layout (✅ = exists)

```
code/
  evolve/
    es/
   ✅   noise.py       # seed → (A_i, B_i); factorised <E_i, E_j> without dense deltas
   ✅   model.py       # TinyLM + the fused population forward y_i = xWᵀ + σ(xB_i)A_iᵀ
   ✅   eggroll.py     # ΔW = (α/Nσ) Σ F̃_i A_i B_iᵀ, rank-normalised fitness shaping
        partition.py   # P2/Proposal B: masks, per-part utility, bandit, epistasis matrix
      diversity/
   ✅   metrics.py     # all E0 candidates side by side + double-centring + log-det volume
   ✅   mechanisms.py  # the 8 benchmarked mechanisms (kept as a harness, none recommended)
      tasks/
   ✅   grammar.py     # synthetic multi-mode grammar, optional skewed mode frequencies
        jepa.py        # P3
      diagnostics/
   ✅   collapse.py    # ⭐ the detector that ships. Zero extra forward passes.
  experiments/
   ✅ e0_metric_sanity.py        # which metric predicts behaviour
   ✅ e1_mechanisms.py           # 8 mechanisms head to head
   ✅ e1b_novelty_sweep.py       # the random-bonus control that overturned E1
   ✅ e1c_decisive.py            # mechanisms vs. σ and lr
   ✅ e1d_niche.py               # the selection injection point
   ✅ e1e_collapse_diagnostic.py # is there any collapse at all?
  tests/
  pyproject.toml
```

`code/experiments/<id>.py` matches `research/experiments/<id>`. Keep that correspondence.

## Design constraints (from [research/problems/P1](../research/problems/P1-continuous-diversity.md))

1. **Never materialise a dense per-member delta.** Everything works on `(A_i, B_i)` or on seeds.
   Any API that hands you an `E_i` of shape `(m, n)` is a bug.
2. **Nothing `O(N²)` on the hot path.** `N` goes to 10^6.
3. **Diversity is a pluggable component with three injection points** — sampling, fitness shaping,
   selection. Keep them separate interfaces; the literature conflates them and it costs clarity.
4. **Diagnostics are not optional.** Every run logs the population's diversity spectra, not just
   its fitness. E0 exists because we didn't have this.
5. **Every new diversity metric is tested against a held-out behavioural target before it is
   allowed near selection.** E0's Proposal C had 18× the discriminative spread of the naive
   baseline and zero predictive power — its histogram looked perfect. Keep the E0 harness as a
   regression test; adding a metric means adding a row to it.

## Implementation stance (Open)

Reuse `eggroll-es` (PyPI, JAX) as reference implementation and correctness oracle. Write our own
perturbation/aggregation layer behind a narrow interface so partitioned masks
([P2](../research/problems/P2-structured-perturbation.md)) and int8 kernels for Pascal can be
swapped in without forking. **Check its licence before anything else.**

## Hardware targets

See [research/problems/P3](../research/problems/P3-objectives-and-hardware.md). Short version:
MacBook for correctness → 3090/4090 for real runs → n × 1080 Ti for scale.

**On 1080 Ti: fp32 or int8, never fp16.** Gaming Pascal runs fp16 at 1/64 of fp32; DP4A int8 runs
at ~4×. Getting this wrong looks exactly like the algorithm being slow.
