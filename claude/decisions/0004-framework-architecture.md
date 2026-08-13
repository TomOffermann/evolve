# 0004 — Framework architecture: three seams, parallel by construction

**Date:** 2026-08-13 · **Status:** Settled

## Decision

The hand-rolled E0–E4 code is consolidated into `code/evolve/` as a proper framework:
`core/` (protocols, noise, parallel execution, trainer), `operators/` (the EA library),
`objectives/`, `diagnostics/`. The legacy `evolve/es/` and `evolve/tasks/` modules stay as the
experimental record and are still imported by the objectives.

## The three seams

[P1](../../research/problems/P1-continuous-diversity.md) named three injection points for
evolutionary machinery. They are **separate protocols** so a component cannot silently occupy two
at once — which matters because E0–E4 showed they behave completely differently:

| seam | protocol | what E0–E4 found |
|---|---|---|
| how perturbations are drawn | `Sampler` | untested territory; where Proposal B lives |
| fitness → update weight | `Weighting` | every mechanism here failed on dense reward; novelty wins on sparse |
| step size | `SigmaRule` | **dominates everything else** (E1c) |

Each operator's docstring and its registry entry carry its **measured verdict and the regime it
applies to**. `python3 -m evolve.registry` prints the table. This is deliberate: the single most
transferable lesson from this project is that "does this operator help" has no regime-independent
answer, so an operator library that does not carry its evidence is a trap.

## Two API decisions that encode findings

1. **`Evaluation.per_example` is required, not optional.** The Tier-0 signature is the only
   diversity metric that survived E0 (ρ = 0.76–0.87 vs a floor of 0.00) and the objective computes
   it anyway. Mandatory means every diagnostic and every diversity operator works on every
   objective with no per-objective plumbing.
2. **`EvalRequest.crn_seed` carries common random numbers.** Without them, a stochastic
   objective's ES gradient is estimated almost entirely from sampling noise and training runs
   *backwards* (E2). Too important to leave to each objective's discretion.

Non-uniform samplers must return an importance correction (`1/π_i`). The protocol requires it and
a test checks it; the bias is silent and it poisons any bandit built on the utilities.

## Parallelism

Three levels: within a chunk (population is a batch dimension — always on), across chunks
(`Executor.map`, which also bounds peak memory and is what allows N ≫ what fits at once), and
across processes/nodes (not implemented; the wire format is fixed by `ChunkSpec`, a tiny picklable
record, with workers returning only fitness).

**Chunking cannot change results.** Member noise is keyed on *global index*, not chunk id, so
`chunk_size` is purely a memory/parallelism knob.

That invariant is load-bearing and it caught two real bugs during the build:

- The first implementation keyed noise on chunk id, so changing `chunk_size` silently changed
  which perturbation each member drew — a large sharded run could not be verified against a small
  local one, and nothing in the fitness curves would have shown it.
- The countdown objective drew its per-generation problems from a **stateful RNG**. Under the
  thread executor two chunks could miss the cache simultaneously, both advance the RNG, and end up
  scored on *different problem sets* — so the ES update was computed across populations that had
  never been evaluated on the same task. Cost: pass@1 0.117 → 0.020. Fixed by deriving problems
  from the generation number, making concurrent misses compute the same value.

Both are now regression-tested. The second was invisible to a stateless test objective, so the
suite carries a deliberately stateful one.

Note that bit-equality across chunkings is **not** achievable and the test does not demand it:
chunking regroups a float sum and float addition is not associative (~1e-7 relative). Factors are
bit-identical; parameters agree to epsilon.

## Verification

`code/tests/test_framework.py` — 23 invariants.
`code/experiments/e5_framework_regression.py` — reproduces the E2/E4 reference points. A rewrite
that changes the numbers is not a refactor but a new experiment with unknown provenance, so the
framework had to land on them before being used for anything.
