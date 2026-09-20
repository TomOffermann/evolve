# E12 — The part effect is second-order, and at σ = 0.005 it is unmeasurable

**Run:** 2026-09-19 · countdown, 94k TinyPolicy, post-SFT, seed 0
**Code:** `code/experiments/e12_additivity.py`
**Status:** the gate fired twice before the question could be asked. Once the instrument and the
step size were fixed, it was asked, and answered.

## What this was supposed to test

Whether the additive part model of P2 §2,

```
F(W + E_a + E_b) ≈ F(W + E_a) + F(W + E_b) − F(W)
```

survives at n_active > 1, by fitting the part × example effect matrix
`U = (M̃ᵀM̃ + λI)⁻¹ M̃ᵀ F̃` and asking whether `u_p` measured alone agrees with `u_p`
measured in company.

Two prior things failed first, and the second one is a theorem. Findings 1 and 2 are why the
question could not be asked at the operating point; Finding 3 is the answer once it could.

## Finding 1 — the sampled Tier-0 signature has zero reliability on countdown

Reliability ceiling = correlation between two *independent* evaluations of the **same**
population (double-centred per-example fitness, N = 256):

| σ | 1 rollout | 4 rollouts | 16 rollouts |
|---|---|---|---|
| **0.005** (operating) | −0.000 | −0.006 | +0.000 |
| 0.02 | −0.000 | −0.024 | +0.071 |
| 0.05 | +0.000 | +0.017 | +0.207 |
| 0.15 | +0.023 | +0.167 | +0.478 |

At the σ E8/E9 ran at, **which problems a member solves is pure sampling noise**. Member
differences in the bit vector carry no reproducible information whatsoever.

E0 validated the Tier-0 signature at ρ = 0.76–0.87 — on the *grammar* task, with dense
log-likelihood reward. It does not survive the move to sparse binary reward in its
sampled form. This is rule 3 again, one level down: **an instrument is only as general as
the benchmark that validated it.**

Direct consequence: anything on this benchmark indexed by the sampled per-example vector
is indexing noise. Direction B's archive ("index by the double-centred per-example fitness
vector from Tier-0") is the clearest case.

**Fix, implemented.** `--readout exact` computes the signature instead of sampling it:

```
q_ib = Σ_{s ∈ S_b} Π_t p_i(s_t | ctx_b, s_<t)
```

`S_b` is the full set of correct token sequences for problem b. Countdown's output is
`a op b op c` with (a,b,c) a permutation of the three given numbers — 3!·3·3 = 54
candidates, max 7 of them correct — so it enumerates exactly. Teacher-forced, five
forward passes, **zero sampling noise**: reliability 1.000 by construction.

This is a change of *instrument*, not of objective. Training still optimises the binary
reward. `q_ib` is exactly the quantity the binary reward is one Bernoulli sample of.

## Finding 2 — with a noiseless instrument, the part effect still doesn't replicate

`part_cos` = mean over parts of cos(u_p from population A, u_p from population B), two
independently drawn populations, n_active = 1, exact readout:

| P | N | members/part | σ=0.005 | σ=0.05 | σ=0.2 |
|---|---|---|---|---|---|
| 4 | 256 | 64 | +0.308 | +0.656 | +0.722 |
| 4 | 1024 | 256 | **−0.433** | +0.760 | +0.915 |
| 8 | 256 | 32 | −0.119 | +0.546 | +0.728 |
| 8 | 1024 | 128 | **+0.027** | +0.629 | +0.897 |

Read the two bold cells. Quadrupling N at σ = 0.005 buys **nothing** — it is not a
sample-size problem. At σ = 0.05–0.2 the same estimator replicates at 0.63–0.92 and
improves with N exactly as an estimator should.

The full E12 run at the operating point (P=16, N=256, σ=0.005, 3 repeats) returns
part_cos ≈ 0 and in_regime ≈ 0.04 at every n_active, and the script correctly refuses to
report a ratio.

## Finding 3 — where it *is* measurable, superposition holds at k=2 and decays

Run at σ = 0.2, P = 8, N = 512, exact readout, 2 repeats. `additivity` = part_cos(k) / part_cos(1),
denominator measured on an independent population so it is not 1.0 by construction:

| n_active | part_cos | shuffled control | **additivity** | in_regime | transfer |
|---|---|---|---|---|---|
| 1 | +0.816 | +0.096 | **1.000** | +0.188 | +0.214 |
| 2 | +0.749 | +0.067 | **0.918** | +0.158 | +0.193 |
| 4 | +0.571 | +0.045 | **0.700** | +0.067 | +0.124 |

So the P2 §2 approximation is good to ~8% at k = 2 and has lost ~30% by k = 4. Decoding two
simultaneous perturbations is defensible; four is already paying a real price. The
swarm-multiplication argument survives, but at a much smaller multiplier than "one rollout carries
P hypotheses" suggests.

Also at this operating point, the E0 instrument test one level up returns
**ρ_parts = 0.78** — part-signature distance does predict held-out behavioural decorrelation
between parts, on an independent population and on problems the signatures were never fitted on.
That is the first evidence that a behaviourally-defined partition has something to define itself
on. It is one seed at a σ 40× the training step, and it means nothing until the control below runs.

*(The shuffled control initially read +0.15 to +0.23 at k = 2. That was not a leak — one random
permutation of 8 labels leaves parts fixed often enough to read +0.2 by chance. It is now averaged
over 32 permutations and sits at +0.05 to +0.10.)*

## Why — and this is the part that generalises

Expand the per-example score around W:

```
q_ib = q_b + ⟨g_b , E_i⟩ + ½ E_iᵀ H_b E_i + O(σ³),     E_i = σ · (M_i ⊙ Z_i),  Z ~ N(0,I)
```

Condition on member i having perturbed part p and take the expectation over the direction
`Z`. The **first-order term vanishes**: `E[⟨g_b, E_i⟩ | part p] = 0`, because `Z` is
zero-mean whatever the mask is. The part identity survives only in the curvature term:

```
E[q_ib | part p] − q_b  =  ½ σ² · tr( H_b |_p )           ← the signal, O(σ²)
Var[q_ib | part p]      =  σ² · ‖ g_b |_p ‖²              ← the noise,  O(σ²) in variance, O(σ) in sd
```

So for n members per part:

```
SNR( û_p )  ∝  σ · √n · tr(H_b|_p) / ‖g_b|_p‖
```

**The part-effect matrix U is a curvature object estimated against gradient noise.** Its
signal is O(σ²) and its noise is O(σ) — the ratio is linear in σ. Dropping σ by 40×
(0.2 → 0.005) costs a factor of 40 in SNR, recoverable only by 1600× the members per part.
N = 1024 gives 4×. That is exactly the table above, and it is why more samples do not help.

## What this changes

**It does not touch E8/E9.** Partitioned perturbation works through the first-order
channel — *what gets perturbed*, and how much drift accumulates. That mechanism needs no
estimate of `u_p` at all. The one mechanism that cleared a tuned control is precisely the
one that does not require the quantity that turns out to be unmeasurable. Decision 0006
stands untouched.

**It re-explains the E9 selective-update null.** `claude/context.md` and
`cpu_benchmark/run.py` attribute "selective update adds little" to P = 3 being too coarse
for utility gating to bite. The measurement says otherwise: per-part utilities are O(σ²)
quantities estimated against O(σ) noise at N = 128, σ = 0.005, ~43 members per part. The
gate was switching on noise. Coarseness was not the problem, and this is checkable —
rerun arm 3 at σ = 0.05 and the gate should start biting.

**It predicts Direction D's shape.** Finer partitions cut members per part as 1/P, so
SNR falls as 1/√P. P = 16 scoring below P = 8 (0.231 vs 0.250) is what the scaling law
says should happen, and it is not "overshooting" in any mysterious sense.

**It does not kill the behavioural-partition programme — it prices it.** `u_p` is
estimable at σ ≈ 0.05–0.2 with n ≈ 10² members per part. That is 10–40× the training σ,
so it cannot come from the training population. It needs a **two-timescale design**:

> Train at σ_train = 0.005. Every K generations spend one **probe generation** at
> σ_probe ≈ 0.05–0.2 with N_probe ≈ 1024 and n_active = 1, used only to refit U —
> never to update W. The probe is a measurement, so its bias does not enter the
> estimator; it informs the partition, the merge/split moves and the sampling floor
> for the next K generations.

At N_probe = 1024 a probe costs about 8 training generations at N = 128. At K = 25 that
is ~30% overhead, and it is the price of having any per-part signal at all. Whether the
resulting partition beats a random partition at matched P and matched schedule is the
original question, now askable.

## Caveats, stated plainly

- One seed, gen 0, post-SFT, 94k params, and Finding 3 at reduced settings (B = 64,
  800 SFT steps) — it is a direction, not a number to quote. The curvature-to-gradient ratio
  `tr(H|_p)/‖g|_p‖` may move during training; `--train-gens` exists to check and has
  not been run.
- σ = 0.2 is 40× the operating step. Nothing here shows the partition estimated at
  probe σ is the *right* partition for training σ — only that it is estimable. That is
  the next control, and it is not optional.
- The secondary test (part-signature distance vs held-out behavioural decorrelation)
  initially returned ρ = 0.651 while part_cos was ~0. That was a leak: held-out
  *problems* but the *same* population, so the two signatures shared their within-part
  directions. Fixed in the committed script — it now draws an independent population.
  Treat any ρ_parts number from before that fix as void.

## Files

- `code/experiments/e12_additivity.py` — calibration, exact/binary readouts, the
  part_cos statistic with its replication ceiling, shuffled-label control.
- `e12_probe_part_replication.py` — the P × N × σ sweep that produced Finding 2.

```bash
python code/experiments/e12_additivity.py --calibrate          # instrument reliability first
python code/experiments/e12_additivity.py --readout exact --parts 8 \
    --n-active 1 2 4 --sigmas 0.05 0.2 --n-pop 1024 --repeats 3
```
