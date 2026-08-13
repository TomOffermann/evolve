# DEGA / DiPEC — Diversity-Preserving Exploitation of Crossover

**Paper:** *Diversity-Preserving Exploitation of Crossover*, arXiv:2507.01524 (FOGA 2025 /
Theoretical Computer Science). **Status:** Settled as the conceptual template we want to port.

## The tension it solves

Crossover only helps if the population is diverse. But exploiting crossover — taking the fitter
offspring — *destroys* diversity, because the offspring sits roughly halfway between its parents.
Classic antagonism. Most GAs paper over it with mutation rate tuning or explicit niching.

## The mechanism

`(2+1)-DEGA` keeps a population of exactly two points `P = {x¹, x²}` with `LO(x¹) ≤ LO(x²)`,
and alternates two phases:

**Diversity phase** (`f(x¹) = f(x²)`): mutate at rate `1/n`. Among the three candidate points,
keep **the pair with the largest Hamming distance**. Fitness ties are broken purely by diversity.

**Exploitation phase** (`f(x¹) < f(x²)`): here's the trick. Uniform crossover producing a fitter
child `y` gives an improving mask `m = x¹ ⊕ y` that mixes *critical* bit flips with a pile of
irrelevant ones. Taking all of `m` halves the diversity. Instead, DiPEC **subsamples the mask**:
keep each one-bit of `m` independently with probability `1/λ`. Concretely, the biased operator
`Crossover(x¹, x², 1/λ)` takes each bit from `x²` with probability `1/λ`, otherwise from `x¹`,
and is resampled until it produces a strict improvement:

```
repeat  y = Crossover(x¹, x², 1/λ)  until  f(y) > f(x¹)
x¹ ← y
```

Expected diversity retained:

```
E[H(x², y')] = (1 − 1/(2λ)) · H(x¹, x²)
```

i.e. you lose a `1/(2λ)` fraction of the distance instead of the usual **half**. You buy fitness
in small, cheap, targeted increments and keep the population spread out.

## Why it's not just a heuristic

**Theorem 4.1:** for `2 ≤ λ = o(n)` on LeadingOnes,
`E[T] = O(λn + n² log n / √λ)`; at `λ = (n log n)^{2/3}` this gives

```
E[T] = O(n^{5/3} log^{2/3} n)
```

versus `Θ(n²)` for standard GAs. Breaking the `n²` barrier on LeadingOnes with a *natural*
algorithm — previously only artificial, hand-tailored algorithms managed it.

## What we actually want from it

Three transferable ideas, in decreasing order of how obviously they port:

1. **Phase structure.** Alternate "spend the budget on fitness" and "spend the budget on
   spreading out", switched by whether the population is currently tied. Ports cleanly.
2. **Partial-step exploitation.** Don't take the whole improving direction; take a random
   `1/λ` subsample of it. **Does not port** — measured, twice.

   [E1](../experiments/E1-results.md) reported it as catastrophic at a single `λ = 4`, which was
   not a fair test. [E4](../experiments/E4-results.md) swept `λ ∈ {1.25, 1.5, 2, 4, 8}` on the
   sparse-reward benchmark: **monotone degradation, no sweet spot.** pass@1 falls 0.122 → 0.117 →
   0.113 → 0.095 → 0.079 as λ rises, and `λ → 1` recovers the baseline exactly as it must.

   The reason is structural. In DEGA the improving mask has *meaningful components*, so dropping
   some of them keeps a valid, smaller improvement. In ES the population **is** a Monte-Carlo
   gradient estimate, so subsampling it multiplies estimator variance by roughly λ and buys
   nothing. Same operator, different kind of object.
3. **Max-distance tie-breaking.** Needs a distance. Hamming on a bitstring is meaningful because
   the genotype is short and one bit flip is one meaningful change. **This is the part that does
   not port**, and it's the whole of [P1](../problems/P1-continuous-diversity.md).

## Caveat we should keep honest about

DEGA's guarantees are for `(2+1)` on LeadingOnes — a two-member population on a unimodal
pseudo-Boolean benchmark. Nothing about the theory survives contact with `N = 10^5` Gaussian
perturbations of a transformer. We are borrowing the *mechanism design*, not the theorem.
