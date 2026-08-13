"""Countdown -- a minimal LLM post-training task with sparse verifiable reward.

This replaces the dense-log-likelihood grammar toy used in E0/E1, which was the
wrong regime: EGGROLL's actual case is RL-style post-training, where reward is
**sparse, binary and verifiable**, and where diversity collapse (pass@1 up, pass@k
down) is a documented failure. The grammar task could not exhibit that failure, so
it could not test any mechanism meant to prevent it.

Task. Given 3 numbers and a target, emit an expression `a op b op c` evaluating
left-to-right to the target. Numbers 1..9, ops {+,-,*}.

    context:  [n1, n2, n3, TARGET]        (4 tokens)
    output:   [a, op1, b, op2, c]         (5 tokens)
    reward:   1 if the expression uses each number once and hits the target, else 0

Why this task and not sorting or arithmetic: **most problems have several distinct
correct expressions**. That makes pass@k meaningful and gives diversity something
real to preserve. A task with one right answer cannot show diversity collapse.

Targets are emitted as a single token from the enumerated set of achievable values,
so sequences stay short enough for a tiny model on a laptop.

SFT/post-training split, mirroring real practice: the base model is cloned on
syntactically valid expressions paired with a *random* target token. It therefore
learns the output format and to use the given numbers, but not the target semantics.
Post-training has to supply the actual capability.
"""

import itertools

import numpy as np

OPS = ["+", "-", "*"]


def _apply(a, op, b):
    return a + b if op == "+" else (a - b if op == "-" else a * b)


def _eval(a, o1, b, o2, c):
    """Left-to-right, no precedence -- keeps the verifier unambiguous."""
    return _apply(_apply(a, o1, b), o2, c)


class Countdown:
    def __init__(self, n_lo=1, n_hi=9, seed=0):
        self.nums = list(range(n_lo, n_hi + 1))
        # token layout: 0=PAD, 1..9=digits, then ops, then targets
        self.tok_num = {n: n for n in self.nums}
        base = n_hi + 1
        self.tok_op = {o: base + i for i, o in enumerate(OPS)}
        base += len(OPS)

        vals = sorted({_eval(a, o1, b, o2, c)
                       for a, b, c in itertools.permutations(self.nums, 3)
                       for o1 in OPS for o2 in OPS})
        self.tok_target = {v: base + i for i, v in enumerate(vals)}
        self.vocab_size = base + len(vals)
        self.ctx = 9                      # 4 context + 5 emitted
        self.out_len = 5
        self.id_to_op = {v: k for k, v in self.tok_op.items()}
        self.id_to_num = {v: k for k, v in self.tok_num.items()}

    # ------------------------------------------------------------------ problems

    def problems(self, n, seed):
        """Each problem is guaranteed solvable: the target is generated from a
        random valid expression over the sampled numbers."""
        rng = np.random.default_rng(seed)
        P = []
        for _ in range(n):
            trip = rng.choice(self.nums, size=3, replace=False)
            a, b, c = (int(x) for x in rng.permutation(trip))
            o1, o2 = OPS[rng.integers(3)], OPS[rng.integers(3)]
            P.append((sorted(int(x) for x in trip), _eval(a, o1, b, o2, c)))
        return P

    def context(self, problems):
        """(B, 4) int array: three numbers then the target token."""
        return np.array(
            [[self.tok_num[n] for n in nums] + [self.tok_target[t]]
             for nums, t in problems], dtype=np.int64)

    # ------------------------------------------------------------------ verifier

    def reward(self, problems, out):
        """out: (B, 5) token ids. Returns (B,) float32 of 0/1.

        Invalid output -- wrong token type in a slot, or not a permutation of the
        given numbers -- scores 0, exactly as a real verifier would."""
        R = np.zeros(len(problems), dtype=np.float32)
        for i, (nums, target) in enumerate(problems):
            t = out[i]
            a, o1, b, o2, c = t[0], t[1], t[2], t[3], t[4]
            if a not in self.id_to_num or b not in self.id_to_num or c not in self.id_to_num:
                continue
            if o1 not in self.id_to_op or o2 not in self.id_to_op:
                continue
            used = sorted([self.id_to_num[a], self.id_to_num[b], self.id_to_num[c]])
            if used != list(nums):
                continue
            if _eval(self.id_to_num[a], self.id_to_op[o1], self.id_to_num[b],
                     self.id_to_op[o2], self.id_to_num[c]) == target:
                R[i] = 1.0
        return R

    def n_solutions(self, nums, target):
        """How many distinct correct expressions a problem admits. Used to check
        that the task really does reward diversity."""
        return sum(1 for a, b, c in itertools.permutations(nums)
                   for o1 in OPS for o2 in OPS
                   if _eval(a, o1, b, o2, c) == target)

    # ----------------------------------------------------------------- SFT data

    def sft_data(self, n, seed):
        """Syntactically valid expressions paired with a RANDOM target token.

        Teaches format and number usage, deliberately not the target mapping --
        so post-training has something real to do."""
        rng = np.random.default_rng(seed)
        X, Y = [], []
        targets = list(self.tok_target.values())
        for _ in range(n):
            trip = [int(x) for x in rng.choice(self.nums, size=3, replace=False)]
            a, b, c = (int(x) for x in rng.permutation(trip))
            o1, o2 = OPS[rng.integers(3)], OPS[rng.integers(3)]
            ctx = [self.tok_num[n_] for n_ in sorted(trip)] + [int(rng.choice(targets))]
            seq = ctx + [self.tok_num[a], self.tok_op[o1], self.tok_num[b],
                         self.tok_op[o2], self.tok_num[c]]
            for t in range(self.out_len):                     # one sample per step
                X.append(seq[: 4 + t] + [0] * (self.ctx - 4 - t))
                Y.append(seq[4 + t])
        return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)
