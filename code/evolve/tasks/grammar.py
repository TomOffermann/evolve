"""A small synthetic language with several independent sub-skills.

Why synthetic: E0 needs a task where *behavioural decorrelation* between population
members is a meaningful thing to measure. That requires the task to contain several
distinguishable sub-skills, so that two members can plausibly be good at different
things. A generic corpus gives you one blurry skill; this gives you `n_modes`.

The grammar: `n_modes` word families. Each family has its own alphabet subset and its
own fixed-length template. Sequences are `<mode-marker><word><separator>` repeated.
Predicting the next character requires (a) recognising the marker and (b) knowing that
family's template. Those are separable skills.
"""

import numpy as np

SEP = 0  # separator token


class GrammarTask:
    def __init__(self, n_modes=4, word_len=4, alphabet_per_mode=5, seq_len=32, seed=0,
                 skew=None):
        """`skew`: mode sampling probabilities. Pass e.g. [.70,.20,.07,.03] to create
        rare modes. This is what makes diversity mechanisms *testable*: a greedy
        optimiser fits the dominant mode and abandons the rare ones, which is the
        same failure as entropy/diversity collapse in RLVR. On a uniform task with no
        local optima, no diversity mechanism can help and every arm ties."""
        self.n_modes = n_modes
        self.word_len = word_len
        self.seq_len = seq_len
        self.skew = np.ones(n_modes) / n_modes if skew is None else np.array(skew, float)
        self.skew = self.skew / self.skew.sum()
        rng = np.random.default_rng(seed)

        # token layout: 0 = separator, 1..n_modes = mode markers, then per-mode alphabets
        self.marker = {m: 1 + m for m in range(n_modes)}
        base = 1 + n_modes
        self.alphabet = {
            m: list(range(base + m * alphabet_per_mode, base + (m + 1) * alphabet_per_mode))
            for m in range(n_modes)
        }
        self.vocab_size = base + n_modes * alphabet_per_mode

        # each mode gets a small fixed set of legal words -> learnable structure
        self.words = {
            m: [list(rng.choice(self.alphabet[m], size=word_len)) for _ in range(3)]
            for m in range(n_modes)
        }

    def sample(self, n, seed):
        """Returns (n, seq_len) int array, plus the mode of each position's word."""
        rng = np.random.default_rng(seed)
        seqs, modes = [], []
        for _ in range(n):
            s, md = [], []
            while len(s) < self.seq_len + 1:
                m = int(rng.choice(self.n_modes, p=self.skew))
                w = self.words[m][int(rng.integers(len(self.words[m])))]
                chunk = [self.marker[m]] + list(w) + [SEP]
                s.extend(chunk)
                md.extend([m] * len(chunk))
            seqs.append(s[: self.seq_len + 1])
            modes.append(md[: self.seq_len + 1])
        return np.array(seqs, dtype=np.int64), np.array(modes, dtype=np.int64)

    def contexts(self, n, ctx, seed):
        """Flatten into (inputs, targets, mode) triples for next-token prediction."""
        seqs, modes = self.sample(n, seed)
        X, Y, M = [], [], []
        for s, md in zip(seqs, modes):
            for i in range(ctx, len(s)):
                X.append(s[i - ctx : i])
                Y.append(s[i])
                M.append(md[i])
        return (
            np.array(X, dtype=np.int64),
            np.array(Y, dtype=np.int64),
            np.array(M, dtype=np.int64),
        )
