"""The candidate diversity metrics from research/problems/P1, side by side.

Every metric here returns an (N, N) *distance* matrix (higher = more diverse), so
they can be compared against the same target on the same population.

Design rule (C1): nothing in here may materialise a dense per-member delta.
"""

import torch

from ..es.noise import inner_products


def _cos_from_gram(G, eps=1e-12):
    d = torch.sqrt(torch.diagonal(G).clamp_min(eps))
    return G / (d[:, None] * d[None, :])


def _cos_dist(emb, eps=1e-12):
    """Cosine distance between rows of `emb` (N, k)."""
    e = emb / emb.norm(dim=1, keepdim=True).clamp_min(eps)
    return 1.0 - e @ e.T


# ---------------------------------------------------------------- genotype space

def param_cosine(fac):
    """#1: cosine distance between full parameter deltas. The null hypothesis.

    Computed exactly from the factors, summed over weight matrices."""
    G = None
    for A, B in fac.values():
        g = inner_products(A, B)
        G = g if G is None else G + g
    return 1.0 - _cos_from_gram(G)


def layer_energy(fac):
    """#5: per-layer energy profile ||E_i^(l)||. A cheap L-dim 'where did it push'."""
    cols = []
    for A, B in fac.values():
        # ||A B^T||_F^2 = sum((A^T A) * (B^T B))
        AA = torch.einsum("imr,ims->irs", A, A)
        BB = torch.einsum("inr,ins->irs", B, B)
        cols.append((AA * BB).sum(dim=(-1, -2)).sqrt())
    return _cos_dist(torch.stack(cols, dim=1))


def signed_hamming(fac, k=512, seed=0):
    """#4: binarise the delta on a random coordinate subset -> literal Hamming.

    The most direct DEGA transplant. Samples k (row, col) coordinates and takes
    sign(E_i) there; E[row,col] = sum_r A[row,r] B[col,r], no dense delta needed."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    bits = []
    for A, B in fac.values():
        m, n = A.shape[1], B.shape[1]
        ri = torch.randint(m, (k,), generator=g).to(A.device)
        ci = torch.randint(n, (k,), generator=g).to(A.device)
        bits.append(torch.sign((A[:, ri, :] * B[:, ci, :]).sum(-1)))
    b = torch.cat(bits, dim=1)
    N, K = b.shape
    return (K - b @ b.T) / (2.0 * K)  # normalised Hamming distance


# ---------------------------------------------------------- active subspace (C)

class SubspaceSketch:
    """Proposal C: rank-k sketch of recent updates, held per weight matrix."""

    def __init__(self, k=16, history=8):
        self.k, self.history = k, history
        self.buf = {}

    def observe(self, updates):
        for name, dW in updates.items():
            q = self.buf.setdefault(name, [])
            q.append(dW.flatten().clone())
            if len(q) > self.history:
                q.pop(0)

    def basis(self, shapes):
        """{name: (k, m, n)} orthonormal basis matrices, or None if not warm yet."""
        if not self.buf:
            return None
        out = {}
        for name, q in self.buf.items():
            if len(q) < 2:
                return None
            M = torch.stack(q, dim=0)                       # (H, m*n)
            _, _, Vh = torch.linalg.svd(M, full_matrices=False)
            k = min(self.k, Vh.shape[0])
            out[name] = Vh[:k].reshape(k, *shapes[name])
        return out


def active_subspace(fac, basis):
    """#6 / Proposal C: distance between projections onto the active subspace.

    <U^(l), A_i B_i^T> = sum_r (A_i[:,r]^T U^(l) B_i[:,r]) -- factorised, no dense E."""
    if basis is None:
        return None
    cols = []
    for name, (A, B) in fac.items():
        U = basis[name]                                      # (k, m, n)
        UB = torch.einsum("kmn,inr->ikmr", U, B)             # (N, k, m, r)
        cols.append(torch.einsum("imr,ikmr->ik", A, UB))     # (N, k)
    return _cos_dist(torch.cat(cols, dim=1))


# ------------------------------------------------------ function space (Prop. A)

def delta_activation_signature(sig, k=128, seed=0, centre=True):
    """Proposal A Tier 1: sketch of a per-member output delta.

    `sig` is (N, B, V). Pass the *output layer's own* low-rank delta (free under
    C4) or the *total* output delta (needs the base forward, which EGGROLL computes
    anyway). E0 shows these behave very differently -- see the experiment notes."""
    N = sig.shape[0]
    flat = sig.reshape(N, -1)
    if centre:
        flat = flat - flat.mean(0, keepdim=True)  # remove the shared component
    g = torch.Generator(device="cpu").manual_seed(seed)
    S = torch.randn(flat.shape[1], k, generator=g).to(flat.device) / k**0.5
    return _cos_dist(flat @ S)


def fitness_vector_signature(f):
    """Proposal A Tier 0: per-example fitness vector. Free.

    Double-centred: remove per-example difficulty (dim=0) and per-member skill
    (dim=1), leaving only *what is distinctive about this member's profile*.
    Without this you mostly measure "who is good", not "who is different"."""
    return _cos_dist(double_centre(f))


def double_centre(f):
    return f - f.mean(0, keepdim=True) - f.mean(1, keepdim=True) + f.mean()


def task_projected_signature(delta, Y):
    """The same output delta as `delta_activation_signature`, but projected onto the
    coordinate the objective actually cares about instead of randomly sketched.

    E0's central finding lives in the gap between these two functions: a random
    sketch of the output delta is nearly uninformative, while the *same* delta read
    along the task direction is not. The output space is mostly task-irrelevant
    variation, and a JL sketch preserves it faithfully -- which is the problem."""
    d = delta.gather(2, Y.view(1, -1, 1).expand(delta.shape[0], -1, 1)).squeeze(2)
    return _cos_dist(double_centre(d))


# ------------------------------------------------------------------ aggregation

def log_det_volume(dist, tau=1.0, eps=1e-6):
    """DvD population volume: log det of the kernel Gram matrix."""
    K = torch.exp(-dist / tau)
    K = K + eps * torch.eye(K.shape[0], device=K.device)
    return torch.linalg.slogdet(K)[1]


def effective_rank(emb, eps=1e-12):
    """exp(entropy of normalised singular values). How many directions are live."""
    s = torch.linalg.svdvals(emb - emb.mean(0, keepdim=True))
    p = s / s.sum().clamp_min(eps)
    return torch.exp(-(p * torch.log(p.clamp_min(eps))).sum())
