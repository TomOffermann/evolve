"""JEPA on pendulum -- a self-supervised objective where collapse is THE failure.

Motivation (P3). Joint-Embedding Predictive Architectures predict *latent* embeddings
of future observations. Their central failure is representation collapse: the encoder
can drive the loss to zero by emitting a constant. The whole apparatus of
stop-gradients, EMA target encoders and architectural asymmetry exists to stop
gradient descent from finding that shortcut.

With ES the fitness is a black box, so anti-collapse can go **directly into the
objective** -- including terms that are rank-based, discrete or otherwise
non-differentiable. The claim worth testing: those asymmetry hacks are artefacts of
*how we optimise*, not of the objective.

This is also the right complement to countdown: countdown has sparse reward and
policy-level collapse; JEPA has dense reward and *representation*-level collapse.
Between them they cover both failure modes the programme cares about.

Setup:
    dynamics  theta'' = -(g/l) sin(theta) - damping * theta' + torque
    obs       (cos t, sin t, theta_dot)              3-dim
    encoder   obs -> z                               latent
    predictor (z_t, a_t) -> z_hat_{t+1}
    fitness   -(prediction error) - anti-collapse penalties

Evaluation is deliberately NOT the fitness: a collapsed encoder scores a perfect
prediction loss. We report the latent's effective rank and a linear-probe R^2 for
recovering true state, which a collapsed encoder cannot fake.
"""

import numpy as np
import torch


class Pendulum:
    """Batched pendulum. State (theta, theta_dot); action = torque."""

    def __init__(self, dt=0.05, g=9.81, l=1.0, damping=0.1, max_torque=2.0):
        self.dt, self.g, self.l = dt, g, l
        self.damping, self.max_torque = damping, max_torque

    def reset(self, n, rng):
        th = rng.uniform(-np.pi, np.pi, n)
        thd = rng.uniform(-1.0, 1.0, n)
        return np.stack([th, thd], 1).astype(np.float32)

    def step(self, s, a):
        th, thd = s[:, 0], s[:, 1]
        a = np.clip(a, -self.max_torque, self.max_torque)
        thdd = -(self.g / self.l) * np.sin(th) - self.damping * thd + a
        thd2 = thd + self.dt * thdd
        th2 = th + self.dt * thd2
        return np.stack([th2, thd2], 1).astype(np.float32)

    def obs(self, s):
        return np.stack([np.cos(s[:, 0]), np.sin(s[:, 0]), s[:, 1] / 8.0], 1).astype(np.float32)

    def trajectories(self, n, T, seed):
        """Returns obs (n, T+1, 3), actions (n, T, 1), true states (n, T+1, 2)."""
        rng = np.random.default_rng(seed)
        s = self.reset(n, rng)
        O, A, S = [self.obs(s)], [], [s]
        for _ in range(T):
            a = rng.uniform(-self.max_torque, self.max_torque, n).astype(np.float32)
            s = self.step(s, a)
            O.append(self.obs(s))
            A.append(a[:, None])
            S.append(s)
        return (np.stack(O, 1), np.stack(A, 1).astype(np.float32), np.stack(S, 1))


class JEPA:
    """Encoder + predictor, both small MLPs, perturbed EGGROLL-style."""

    def __init__(self, obs_dim=3, act_dim=1, latent=8, hidden=64, device="cpu", seed=0):
        g = torch.Generator().manual_seed(seed)
        self.latent = latent
        self.device = device
        self.W = {
            "enc1": torch.randn(hidden, obs_dim, generator=g) / obs_dim**0.5,
            "enc2": torch.randn(latent, hidden, generator=g) / hidden**0.5,
            "pred1": torch.randn(hidden, latent + act_dim, generator=g) / (latent + act_dim) ** 0.5,
            "pred2": torch.randn(latent, hidden, generator=g) / hidden**0.5,
        }
        self.W = {k: v.to(device) for k, v in self.W.items()}

    @property
    def shapes(self):
        return {k: tuple(v.shape) for k, v in self.W.items()}

    def n_params(self):
        return sum(v.numel() for v in self.W.values())

    @staticmethod
    def _lin(x, W, fac, name, s):
        """y = x W^T + s (x B) A^T, batched over the population."""
        y = x @ W.T
        if fac is not None:
            A, B = fac[name]
            y = y + s * torch.einsum("nbi,nir,nor->nbo", x, B, A)
        return y

    def encode(self, obs, fac=None, sigma=0.0, rank=1):
        s = sigma / rank**0.5
        h = torch.tanh(self._lin(obs, self.W["enc1"], fac, "enc1", s))
        return self._lin(h, self.W["enc2"], fac, "enc2", s)

    def predict(self, z, a, fac=None, sigma=0.0, rank=1):
        s = sigma / rank**0.5
        h = torch.tanh(self._lin(torch.cat([z, a], -1), self.W["pred1"], fac, "pred1", s))
        return self._lin(h, self.W["pred2"], fac, "pred2", s)


def fitness(model, obs_t, obs_t1, act, fac=None, sigma=0.0, rank=1,
            var_weight=1.0, cov_weight=0.5, scale_normalise=False, mask_action=True,
            per_example=True):
    """VICReg-style objective, evaluated as a black box.

    The variance hinge and covariance penalty are the anti-collapse terms. With ES
    they need not be differentiable -- they simply enter the fitness. Returns
    (F, per_example_fitness, latents)."""
    # Hiding the action makes the next state genuinely unpredictable: the torque is
    # random, so the best honest predictor still carries irreducible error, while a
    # constant encoder achieves exactly zero. Collapse therefore strictly wins.
    #
    # This matters. With the action visible the pendulum is deterministic, accurate
    # prediction is easy, and collapse is never the attractor -- so the "prediction
    # only" control does not fail and the experiment tests nothing. Real JEPAs
    # collapse precisely when prediction is hard.
    if mask_action:
        act = torch.zeros_like(act)

    z_t = model.encode(obs_t, fac, sigma, rank)
    z_t1 = model.encode(obs_t1, fac, sigma, rank)
    z_hat = model.predict(z_t, act, fac, sigma, rank)

    pred_err = ((z_hat - z_t1) ** 2).mean(-1)                    # (N, B)

    # variance hinge: every latent dim must keep a standard deviation of >= 1
    std = z_t.std(dim=1)                                          # (N, D)
    var_pen = torch.relu(1.0 - std).mean(-1, keepdim=True)        # (N, 1)

    # covariance: off-diagonal decorrelation
    zc = z_t - z_t.mean(dim=1, keepdim=True)
    B = zc.shape[1]
    cov = torch.einsum("nbi,nbj->nij", zc, zc) / max(B - 1, 1)
    off = cov - torch.diag_embed(torch.diagonal(cov, dim1=-2, dim2=-1))
    cov_pen = (off**2).sum((-2, -1), keepdim=False)[:, None] / z_t.shape[-1]

    # `scale_normalise` divides the prediction error by the latent variance, which
    # makes shrinking the latents worthless. That is itself an anti-collapse
    # mechanism, so it must default OFF -- with it on, the "prediction only" control
    # cannot collapse and the experiment proves nothing. (It also makes the
    # objective scale-free, under which the latents simply explode instead.)
    if scale_normalise:
        pred_err = pred_err / (z_t1.var(dim=1).mean(-1, keepdim=True) + 1e-6)
    per_ex = -pred_err - var_weight * var_pen - cov_weight * cov_pen
    F = per_ex.mean(-1)
    return F, (per_ex if per_example else None), z_t


@torch.no_grad()
def evaluate(model, obs, states, fac=None, sigma=0.0):
    """Collapse-proof evaluation. A collapsed encoder gets a perfect prediction
    loss, so we never score on that. Effective rank and a linear probe for the
    true state cannot be faked by collapsing."""
    z = model.encode(obs.unsqueeze(0), fac, sigma).squeeze(0)      # (B, D)
    zc = z - z.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(zc)
    p = sv / sv.sum().clamp_min(1e-12)
    eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))

    # linear probe: can true (cos, sin, theta_dot) be recovered from the latent?
    X = torch.cat([zc, torch.ones(zc.shape[0], 1)], 1)
    Y = states - states.mean(0, keepdim=True)
    sol = torch.linalg.lstsq(X, Y).solution
    resid = ((Y - X @ sol) ** 2).sum()
    r2 = float(1.0 - resid / (Y**2).sum().clamp_min(1e-12))
    return eff_rank, r2, float(z.std(0).mean())
