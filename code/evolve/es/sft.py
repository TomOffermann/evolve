"""Supervised fine-tuning of the base policy, by gradient descent.

Deliberately gradient-based: this mirrors real practice, where SFT gives the model
its output *format* and RL/ES post-training supplies the *capability*. It also keeps
the two phases cleanly separable, so post-training results are not confounded by how
much the base model already knew.

The clone targets are syntactically valid expressions paired with a **random** target
token, so the model learns to emit well-formed expressions using the given numbers
and learns nothing about the target mapping. See `tasks/countdown.py`.
"""

import torch
import torch.nn.functional as F


def sft(policy, X, Y, steps=1500, bs=512, lr=3e-3, seed=0, log_every=None):
    params = [v.requires_grad_(True) for v in policy.W.values()]
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.Generator().manual_seed(seed)
    for i in range(steps):
        idx = torch.randint(X.shape[0], (bs,), generator=g)
        loss = F.cross_entropy(policy.logits_base(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_every and i % log_every == 0:
            print(f"    sft {i:5d}  loss {loss.item():.4f}")
    for v in policy.W.values():
        v.requires_grad_(False)
    return float(loss.item())
