"""Batch-environment generator for the per-NTP noise (sigma) taxonomy study.

Given a fixed set of target trajectories ``X`` and a *regime*, ``build_batch`` returns the
actual batch the scorer should run for repeat ``k``, together with the row indices at which the
targets live (so their logits can be extracted afterwards) and the realized batch size ``B``.

Regimes:
- ``warm_iso`` / ``fresh``: the batch is exactly ``X`` (B = N), identical every repeat. Isolates
  pure repeat noise (HF SDPA intra-kernel atomics / engine allocator drift); for ``fresh`` the
  orchestrator additionally runs each repeat in a brand-new process (empty cache / re-autotune).
- ``warm_serve``: prepend a deterministic-per-(seed, k) number of filler rows sampled from
  ``filler_pool`` and scatter the targets at random positions, so both the batch size B *and* the
  targets' position-within-batch vary across repeats. Batch shape/position is the dominant
  cross-environment sigma driver (v2 Finding 1: B 512->256 moves logits ~3-7 orders more than
  repeat noise), so this is the realistic "queried at an arbitrary serving moment" perturbation.
"""
from __future__ import annotations

import torch


def build_batch(X, regime, k, filler_pool, seed=0, serve_B_choices=None):
    """Return ``{"batch": LongTensor[B, L], "target_rows": list[int], "B": int}`` for repeat ``k``.

    The targets are always recoverable and unchanged: ``batch[target_rows] == X``.
    """
    N = X.shape[0]
    if regime in ("warm_iso", "fresh"):
        return {"batch": X, "target_rows": list(range(N)), "B": N}
    if regime != "warm_serve":
        raise ValueError(f"unknown regime: {regime!r}")

    g = torch.Generator().manual_seed(seed * 100003 + k)
    choices = serve_B_choices or [0, N // 2, N, 2 * N]      # filler counts to draw from
    m = int(choices[torch.randint(len(choices), (1,), generator=g).item()])
    m = min(m, filler_pool.shape[0])
    if m == 0:
        return {"batch": X, "target_rows": list(range(N)), "B": N}

    idx = torch.randperm(filler_pool.shape[0], generator=g)[:m]
    filler = filler_pool[idx]
    B = m + N
    perm = torch.randperm(B, generator=g)                  # scatter targets among filler
    tgt_pos = sorted(perm[:N].tolist())
    fill_pos = sorted(perm[N:].tolist())
    batch = torch.empty((B, X.shape[1]), dtype=X.dtype)
    for r, p in enumerate(tgt_pos):
        batch[p] = X[r]
    for r, p in enumerate(fill_pos):
        batch[p] = filler[r]
    return {"batch": batch, "target_rows": tgt_pos, "B": B}
