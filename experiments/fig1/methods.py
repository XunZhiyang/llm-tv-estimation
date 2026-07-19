"""Estimators for Figure 1 (shared pool design).

All methods read from pi_pool / mu_pool (built by cache_build.build_pool).

CAVEAT (fused methods on engine pools): methods 1/5 mix sample-time L with a
scored L and are only valid when both carry the same normalization. HF pools
satisfy this; engine pools do NOT (phase_sample's sample_lps are the engines'
raw pre-truncation log-softmax, while replay scores are top-k renormalized) —
do not run the fused methods on engine pools.

  Method 1 fused (V_2-LR):  pi_pool.sample_L_pi + pi_pool.score_L_mu[:, 0, :]
  Method 5 fused (Pinsker): same fused inputs
  Method 4 π-direct K:      pi_pool.score_L_{pi,mu}[:, :K, :]
  Direct-M K baseline:      pi_pool[:N_each] + mu_pool[:N_each], symmetric
                             Z_K = tanh(|Σ_t Δloĝ|/2), avg over both halves
  Method 3 MLMC logit:      mixture pool sliced into disjoint levels by index;
                             coupling within a level (first r_l score evals)
  Method 2 MLMC one-hot:    same MLMC structure, sketches drawn as
                             Bern(exp(score_L)) per (i, k, t) on the fly

Budget formula (forwards = batched calls × n):
  Method 1, 5:    n × N × 2                  (1 sample + 1 μ score per traj)
  Method 4 (K):   n × N × (1 + 2K)
  Direct-M (K):   n × (N_pi + N_mu) × (1 + 2K)
  MLMC:           n × Σ_l (N_pi_l + N_mu_l) × (1 + 2 r_l)
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# ── helpers ──────────────────────────────────────────────────────────────────

def _logmeanexp_K(score_L: torch.Tensor, K: int) -> torch.Tensor:
    """log( (1/K) Σ_k exp(score_L[:, k, :]) ) over the first K of K_max evals.
    score_L: (N, K_max, n) → (N, n).  -inf entries handled by logsumexp."""
    return torch.logsumexp(score_L[:, :K, :], dim=1) - math.log(K)


def _Z_symmetric(L_pi: torch.Tensor, L_mu: torch.Tensor) -> torch.Tensor:
    """Z(X) = tanh(|L_pi(X) - L_mu(X)| / 2) per trajectory.
    L_pi, L_mu: (N,) trajectory-total log-probs (may contain ±inf).
    Returns (N,) in [0, 1].
    """
    delta = L_pi - L_mu                                   # (N,)
    # If both -inf: delta = NaN → treat Z=0 (no info).
    # If one -inf, other finite: |Δ| = +inf → tanh = 1.
    Z = torch.tanh(delta.abs() / 2.0)
    Z = torch.where(torch.isnan(Z), torch.zeros_like(Z), Z)
    return Z


def _v2_lr(L_pi: torch.Tensor, L_mu: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """V_2 LR per traj: R = (1 - exp(L_mu - L_pi))_+, handling ±inf gracefully.
       Both -inf  → R = 0 (no info)
       Only L_mu = -inf → R = 1  (μ assigns 0 mass to this traj)
       Only L_pi = -inf → R = 0  (μ/π = ∞)
       Both finite → standard (1 - exp(diff))_+ in [0, 1]."""
    pi_ninf = torch.isneginf(L_pi)
    mu_ninf = torch.isneginf(L_mu)
    both_ninf = pi_ninf & mu_ninf
    mu_only   = mu_ninf & ~pi_ninf
    finite    = ~pi_ninf & ~mu_ninf

    R = torch.zeros_like(L_pi)
    R[mu_only] = 1.0
    if finite.any():
        d = L_mu[finite] - L_pi[finite]
        R[finite] = (1.0 - torch.exp(d.clamp(max=0.0))).clamp(min=0.0)
    diag = {
        "n_both_neg_inf": int(both_ninf.sum().item()),
        "n_mu_only_neg_inf": int(mu_only.sum().item()),
        "n_pi_only_neg_inf": int((pi_ninf & ~mu_ninf).sum().item()),
        "n_finite": int(finite.sum().item()),
    }
    return R, diag


# ── Method 1 fused ───────────────────────────────────────────────────────────

def method1_fused(pi_pool: dict) -> dict:
    """V_2-LR with sample-time L_π + 1 fresh L_μ score."""
    L_pi = pi_pool["sample_L_pi"].sum(dim=-1)            # (N,)
    L_mu = pi_pool["score_L_mu"][:, 0, :].sum(dim=-1)
    R, diag = _v2_lr(L_pi, L_mu)
    N = L_pi.shape[0]
    n = pi_pool["meta"]["n"]
    return {
        "estimate": float(R.mean().item()),
        "stderr": float(R.std(unbiased=True).item() / math.sqrt(N)) if N > 1 else 0.0,
        "N": N, "K": 1,
        "budget_forwards": 2 * n * N,
        **{f"diag_{k}": v for k, v in diag.items()},
    }


# ── Method 5 fused (Pinsker) ─────────────────────────────────────────────────

def method5_fused(pi_pool: dict) -> dict:
    """k3 Bregman estimator on fused inputs (sample-time L_π, 1 fresh L_μ).
        k3(Δ) = exp(-Δ) + Δ - 1, Δ = L_π - L_μ, ≥ 0 by construction.
    Pinsker: TV ≤ sqrt(k3_mean / 2). Replaces the older k1+clamp version."""
    L_pi = pi_pool["sample_L_pi"].sum(dim=-1)
    L_mu = pi_pool["score_L_mu"][:, 0, :].sum(dim=-1)
    delta = L_pi - L_mu                                  # (N,)
    k3 = torch.exp(-delta) + delta - 1.0

    inf_count = int(torch.isposinf(k3).sum().item())
    nan_cnt = int(torch.isnan(k3).sum().item())
    finite = k3[torch.isfinite(k3)]
    if len(finite) == 0:
        KL_finite = float("inf"); estimate = float("inf")
    else:
        KL_finite = float(finite.mean().item())
        estimate = math.sqrt(KL_finite / 2.0)

    N = L_pi.shape[0]
    n = pi_pool["meta"]["n"]
    return {
        "estimate": estimate, "KL_hat_finite": KL_finite,
        "inf_rate": (inf_count + nan_cnt) / N, "n_finite": len(finite),
        "N": N, "budget_forwards": 2 * n * N,
        "estimator": "k3",
    }


# ── Method 5 K-avg (Pinsker on K-averaged log-probs, k3 Bregman) ─────────────

def method5_pi_direct_K(pi_pool: dict, K: int) -> dict:
    """Pinsker on K-averaged log-probs using the k3 Bregman estimator
        k3(Δ) = exp(-Δ) + Δ - 1,    Δ = L_π_K - L_μ_K,
    which is ≥ 0 by construction (Bregman of -log at 1) and unbiased for KL
    when the K-averaged log-probs are exact. With sdpa noise k3 is biased
    upward by ~½·Var(Δ); we report it as-is — Pinsker is loose by design.
    Replaces the older k1 = L_π - L_μ + clamp(max=0) version, which silently
    hid noise-driven negative KL_hat as TV=0."""
    L_pi = _logmeanexp_K(pi_pool["score_L_pi"], K).sum(dim=-1)
    L_mu = _logmeanexp_K(pi_pool["score_L_mu"], K).sum(dim=-1)
    delta = L_pi - L_mu                                  # (N,)
    k3 = torch.exp(-delta) + delta - 1.0                 # ≥ 0 a.s. when finite

    pos_inf = int(torch.isposinf(k3).sum().item())   # Δ=±∞ either side
    nan_cnt = int(torch.isnan(k3).sum().item())      # Δ=NaN (both -inf)
    finite = k3[torch.isfinite(k3)]
    N = L_pi.shape[0]; n = pi_pool["meta"]["n"]

    if len(finite) == 0:
        KL_finite = float("inf"); KL_se = float("inf"); estimate = float("inf")
    else:
        KL_finite = float(finite.mean().item())
        KL_se = float(finite.std(unbiased=True).item() / math.sqrt(len(finite))) \
                if len(finite) > 1 else 0.0
        estimate = math.sqrt(KL_finite / 2.0)        # k3 ≥ 0, no clamp needed

    return {
        "estimate": estimate, "KL_hat_finite": KL_finite, "KL_se": KL_se,
        "stderr": KL_se,
        "inf_rate": (pos_inf + nan_cnt) / N,
        "n_finite": len(finite),
        "n_pos_inf": pos_inf, "n_nan": nan_cnt,
        "K": K, "N": N,
        "budget_forwards": n * N * (1 + 2 * K),
        "estimator": "k3",
    }


# ── Method 4: π-direct K (one-sided V_2 LR) ──────────────────────────────────

def method4_pi_direct_K(pi_pool: dict, K: int) -> dict:
    """V_2 = E_{X~π}[(1 - μ̂_K(X)/π̂_K(X))_+], K-averaged in prob space."""
    L_pi = _logmeanexp_K(pi_pool["score_L_pi"], K).sum(dim=-1)
    L_mu = _logmeanexp_K(pi_pool["score_L_mu"], K).sum(dim=-1)
    R, diag = _v2_lr(L_pi, L_mu)
    N = L_pi.shape[0]
    n = pi_pool["meta"]["n"]
    return {
        "estimate": float(R.mean().item()),
        "stderr": float(R.std(unbiased=True).item() / math.sqrt(N)) if N > 1 else 0.0,
        "K": K, "N": N,
        "budget_forwards": n * N * (1 + 2 * K),
        **{f"diag_{k}": v for k, v in diag.items()},
    }


# ── Direct-M K baseline (symmetric mixture, no MLMC) ────────────────────────

def _Z_K_logit(pool: dict, K: int, idx_slice: slice) -> torch.Tensor:
    """Per-traj Z_K(X) = tanh(|L_pi_K - L_mu_K|/2), logit (probability) sketch."""
    L_pi_K = _logmeanexp_K(pool["score_L_pi"][idx_slice], K)   # (N', n)
    L_mu_K = _logmeanexp_K(pool["score_L_mu"][idx_slice], K)
    return _Z_symmetric(L_pi_K.sum(-1), L_mu_K.sum(-1))


def direct_M_K(pi_pool: dict, mu_pool: dict, K: int, N_each: int) -> dict:
    """Stratified mixture estimator, K-averaged probability sketches, no MLMC.
    estimate = ½ E_{X~π}[Z_K] + ½ E_{X~μ}[Z_K]
    """
    sl = slice(0, N_each)
    Z_pi = _Z_K_logit(pi_pool, K, sl)                    # (N_each,)
    Z_mu = _Z_K_logit(mu_pool, K, sl)
    estimate = 0.5 * (Z_pi.mean().item() + Z_mu.mean().item())
    var = 0.25 * (Z_pi.var(unbiased=True) / N_each + Z_mu.var(unbiased=True) / N_each)
    n = pi_pool["meta"]["n"]
    return {
        "estimate": float(estimate),
        "stderr": float(var.sqrt().item()),
        "K": K, "N_pi": N_each, "N_mu": N_each,
        "budget_forwards": n * (2 * N_each) * (1 + 2 * K),
    }


# ── MLMC: logit sketch ───────────────────────────────────────────────────────

def _draw_bern(pool: dict, K_high: int, idx_slice: slice,
               rng: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-draw K_high Bern per (i, k, t) so subsequent Z_K can slice [:K] for
    proper MLMC coupling between Z_{r_l} and Z_{r_{l-1}} on the same trajectories."""
    sp = pool["score_L_pi"][idx_slice, :K_high, :]
    sm = pool["score_L_mu"][idx_slice, :K_high, :]
    return (torch.bernoulli(sp.exp().clamp(0.0, 1.0), generator=rng),
            torch.bernoulli(sm.exp().clamp(0.0, 1.0), generator=rng))


def _Z_K_onehot_from_bern(bern_pi: torch.Tensor, bern_mu: torch.Tensor, K: int) -> torch.Tensor:
    """Z_K from pre-drawn Bern tensors. K ≤ bern.shape[1].

    Zeros must propagate as true -inf so that _Z_symmetric implements the
    paper's eq:hzdef convention: both trajectory products zero -> NaN -> Z=0;
    exactly one zero -> |delta|=inf -> Z=1. A min-clamp here (removed
    2026-06-10) silently turned Z into a zero-token-count comparison,
    producing an unphysical Y_0 ~ 0.95.
    """
    p_hat_pi = bern_pi[:, :K, :].mean(dim=1)             # (N', n)
    p_hat_mu = bern_mu[:, :K, :].mean(dim=1)
    L_pi = p_hat_pi.log().sum(-1)
    L_mu = p_hat_mu.log().sum(-1)
    return _Z_symmetric(L_pi, L_mu)


def _level_slices(N_per_side_total: int, level_N_per_side: list[int]) -> list[slice]:
    """Cumulative disjoint slices into the pool for each level."""
    assert sum(level_N_per_side) <= N_per_side_total
    slices = []
    off = 0
    for n_l in level_N_per_side:
        slices.append(slice(off, off + n_l))
        off += n_l
    return slices


def mlmc(
    pi_pool: dict, mu_pool: dict,
    levels: list[tuple[int, int]],   # [(N_per_side_l, r_l), ...]
    sketch: str = "logit",            # "logit" or "onehot"
    L_use: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """Multilevel mixture estimator with telescoping coupling.

    Each level uses a disjoint slice of (pi_pool, mu_pool), with N_per_side_l
    trajectories from each pool. r_l is the # of fresh score evals used at level l.
    Coupling: high-resolution Z uses first r_l evals; low-res Z uses first r_{l-1}.

    sketch = "logit"  : K-averaged probabilities (deterministic given pool)
    sketch = "onehot" : Bern(p) at each (i, k, t), then averaged (random)

    L_use = number of levels INCLUDED minus 1 (so L_use=2 → levels 0,1,2).
    """
    if L_use is None:
        L_use = len(levels) - 1
    assert 0 <= L_use < len(levels)
    n = pi_pool["meta"]["n"]

    level_N = [N_l for N_l, _ in levels]
    sl_per_level = _level_slices(pi_pool["meta"]["N"], level_N)
    # Same slice indices into mu_pool (assumes mu_pool has ≥ same N per side).
    assert mu_pool["meta"]["N"] >= sum(level_N)

    rng = torch.Generator(device="cpu"); rng.manual_seed(seed)
    if sketch not in ("logit", "onehot"):
        raise ValueError(sketch)

    Y_levels = []
    budget = 0
    for ell in range(L_use + 1):
        N_l, r_l = levels[ell]
        sl = sl_per_level[ell]
        r_prev = levels[ell - 1][1] if ell > 0 else None
        if r_prev is not None and r_prev > r_l:
            raise ValueError(f"non-monotone r at level {ell}: r_prev={r_prev} > r_l={r_l}")

        if sketch == "logit":
            Z_pi_high = _Z_K_logit(pi_pool, r_l, sl)
            Z_mu_high = _Z_K_logit(mu_pool, r_l, sl)
            if ell == 0:
                Z_pi, Z_mu = Z_pi_high, Z_mu_high
            else:
                Z_pi = Z_pi_high - _Z_K_logit(pi_pool, r_prev, sl)
                Z_mu = Z_mu_high - _Z_K_logit(mu_pool, r_prev, sl)
        else:  # onehot — pre-draw Bern once per side, slice for high/low
            bp_pi, bm_pi = _draw_bern(pi_pool, r_l, sl, rng)
            bp_mu, bm_mu = _draw_bern(mu_pool, r_l, sl, rng)
            Z_pi_high = _Z_K_onehot_from_bern(bp_pi, bm_pi, r_l)
            Z_mu_high = _Z_K_onehot_from_bern(bp_mu, bm_mu, r_l)
            if ell == 0:
                Z_pi, Z_mu = Z_pi_high, Z_mu_high
            else:
                Z_pi = Z_pi_high - _Z_K_onehot_from_bern(bp_pi, bm_pi, r_prev)
                Z_mu = Z_mu_high - _Z_K_onehot_from_bern(bp_mu, bm_mu, r_prev)

        Y = 0.5 * (Z_pi.mean().item() + Z_mu.mean().item())
        Y_levels.append({
            "ell": ell, "r": r_l, "N_per_side": N_l,
            "Y_pi": float(Z_pi.mean().item()),
            "Y_mu": float(Z_mu.mean().item()),
            "Y": float(Y),
        })
        budget += 2 * N_l * (1 + 2 * r_l)

    estimate = sum(y["Y"] for y in Y_levels)
    return {
        "estimate": float(estimate),
        "Y_levels": Y_levels,
        "L_use": L_use, "sketch": sketch,
        "budget_forwards": n * budget,
    }
