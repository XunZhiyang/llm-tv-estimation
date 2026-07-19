"""CPU estimators for the synthetic lower-bound tree experiment."""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from experiments.synthetic.lower_bound_instance import LowerBoundTreeInstance


def _softmax_binary_logits(log_p0: np.ndarray, log_p1: np.ndarray) -> np.ndarray:
    mx = np.maximum(log_p0, log_p1)
    e0 = np.exp(log_p0 - mx)
    e1 = np.exp(log_p1 - mx)
    return e0 / (e0 + e1)


def _logmeanexp(a: np.ndarray, axis: int) -> np.ndarray:
    mx = np.max(a, axis=axis, keepdims=True)
    ok = np.isfinite(mx)
    shifted = np.where(ok, a - mx, -np.inf)
    out = np.squeeze(mx, axis=axis) + np.log(np.mean(np.exp(shifted), axis=axis))
    out = np.where(np.squeeze(ok, axis=axis), out, -np.inf)
    return out


def v2_lr_from_logs(L_p: np.ndarray, L_q: np.ndarray) -> np.ndarray:
    R = np.zeros_like(L_p, dtype=np.float64)
    finite = np.isfinite(L_p) & np.isfinite(L_q)
    q_zero = np.isfinite(L_p) & np.isneginf(L_q)
    R[q_zero] = 1.0
    if finite.any():
        diff = np.minimum(L_q[finite] - L_p[finite], 0.0)
        R[finite] = np.maximum(1.0 - np.exp(diff), 0.0)
    return R


def exact_v2_lr(
    inst: LowerBoundTreeInstance,
    N: int,
    rng: np.random.Generator,
) -> float:
    X = inst.sample("p", N, rng)
    L_p = inst.log_prob(X, "p")
    L_q = inst.log_prob(X, "q")
    return float(v2_lr_from_logs(L_p, L_q).mean())


def method1_noisy_plugin_estimate(
    inst: LowerBoundTreeInstance,
    N: int,
    rng: np.random.Generator,
    *,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> float:
    """Naive Method 1 run on a noisy oracle as if the returned values were exact.

    It samples X~P, takes one noisy probability sketch under P and Q, treats
    those as exact log-probabilities, and plugs them into the V2-LR formula.
    This is expected to have a fixed-noise plateau as N grows.
    """
    X = inst.sample("p", N, rng)
    L_p = noisy_logprob(inst, X, "p", 1, rng, mode=mode, sigma=sigma)
    L_q = noisy_logprob(inst, X, "q", 1, rng, mode=mode, sigma=sigma)
    return float(v2_lr_from_logs(L_p, L_q).mean())


def noisy_logprob(
    inst: LowerBoundTreeInstance,
    X: np.ndarray,
    side: str,
    K: int,
    rng: np.random.Generator,
    *,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> np.ndarray:
    """K-averaged trajectory log-probability estimates.

    mode="prob_gaussian": add zero-mean Gaussian noise directly to the target
        probability.  The noise scale is sigma * sqrt(p(1-p)), matching the
        O(p) variance scale of calibrated probability sketches; deterministic
        transitions stay deterministic and K-averaging converges back to p.
    mode="gaussian_logit": add N(0,sigma^2) to binary logits at each prefix.
        This is a simple noisy-logit stress test; additive Gaussian logits are
        not exactly calibrated after softmax unless sigma is tiny.
    mode="onehot": estimate the target coordinate with K Bernoulli sketches.
    mode="exact": repeat the exact target probability K times.
    """
    target = inst.target_probs_along_path(X, side)  # (N, n)
    if mode == "exact":
        return np.log(target).sum(axis=1)

    if mode == "onehot":
        draws = rng.random((len(X), K, inst.n)) < target[:, None, :]
        avg = draws.mean(axis=1)
        with np.errstate(divide="ignore"):
            return np.log(avg).sum(axis=1)

    if mode == "prob_gaussian":
        scale = sigma * np.sqrt(np.clip(target * (1.0 - target), 0.0, None))
        draws = target[:, None, :] + rng.normal(0.0, scale[:, None, :], size=(len(X), K, inst.n))
        avg = draws.mean(axis=1)
        # The unclipped average is calibrated.  This clip is only a numerical
        # guard for the logarithm on extremely rare Gaussian tail events.
        avg = np.clip(avg, 1e-12, 1.0)
        return np.log(avg).sum(axis=1)

    if mode != "gaussian_logit":
        raise ValueError(f"Unknown score mode: {mode}")

    # Binary conditional.  We only need the realized token's noisy probability.
    p_tok = np.clip(target, 0.0, 1.0)
    p_other = 1.0 - p_tok
    with np.errstate(divide="ignore"):
        log_tok = np.log(p_tok)[:, None, :]
        log_other = np.log(p_other)[:, None, :]

    noise_tok = rng.normal(0.0, sigma, size=(len(X), K, inst.n))
    noise_other = rng.normal(0.0, sigma, size=(len(X), K, inst.n))
    noisy_tok_prob = _softmax_binary_logits(log_tok + noise_tok, log_other + noise_other)
    with np.errstate(divide="ignore"):
        log_noisy = np.log(noisy_tok_prob)
    return _logmeanexp(log_noisy, axis=1).sum(axis=1)


def symmetric_Z(L_p: np.ndarray, L_q: np.ndarray) -> np.ndarray:
    delta = L_p - L_q
    Z = np.tanh(np.abs(delta) / 2.0)
    return np.where(np.isnan(Z), 0.0, Z)


def direct_m_estimate(
    inst: LowerBoundTreeInstance,
    N_each: int,
    K: int,
    rng: np.random.Generator,
    *,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> float:
    X_p = inst.sample("p", N_each, rng)
    X_q = inst.sample("q", N_each, rng)

    Z_p = symmetric_Z(
        noisy_logprob(inst, X_p, "p", K, rng, mode=mode, sigma=sigma),
        noisy_logprob(inst, X_p, "q", K, rng, mode=mode, sigma=sigma),
    )
    Z_q = symmetric_Z(
        noisy_logprob(inst, X_q, "p", K, rng, mode=mode, sigma=sigma),
        noisy_logprob(inst, X_q, "q", K, rng, mode=mode, sigma=sigma),
    )
    return float(0.5 * (Z_p.mean() + Z_q.mean()))


def method4_v2_avg_estimate(
    inst: LowerBoundTreeInstance,
    N: int,
    K: int,
    rng: np.random.Generator,
    *,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> float:
    """One-sided K-averaged V2 likelihood-ratio estimator.

    This is the synthetic analogue of Method 4: sample X~P, score the same
    trajectories under P and Q with K-averaged noisy probability sketches, and
    average (1 - Q_hat(X) / P_hat(X))_+.
    """
    X = inst.sample("p", N, rng)
    L_p = noisy_logprob(inst, X, "p", K, rng, mode=mode, sigma=sigma)
    L_q = noisy_logprob(inst, X, "q", K, rng, mode=mode, sigma=sigma)
    return float(v2_lr_from_logs(L_p, L_q).mean())


def k3_pinsker_estimate(
    inst: LowerBoundTreeInstance,
    N: int,
    rng: np.random.Generator,
    *,
    K: int = 1,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> float:
    """K3 KL estimator followed by Pinsker's inequality.

    Samples X~P, estimates Δ=log P(X)-log Q(X), averages
    k3(Δ)=exp(-Δ)+Δ-1, then returns sqrt(mean(k3)/2).
    """
    X = inst.sample("p", N, rng)
    L_p = noisy_logprob(inst, X, "p", K, rng, mode=mode, sigma=sigma)
    L_q = noisy_logprob(inst, X, "q", K, rng, mode=mode, sigma=sigma)
    delta = L_p - L_q
    with np.errstate(over="ignore", invalid="ignore"):
        k3 = np.exp(-delta) + delta - 1.0
    finite = k3[np.isfinite(k3)]
    if len(finite) == 0:
        return float("inf")
    kl_hat = float(finite.mean())
    return math.sqrt(max(kl_hat, 0.0) / 2.0)


def _paired_Z(
    inst: LowerBoundTreeInstance,
    X: np.ndarray,
    K_high: int,
    rng: np.random.Generator,
    *,
    mode: str,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cumulative trajectory log-prob estimates for K=1..K_high."""
    if mode == "exact":
        Lp = inst.log_prob(X, "p")
        Lq = inst.log_prob(X, "q")
        return np.repeat(Lp[:, None], K_high, axis=1), np.repeat(Lq[:, None], K_high, axis=1)

    # Generate target-prob sketches for P and Q once, then slice first K.
    logs = []
    for side in ("p", "q"):
        target = inst.target_probs_along_path(X, side)
        if mode == "onehot":
            draws = rng.random((len(X), K_high, inst.n)) < target[:, None, :]
            avg_cum = np.cumsum(draws, axis=1) / np.arange(1, K_high + 1)[None, :, None]
            with np.errstate(divide="ignore"):
                logs.append(np.log(avg_cum).sum(axis=2))
        elif mode == "prob_gaussian":
            scale = sigma * np.sqrt(np.clip(target * (1.0 - target), 0.0, None))
            draws = target[:, None, :] + rng.normal(
                0.0, scale[:, None, :], size=(len(X), K_high, inst.n)
            )
            avg = np.cumsum(draws, axis=1) / np.arange(1, K_high + 1)[None, :, None]
            avg = np.clip(avg, 1e-12, 1.0)
            logs.append(np.log(avg).sum(axis=2))
        elif mode == "gaussian_logit":
            p_tok = np.clip(target, 0.0, 1.0)
            p_other = 1.0 - p_tok
            with np.errstate(divide="ignore"):
                log_tok = np.log(p_tok)[:, None, :]
                log_other = np.log(p_other)[:, None, :]
            nt = rng.normal(0.0, sigma, size=(len(X), K_high, inst.n))
            no = rng.normal(0.0, sigma, size=(len(X), K_high, inst.n))
            probs = _softmax_binary_logits(log_tok + nt, log_other + no)
            with np.errstate(divide="ignore"):
                log_probs = np.log(probs)
            # log average of probabilities for prefixes, then product over t.
            cumsum = np.cumsum(np.exp(log_probs), axis=1)
            avg = cumsum / np.arange(1, K_high + 1)[None, :, None]
            with np.errstate(divide="ignore"):
                logs.append(np.log(avg).sum(axis=2))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return logs[0], logs[1]


def mlmc_estimate(
    inst: LowerBoundTreeInstance,
    levels: Iterable[tuple[int, int]],
    rng: np.random.Generator,
    *,
    mode: str = "gaussian_logit",
    sigma: float = 0.25,
) -> float:
    """Stratified mixture MLMC estimate.

    levels is [(N_per_side_l, r_l), ...] with increasing r_l.
    Each level uses fresh trajectories; within a level, coarse and fine reuse
    the first r_{l-1} sketches from the fine sample.
    """
    total = 0.0
    prev_r = None
    for ell, (N_each, r_l) in enumerate(levels):
        if prev_r is not None and r_l <= prev_r:
            raise ValueError("levels must have strictly increasing r.")

        level_vals = []
        for side_sample in ("p", "q"):
            X = inst.sample(side_sample, N_each, rng)
            Lp_cum, Lq_cum = _paired_Z(inst, X, r_l, rng, mode=mode, sigma=sigma)
            Z_high = symmetric_Z(Lp_cum[:, r_l - 1], Lq_cum[:, r_l - 1])
            if ell == 0:
                Y = Z_high
            else:
                Z_low = symmetric_Z(Lp_cum[:, prev_r - 1], Lq_cum[:, prev_r - 1])
                Y = Z_high - Z_low
            level_vals.append(Y.mean())
        total += 0.5 * float(level_vals[0] + level_vals[1])
        prev_r = r_l
    return total


def budget_direct_m(inst: LowerBoundTreeInstance, N_each: int, K: int) -> int:
    return inst.n * (2 * N_each) * (1 + 2 * K)


def budget_method4(inst: LowerBoundTreeInstance, N: int, K: int) -> int:
    return inst.n * N * (1 + 2 * K)


def budget_method1(inst: LowerBoundTreeInstance, N: int) -> int:
    return 2 * inst.n * N


def budget_mlmc(inst: LowerBoundTreeInstance, levels: Iterable[tuple[int, int]]) -> int:
    return int(sum(inst.n * (2 * N) * (1 + 2 * r) for N, r in levels))
