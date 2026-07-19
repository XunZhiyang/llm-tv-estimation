"""Re-run all 6 Fig1 estimators from saved pi_pool.pt + mu_pool.pt — no GPU.

DENSE mode: each estimator evaluated at many (K, N) or (L, N_scale) configs
on the same cache → smooth curves not 4 scatter points.

Method 1, 5:  N sub-batch  → ~10 points
Method 4:     (K, N) grid  → 8 K × 4-6 N
Direct-M:     (K, N) grid  → 8 K × 4-6 N
MLMC logit/onehot: (L_use, N_scale) grid → 4 L × 3 scales
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.fig1.methods import (
    method1_fused, method5_fused, method4_pi_direct_K, method5_pi_direct_K,
    direct_M_K, mlmc,
)


def parse_levels(s: str):
    return [tuple(int(x) for x in tok.split(":")) for tok in s.split(",")]


def row_completed(pool: dict) -> torch.Tensor:
    if "row_completed_k" in pool:
        return pool["row_completed_k"]
    completed = int(pool["meta"].get("completed_k", pool["meta"]["K_max"]))
    return torch.full((pool["meta"]["N"],), completed, dtype=torch.int16)


def slice_pool(pool: dict, n_keep: int, min_K: int = 0) -> dict | None:
    """Return the first n_keep trajectories with at least min_K completed scores."""
    eligible = torch.nonzero(row_completed(pool) >= min_K, as_tuple=False).flatten()
    if len(eligible) < n_keep:
        return None
    idx = eligible[:n_keep]
    out = {k: (v[idx] if isinstance(v, torch.Tensor) and v.shape[:1] == (pool["meta"]["N"],) else v)
           for k, v in pool.items()}
    out["meta"] = {**pool["meta"], "N": n_keep}
    if "row_completed_k" in out:
        out["row_completed_k"] = out["row_completed_k"].clone()
    return out


def scaled_levels(base_levels, scale: float):
    """Multiply each level's N_per_side by `scale`, round, clip ≥ 1."""
    out = []
    for N_l, r_l in base_levels:
        N_new = max(1, int(round(N_l * scale)))
        out.append((N_new, r_l))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pi_pool", default="results/fig1/pi_pool.pt")
    p.add_argument("--mu_pool", default="results/fig1/mu_pool.pt")
    p.add_argument("--curves_in", default="results/fig1/fig1_curves.json")
    p.add_argument("--curves_out", default="results/fig1/fig1_curves.json")
    p.add_argument("--N_grid", default="10,20,30,50,75,100",
                   help="Sub-batch sizes for Methods 1/5 and 4/Direct-M")
    p.add_argument("--K_grid", default="1,2,3,4,5,6,7,8")
    p.add_argument("--mlmc_scales", default="0.4,0.6,0.8,1.0",
                   help="Scales of base mlmc_levels (1.0 = full)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    pi_pool = torch.load(args.pi_pool, weights_only=False)
    mu_pool = torch.load(args.mu_pool, weights_only=False)
    cfg_old = json.load(open(args.curves_in))["config"]

    available_K = min(
        int(row_completed(pi_pool).max().item()),
        int(row_completed(mu_pool).max().item()),
        int(pi_pool["meta"]["K_max"]),
        int(mu_pool["meta"]["K_max"]),
    )
    K_grid = [int(x) for x in args.K_grid.split(",")]
    K_grid = [k for k in K_grid if k <= available_K]
    if not K_grid:
        raise ValueError(f"No usable K in grid; available_K={available_K}")
    N_grid = [int(x) for x in args.N_grid.split(",")]
    base_levels = [tuple(L) for L in cfg_old["mlmc_levels"]]
    mlmc_scales = [float(x) for x in args.mlmc_scales.split(",")]
    seed = args.seed if args.seed is not None else cfg_old["seed"]

    print(f"[dense] pi N={pi_pool['meta']['N']}, mu N={mu_pool['meta']['N']}, "
          f"K_max={pi_pool['meta']['K_max']}, available_K={available_K}")
    print(
        f"[dense] row completed: pi min/max={int(row_completed(pi_pool).min())}/"
        f"{int(row_completed(pi_pool).max())}, mu min/max={int(row_completed(mu_pool).min())}/"
        f"{int(row_completed(mu_pool).max())}"
    )
    print(f"[dense] N_grid={N_grid}, K_grid={K_grid}, mlmc_scales={mlmc_scales}")

    # ── Method 1, 5 fused: vary N ──────────────────────────────────────────
    print("\n=== Method 1 / 5 fused (N sweep) ===")
    m1_curve, m5_curve = [], []
    for N in N_grid:
        sl = slice_pool(pi_pool, N, min_K=1)
        if sl is None:
            continue
        r1 = method1_fused(sl); r5 = method5_fused(sl)
        m1_curve.append(r1); m5_curve.append(r5)
        print(f"  N={N:3d}  M1: budget={r1['budget_forwards']:>8d} est={r1['estimate']:.3f} SE={r1['stderr']:.3f}  "
              f"M5: KL_fin={r5['KL_hat_finite']:.3f} Pinsker={r5['estimate']:.3f}")

    # ── Method 4 π-direct: (K, N) grid ────────────────────────────────────
    print("\n=== Method 4 π-direct (K × N grid) ===")
    m4_curve = []
    for K in K_grid:
        for N in N_grid:
            sl = slice_pool(pi_pool, N, min_K=K)
            if sl is None:
                continue
            r = method4_pi_direct_K(sl, K=K)
            m4_curve.append(r)
        print(f"  K={K} done")

    # ── Method 5 K-avg Pinsker: (K, N) grid ────────────────────────────────
    print("\n=== Method 5 K-avg Pinsker (K × N grid) ===")
    m5k_curve = []
    for K in K_grid:
        last = None
        for N in N_grid:
            sl = slice_pool(pi_pool, N, min_K=K)
            if sl is None:
                continue
            r = method5_pi_direct_K(sl, K=K)
            m5k_curve.append(r)
            last = (N, r)
        if last is None:
            print(f"  K={K} skipped (no eligible rows)")
        else:
            N, r = last
            print(f"  K={K} done  (last N={N}: KL_fin={r['KL_hat_finite']:.3f} "
                  f"Pinsker={r['estimate']:.3f} inf_rate={r['inf_rate']:.2f})")

    # ── Direct-M: (K, N_each) grid ────────────────────────────────────────
    print("\n=== Direct-M K (K × N_each grid) ===")
    dm_curve = []
    for K in K_grid:
        for N_each in N_grid:
            pi_sl = slice_pool(pi_pool, N_each, min_K=K)
            mu_sl = slice_pool(mu_pool, N_each, min_K=K)
            if pi_sl is None or mu_sl is None:
                continue
            r = direct_M_K(pi_sl, mu_sl, K=K, N_each=N_each)
            dm_curve.append(r)
        print(f"  K={K} done")

    # ── MLMC logit / onehot: (L, scale) grid ──────────────────────────────
    print("\n=== MLMC logit (L × scale) ===")
    mlmc_logit_curve = []
    mlmc_onehot_curve = []
    for scale in mlmc_scales:
        levels = [(N, r) for N, r in scaled_levels(base_levels, scale) if r <= available_K]
        if not levels:
            print(f"  scale={scale}  SKIP (no level with r <= available_K={available_K})")
            continue
        # Skip if any level violates pool sizes.
        total_N = sum(N for N, _ in levels)
        max_r = max(r for _, r in levels)
        pi_sl = slice_pool(pi_pool, total_N, min_K=max_r)
        mu_sl = slice_pool(mu_pool, total_N, min_K=max_r)
        if pi_sl is None or mu_sl is None:
            print(f"  scale={scale}  SKIP (total N={total_N} > pool size)")
            continue
        for L_use in range(len(levels)):
            r_lo = mlmc(pi_sl, mu_sl, levels=levels, sketch="logit",
                        L_use=L_use, seed=seed)
            r_oh = mlmc(pi_sl, mu_sl, levels=levels, sketch="onehot",
                        L_use=L_use, seed=seed)
            r_lo["scale"] = scale; r_oh["scale"] = scale
            mlmc_logit_curve.append(r_lo); mlmc_onehot_curve.append(r_oh)
        print(f"  scale={scale} levels={levels} done")

    # ── Save ───────────────────────────────────────────────────────────────
    curves = {
        "config": cfg_old,
        "ground_truth_TV": 0.0,
        "method1_fused":      m1_curve,    # now a list
        "method5_fused":      m5_curve,
        "method4_pi_direct":  m4_curve,
        "method5_K_avg":      m5k_curve,   # NEW: Pinsker on K-averaged log-probs
        "direct_m_K":         dm_curve,
        "mlmc_logit":         mlmc_logit_curve,
        "mlmc_onehot":        mlmc_onehot_curve,
        "config_dense": {
            "N_grid": N_grid, "K_grid": K_grid,
            "mlmc_scales": mlmc_scales,
            "available_K": available_K,
        },
    }
    out = Path(args.curves_out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"\n[dense] Wrote {out}  "
          f"(M1={len(m1_curve)} M5={len(m5_curve)} M4={len(m4_curve)} "
          f"DM={len(dm_curve)} MLMC={len(mlmc_logit_curve)})")


if __name__ == "__main__":
    main()
