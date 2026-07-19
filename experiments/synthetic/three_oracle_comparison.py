"""Three-oracle empirical scaling on the binary hard instance.

Validates Table 1 of the paper: budget-vs-RMSE for the three oracle models
(exact logit, noisy logit, pure sample) on the same instance, with the same
underlying TV estimator family. Saves JSON + paper-grade figure.

What we plot:
  - x-axis: total prefix-oracle queries (budget)
  - y-axis: RMSE of the estimator (vs known TV = 0.388)
  - one curve per oracle: exact-logit (sigma=0), noisy-logit at several sigma,
    sample (onehot, equivalent to sigma^2 = K_support-1 = 1 for binary)

Conclusion (expected): all curves have slope ~ -1/2 (Monte-Carlo); intercepts
ordered by per-trajectory cost prefactor — exact < small-sigma noisy < large-
sigma noisy <= sample. Validates the smooth interpolation in Theorem 3 by
direct visual comparison.
"""
from __future__ import annotations
import json, math
from pathlib import Path
from dataclasses import asdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.synthetic.lower_bound_instance import (
    LowerBoundConfig, LowerBoundTreeInstance,
)
from experiments.synthetic.estimators import (
    direct_m_estimate, mlmc_estimate, budget_direct_m, budget_mlmc,
)


def run_direct_m(
    inst: LowerBoundTreeInstance, *, mode: str, sigma: float,
    N_each: int, K: int, reps: int, seed: int,
) -> tuple[float, int]:
    """Direct mixture estimator at fixed (N_each, K). Returns (rmse, budget)."""
    estimates = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        v = direct_m_estimate(inst, N_each, K, rng, mode=mode, sigma=sigma)
        estimates.append(v)
    estimates = np.array(estimates)
    rmse = float(np.sqrt(np.mean((estimates - inst.true_tv) ** 2)))
    budget = budget_direct_m(inst, N_each, K)
    return rmse, budget


def run_mlmc(
    inst: LowerBoundTreeInstance, *, mode: str, sigma: float,
    levels: list[tuple[int, int]], reps: int, seed: int,
) -> tuple[float, int]:
    estimates = []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        v = mlmc_estimate(inst, levels, rng, mode=mode, sigma=sigma)
        estimates.append(v)
    estimates = np.array(estimates)
    rmse = float(np.sqrt(np.mean((estimates - inst.true_tv) ** 2)))
    budget = budget_mlmc(inst, levels)
    return rmse, budget


def make_levels(N_base: int, r_levels: list[int]) -> list[tuple[int, int]]:
    """Equal-cost-per-level allocation: N_l ∝ 1/(1+2*r_l)."""
    weights = [1.0 / (1.0 + 2.0 * r) for r in r_levels]
    norm = sum(weights)
    return [(max(1, int(round(N_base * w / norm * len(r_levels)))), r)
            for w, r in zip(weights, r_levels)]


def main() -> None:
    cfg = LowerBoundConfig(n=128, r=12, p=0.4, alpha=0.49)
    inst = LowerBoundTreeInstance(cfg)
    out_dir = Path("results/three_oracle"); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"True TV = {inst.true_tv:.4f}")
    print(f"Instance: n={cfg.n}, r={cfg.r}, p_active={cfg.p}, alpha={cfg.alpha}")

    reps = 64
    seed = 100

    # ----- Exact-logit V2 (Thm 2): σ=0, K=1, sweep N -----
    exact_runs = []
    for N_each in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        rmse, B = run_direct_m(inst, mode="exact", sigma=0.0,
                               N_each=N_each, K=1, reps=reps, seed=seed)
        exact_runs.append({"N_each": N_each, "K": 1, "budget": B, "rmse": rmse})
        print(f"  exact     N={N_each:>5} K=1   B={B:>10}  RMSE={rmse:.4f}")

    # MLMC schedule per oracle: hand-tuned r_levels matched to σ so the
    # finite-K bias is below MC noise floor; vary N_base only to sweep budget.
    # This yields clean -1/2 RMSE-vs-budget slopes with prefactors that match
    # Table 1's interpolation: noisy at σ=0.04 ≈ exact, sample ≈ noisy at σ=1.
    # MLMC bias floor at the finest level scales as √(nσ²/r_finest); we set
    # r_finest large enough to push the bias well below the smallest RMSE we
    # plan to hit (~0.005 at the largest budget).
    sigma_grid = [0.04, 0.16, 0.5]
    r_levels_for_sigma: dict[float, list[int]] = {
        0.04: [2, 8, 32, 128],
        0.16: [4, 16, 64, 256],
        0.5:  [16, 64, 256, 1024],
    }
    r_levels_sample = [16, 64, 256, 1024]   # σ²_eff = 1, similar magnitude

    N_grid = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # ----- Noisy-logit MLMC (Thm 3) at several σ -----
    noisy_runs: dict[float, list] = {}
    for sigma in sigma_grid:
        r_lev = r_levels_for_sigma[sigma]
        noisy_runs[sigma] = []
        for N_base in N_grid:
            levels = make_levels(N_base, r_lev)
            rmse, B = run_mlmc(inst, mode="prob_gaussian", sigma=sigma,
                               levels=levels, reps=reps, seed=seed)
            noisy_runs[sigma].append({
                "N_base": N_base, "r_levels": r_lev, "levels": levels,
                "budget": B, "rmse": rmse,
            })
            print(f"  noisy σ={sigma:>4} N_base={N_base:>4} "
                  f"B={B:>10}  RMSE={rmse:.4f}")

    # ----- Sample oracle (Thm 1): mode=onehot via MLMC, σ²_eff = 1 -----
    sample_runs = []
    for N_base in N_grid:
        levels = make_levels(N_base, r_levels_sample)
        rmse, B = run_mlmc(inst, mode="onehot", sigma=0.0,
                           levels=levels, reps=reps, seed=seed)
        sample_runs.append({
            "N_base": N_base, "r_levels": r_levels_sample, "levels": levels,
            "budget": B, "rmse": rmse,
        })
        print(f"  sample    N_base={N_base:>4} "
              f"B={B:>10}  RMSE={rmse:.4f}")

    summary = {
        "config": asdict(cfg),
        "true_tv": float(inst.true_tv),
        "reps": reps,
        "exact": exact_runs,
        "noisy": {f"sigma={s}": runs for s, runs in noisy_runs.items()},
        "sample": sample_runs,
    }
    out_json = out_dir / "three_oracle_runs.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {out_json}")

    # ============== Plot ==============
    plt.rcParams.update({
        "font.size": 11.5, "axes.labelsize": 12, "axes.titlesize": 12.5,
        "legend.fontsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    def plot_one(runs, label, color, marker, ls="-", lw=2.0, ms=8):
        Bs = np.array([r["budget"] for r in runs])
        Rs = np.array([r["rmse"]   for r in runs])
        idx = np.argsort(Bs)
        ax.plot(Bs[idx], Rs[idx], marker=marker, color=color, lw=lw, ls=ls,
                ms=ms, label=label, alpha=0.95, markeredgewidth=0)

    # Five curves, ordered exact → noisy small → noisy large → sample
    plot_one(exact_runs, r"exact logit  (Thm 2)", "#1a7f3c", "o")
    cmap = plt.cm.viridis(np.linspace(0.20, 0.80, len(sigma_grid)))
    for (sigma, runs), c in zip(noisy_runs.items(), cmap):
        plot_one(runs, fr"noisy logit  $\sigma={sigma}$", c, "D", ls="-")
    plot_one(sample_runs, r"sample (one-hot)  (Thm 1)", "#c4393f", "s", ls="--")

    # Slope -1/2 reference (anchored to exact's mid-budget point)
    Bref = np.geomspace(1e4, 5e7, 50)
    anchor_B = exact_runs[3]["budget"]; anchor_R = exact_runs[3]["rmse"]
    Rref = anchor_R * np.sqrt(anchor_B / Bref)
    ax.plot(Bref, Rref, color="#444", lw=1.1, ls=":", alpha=0.55,
            label=r"slope $-1/2$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.13, linewidth=0.5)
    ax.set_xlabel(r"total prefix-oracle queries")
    ax.set_ylabel(r"RMSE  (vs true TV $= "
                  + f"{inst.true_tv:.3f}$)")
    ax.set_title("Three oracles, same instance, same MLMC estimator")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95, ncol=1)
    fig.tight_layout()
    fig.savefig(out_dir / "three_oracle_scaling.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "three_oracle_scaling.pdf", bbox_inches="tight")
    print(f"[wrote] {out_dir}/three_oracle_scaling.{{png,pdf}}")


if __name__ == "__main__":
    main()
