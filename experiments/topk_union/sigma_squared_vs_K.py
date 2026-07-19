"""sigma^2(K, top-k) calibration from the topk_union incremental cache.

Estimates the within-engine noisy-oracle parameter sigma^2 (Definition 3) on
Qwen3-0.6B (bf16, sdpa) using the K_max=248 repeat-forward cache, and compares
it to the effective support size K_true. Two outputs:

  (1) sigma^2(K) decay: held-out chi^2 between two disjoint K-iter halves,
      with K_avg sweeping the first half. Confirms Lemma 6's 1/K shape.
  (2) sigma^2 vs K_true: at K=1 (single forward), sigma^2 << K_true for every
      top-k truncation, so the noisy-oracle bound (n + n^2 sigma^2)/eps^2
      strictly improves over the sample bound n^2 K/eps^2.

CPU-only. Subsamples cells for tractable runtime.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def per_iter_truncated_prob(
    idx_iter: np.ndarray, val_iter: np.ndarray, k: int,
    tok_to_pos: dict[int, int], support_size: int,
) -> np.ndarray:
    """Top-k truncated softmax for one (cell, iter), expanded to union support."""
    toks = idx_iter[:k]
    logits = val_iter[:k].astype(np.float64)
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    out = np.zeros(support_size, dtype=np.float64)
    for j, t in enumerate(toks):
        out[tok_to_pos[int(t)]] += p[j]
    return out


def chi2_from_avgs(p_a: np.ndarray, p_b: np.ndarray, p_denom: np.ndarray) -> float:
    """Half-split chi^2 (p_a - p_b)^2 / p_denom over the union support.

    p_denom is the full-K_max mean, which is strictly positive on the union
    support (every union token appears in >=1 iteration), so no term is
    dropped. The previous version used p_b itself as the denominator with a
    p_b > 1e-12 mask, silently censoring support-mismatch terms (token present
    in the A half but never in the B half); at k=5 that biased sigma^2 down by
    ~20-25%.
    """
    return float(np.sum((p_a - p_b) ** 2 / p_denom))


def main() -> None:
    cache_path = Path("results/topk_union/incremental_qwen3_0p6b.pt")
    cache = torch.load(cache_path, weights_only=False)
    idx_all = cache["topk_idx"].numpy()           # (N, K_max, n, top_save)
    val_all = cache["topk_val"].float().numpy()
    N, K_max, n, top_save = idx_all.shape
    print(f"[load] N={N} K_max={K_max} n={n} top_save={top_save}")

    out_dir = Path("results/topk_union")

    # K_avg sweep over the FIRST half; reference is the SECOND half (disjoint)
    K_half = K_max // 2                                       # 124
    ref_start = K_half
    K_grid = sorted(set([1, 2, 4, 8, 16, 32, 64, K_half]))
    top_ks = [5, 10, 20, 50, 100]

    n_cells = 1500
    rng = np.random.default_rng(0)
    cell_pi = rng.integers(0, N, size=n_cells)
    cell_it = rng.integers(0, n, size=n_cells)

    chi2_records = {k: {K: [] for K in K_grid} for k in top_ks}
    K_true_records = {k: [] for k in top_ks}

    for c, (ip, it) in enumerate(zip(cell_pi, cell_it)):
        if c % 200 == 0:
            print(f"  cell {c}/{n_cells}")
        idx_cell = idx_all[ip, :, it, :]                       # (K_max, top_save)
        val_cell = val_all[ip, :, it, :]

        for k in top_ks:
            union = np.unique(idx_cell[:, :k].ravel())
            K_true_records[k].append(union.size)
            tok_to_pos = {int(t): j for j, t in enumerate(union)}
            S = union.size

            # Pre-compute per-iter probs only once per k
            iter_probs = np.zeros((K_max, S), dtype=np.float64)
            for i in range(K_max):
                iter_probs[i] = per_iter_truncated_prob(
                    idx_cell[i], val_cell[i], k, tok_to_pos, S
                )

            p_ref = iter_probs[ref_start:].mean(axis=0)        # (S,)
            p_full = iter_probs.mean(axis=0)                   # (S,) > 0 on union
            for K_avg in K_grid:
                p_avg = iter_probs[:K_avg].mean(axis=0)
                chi2_records[k][K_avg].append(chi2_from_avgs(p_avg, p_ref, p_full))

    chi2_mean = {
        k: {K: float(np.mean(v)) for K, v in chi2_records[k].items()}
        for k in top_ks
    }
    K_true_mean = {k: float(np.mean(K_true_records[k])) for k in top_ks}

    # Decompose: E[chi^2(p_A, p_B)] = sigma^2 * (1/K_avg + 1/K_half)
    # => sigma^2_K = chi2 / (1/K_avg + 1/K_half)
    sigma2_per_K = {
        k: {K: chi2_mean[k][K] / (1.0 / K + 1.0 / K_half) for K in K_grid}
        for k in top_ks
    }

    summary = {
        "cache": str(cache_path),
        "N": N, "K_max": K_max, "n": n, "top_save": top_save,
        "K_half_reference": K_half,
        "K_grid_avg": K_grid,
        "top_ks": top_ks,
        "n_cells_sampled": n_cells,
        "chi2_mean_per_cell": chi2_mean,
        "sigma2_estimate_per_K": sigma2_per_K,
        "K_true_mean": K_true_mean,
    }
    # New filename: keep the submitted-version (censored) JSON intact as the record.
    out_json = out_dir / "sigma_squared_vs_K_uncensored.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[wrote] {out_json}")

    # ------------- Plot 1: sigma^2 decay vs K_avg -------------
    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 11.5, "axes.titlesize": 12,
        "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(top_ks)))
    K_arr = np.array(K_grid, dtype=float)

    for k, c in zip(top_ks, colors):
        ys = np.array([chi2_mean[k][K] for K in K_grid])
        ax.plot(
            K_arr, ys, marker="o", color=c, lw=1.7, ms=6.5,
            label=fr"top-$k={k}$",
        )

    K_ref = np.geomspace(1, K_half, 60)
    chi2_top20_K1 = chi2_mean[20][1]
    pred = chi2_top20_K1 * (1.0 / K_ref + 1.0 / K_half) / (1.0 / 1 + 1.0 / K_half)
    ax.plot(
        K_ref, pred, "--", color="gray", lw=1.4, alpha=0.75,
        label=r"$\sigma^2(1/K_{\rm avg} + 1/K_{\rm half})$ fit (top-20)",
    )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(r"$K_{\rm avg}$  (number of averaged forward passes, first half)")
    ax.set_ylabel(r"$\chi^2(\hat p_{K_{\rm avg}} \,\|\, \hat p_{\rm ref})$"
                  fr"   reference $=$ avg of last {K_half} iters")
    ax.set_title(
        r"Within-engine $\sigma^2$ decays as $1/K_{\rm avg}$"
        + "\n" + r"Qwen3-0.6B (bf16, sdpa) — empirical confirmation of Lemma 6"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_squared_K_decay_uncensored.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "sigma_squared_K_decay_uncensored.pdf", bbox_inches="tight")
    print(f"[wrote] {out_dir}/sigma_squared_K_decay.{{png,pdf}}")

    # ------------- Plot 2: sigma^2 (single forward) vs K_true -------------
    fig, ax = plt.subplots(figsize=(7.6, 5.4))

    sigma2_K1 = np.array([sigma2_per_K[k][1] for k in top_ks])
    K_true_arr = np.array([K_true_mean[k] for k in top_ks])

    ax.plot(
        top_ks, K_true_arr, marker="s", color="#3a6ea5", lw=2.0, ms=9,
        label=r"$K_{\rm true}$  (union of top-$k$ over $K_{\max}=248$ iters)",
    )
    ax.plot(
        top_ks, sigma2_K1, marker="o", color="#c4393f", lw=2.0, ms=9,
        label=r"$\sigma^2$  (single forward, Def. 3)",
    )

    for k, s, K_t in zip(top_ks, sigma2_K1, K_true_arr):
        if s > 0:
            # annotate just above the K_true (blue) marker
            ax.annotate(
                fr"$K_{{\rm true}}/\sigma^2 = {K_t / s:.0f}\times$",
                xy=(k, K_t),
                xytext=(0, 14), textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9.0, color="#333",
                bbox={"facecolor": "white", "edgecolor": "#cccccc",
                      "alpha": 0.95, "pad": 2.0, "boxstyle": "round,pad=0.28"},
            )

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_xlabel(r"top-$k$ truncation")
    ax.set_ylabel(r"$K_{\rm true}$ (token count)   /   $\sigma^2$ ($\chi^2$ units)")
    ax.set_title(
        r"$\sigma^2 \ll K_{\rm true}$ for every top-$k$ choice"
        + "\n" + r"noisy bound $(n + n^2\sigma^2)/\epsilon^2$ "
        + r"vs sample bound $n^2 K/\epsilon^2$"
    )
    ax.set_xticks(top_ks)
    ax.set_xticklabels([str(k) for k in top_ks])
    ax.set_ylim(8e-4, 4e2)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32),
              ncol=2, frameon=True, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_squared_vs_K_true_uncensored.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "sigma_squared_vs_K_true_uncensored.pdf", bbox_inches="tight")
    print(f"[wrote] {out_dir}/sigma_squared_vs_K_true.{{png,pdf}}")

    print("\n=== sigma^2 estimates (single forward, K_avg=1) ===")
    for k in top_ks:
        print(f"  top-{k:>3}: sigma^2 = {sigma2_per_K[k][1]:.4e}, "
              f"K_true = {K_true_mean[k]:.2f}, "
              f"ratio K_true/sigma^2 = {K_true_mean[k]/max(sigma2_per_K[k][1], 1e-12):.1f}")


if __name__ == "__main__":
    main()
