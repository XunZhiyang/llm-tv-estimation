"""Replot three_oracle from cached JSON without re-running the experiment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    out_dir = Path("results/three_oracle")
    data = json.load(open(out_dir / "three_oracle_runs.json"))
    true_tv = data["true_tv"]

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

    plot_one(data["exact"], r"exact logit  (Thm 2)", "#1a7f3c", "o")
    sigma_keys = [k for k in data["noisy"]]
    sigmas = sorted(float(k.split("=")[1]) for k in sigma_keys)
    cmap = plt.cm.viridis(np.linspace(0.20, 0.80, len(sigmas)))
    for sigma, c in zip(sigmas, cmap):
        runs = data["noisy"][f"sigma={sigma}"]
        plot_one(runs, fr"noisy logit  $\sigma={sigma}$", c, "D", ls="-")
    plot_one(data["sample"], r"sample (one-hot)  (Thm 1)", "#c4393f", "s", ls="--")

    Bref = np.geomspace(1e4, 5e7, 50)
    anchor = data["exact"][3]
    Rref = anchor["rmse"] * np.sqrt(anchor["budget"] / Bref)
    ax.plot(Bref, Rref, color="#444", lw=1.1, ls=":", alpha=0.55,
            label=r"slope $-1/2$")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, which="major", alpha=0.28, linewidth=0.7)
    ax.grid(True, which="minor", alpha=0.13, linewidth=0.5)
    ax.set_xlabel(r"total prefix-oracle queries")
    ax.set_ylabel(rf"RMSE  (vs true TV $= {true_tv:.3f}$)")
    ax.set_title("Three oracles, same instance, same MLMC estimator")
    ax.legend(loc="lower left", fontsize=10, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_dir / "three_oracle_scaling.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "three_oracle_scaling.pdf", bbox_inches="tight")
    print(f"[wrote] {out_dir}/three_oracle_scaling.{{png,pdf}}")


if __name__ == "__main__":
    main()
