"""Replot the synthetic rho-regime figure from cached JSON, with notation aligned to §5.1.

Reads the rows / rho_rows produced by plot_rho_regime_convergence.py and renders
a paper-ready 1xK figure. Does NOT re-run any simulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = [
    ("one_level_K256",       "single-level, $r = 256$",                 "#4e79a7", "o"),
    ("two_level_32_256",     "$2$-level, $(r_0, r_1) = (32, 256)$",     "#f28e2b", "s"),
    ("three_level_8_32_256", "$3$-level, $(r_0, r_1, r_2) = (8, 32, 256)$",  "#d62728", "*"),
    ("four_level_1_8_32_256","$4$-level, $(r_0, r_1, r_2, r_3) = (1, 8, 32, 256)$",  "#59a14f", "D"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_json",
                   default="results/synthetic_rho_regime_p040_4lvl_3col/rho_regime_convergence.json")
    p.add_argument("--out_pdf",
                   default="results/synthetic_rho_regime_p040_4lvl_3col/synthetic_rho_regime.pdf")
    p.add_argument("--out_png",
                   default="results/synthetic_rho_regime_p040_4lvl_3col/synthetic_rho_regime.png")
    args = p.parse_args()

    data = json.load(open(args.input_json))
    rows = data["rows"]
    rho_rows = data["rho_rows"]
    sigmas = data["config"]["sigmas"]

    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    })

    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.4 * len(sigmas), 4.2),
                             sharey=False)
    if len(sigmas) == 1:
        axes = [axes]

    rho_keys_in_order = ["rho_1_8", "rho_8_32", "rho_32_256"]

    for col, sigma in enumerate(sigmas):
        ax = axes[col]
        rho = next(r for r in rho_rows if r["sigma"] == sigma)
        rho_vals = [rho[k] for k in rho_keys_in_order if k in rho]
        rho_strs = [rf"$\rho_{i+1} = {v:.3f}$" for i, v in enumerate(rho_vals)]
        title = (rf"$\sigma = {sigma:g}$" + "\n" + ",  ".join(rho_strs))
        ax.set_title(title)

        for key, label, color, marker in METHODS:
            mr = sorted(
                [r for r in rows if r["sigma"] == sigma and r["method"] == key],
                key=lambda x: x["budget"],
            )
            if not mr: continue
            ax.errorbar(
                [r["budget"] for r in mr],
                [r["abs_error_mean"] for r in mr],
                yerr=[r["abs_error_se"] for r in mr],
                color=color, marker=marker,
                linewidth=1.9, markersize=6.2, capsize=2.4,
                label=label if col == len(sigmas) - 1 else None,
            )

        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("oracle-call budget")
        if col == 0:
            ax.set_ylabel("mean absolute error")

    axes[-1].legend(loc="upper right", framealpha=0.9,
                    handlelength=1.6, borderpad=0.35, labelspacing=0.3)
    fig.tight_layout()

    out_pdf = Path(args.out_pdf); out_png = Path(args.out_png)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    pfd = Path("paper_final_draft/figures")
    if pfd.exists():   # active paper tree, absent in the public code release
        fig.savefig(pfd / "synthetic_rho_regime.pdf", bbox_inches="tight")
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
