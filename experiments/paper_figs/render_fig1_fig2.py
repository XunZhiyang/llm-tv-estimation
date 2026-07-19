"""Render paper/report figures 1 (noisy-oracle characterization) and 2
(denoising / identification) from the committed data files in
results/report_2026-06/figdata_v2/ (see README.md for provenance).

Identical to the 2026-06-24 rendering except the fig-1a condition legend, which
now reads "isolated (fixed batch)" / "+ served (varying batch)" (previously
"intrinsic (isolated batch)" / "+ serving batch").

  python -m experiments.paper_figs.render_fig1_fig2
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results/report_2026-06/figdata_v2"
OUT_REPORT = ROOT / "results/report_2026-06/figures/v3"
OUT_PAPER = ROOT / "paper_final_draft/figures"
OUT_REPORT.mkdir(parents=True, exist_ok=True)
HAVE_PAPER = OUT_PAPER.exists()   # paper tree, absent in the public code release

d3 = json.load(open(DATA / "def3_sigma.json"))
npz = np.load(DATA / "crossmodel_cudnn_hist_ws.npz")
floor = json.load(open(ROOT / "results/hugeR_floor2/analysis/hugeR_matched_floor.json"))
lad = np.load(DATA / "fig2a_ladder_ws.npz", allow_pickle=True)
dis = json.load(open(DATA / "tv_vs_r_disambig_ws.json"))

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 150, "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.linewidth": 0.9})
KC = {"cudnn": "#2ca02c", "flash": "#e8821a", "math": "#8156b5"}
BNAME = {"cudnn": "cuDNN", "flash": "FlashAttn-2", "math": "math"}
MODELS = ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-8B", "Qwen2.5-7B", "Qwen3-30B-MoE"]
short = ["0.6B", "1.7B", "8B", "Qwen2.5\n-7B", "30B\nMoE"]
mcol = {"Qwen3-0.6B": "#9ecae1", "Qwen3-1.7B": "#6baed6", "Qwen3-8B": "#2171b5",
        "Qwen2.5-7B": "#1f9aa3", "Qwen3-30B-MoE": "#c43d3d"}


def sig(m, reg, k):
    return float(d3[m][k][reg]["sigma_by_k"]["20"])


def fig1():
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.4, 3.7),
                                        gridspec_kw={"width_ratios": [1.22, 1, 1]})
    x = np.arange(len(MODELS)); w = 0.26
    for j, k in enumerate(["cudnn", "flash", "math"]):
        pos = x + (j - 1) * w
        serve = np.array([sig(m, "warm_serve", k) for m in MODELS])
        iso = np.array([sig(m, "warm_iso", k) for m in MODELS])
        axA.bar(pos, serve, w, color=KC[k], alpha=0.30, edgecolor=KC[k], lw=.6)
        axA.bar(pos, iso, w, color=KC[k], alpha=1.0, edgecolor="k", lw=.4)
    axA.set_xticks(x); axA.set_xticklabels(short)
    axA.set_ylim(0, 0.115)
    axA.set_ylabel(r"relative std. dev. $\sigma$  (Def. 3, top-$k{=}20$)")
    axA.set_title("(a)  noisy-oracle $\\sigma$ by backend")
    kh = [Patch(facecolor=KC[k], edgecolor="k", label=BNAME[k]) for k in ["cudnn", "flash", "math"]]
    lg1 = axA.legend(handles=kh, loc="upper left", frameon=False); axA.add_artist(lg1)
    axA.legend(handles=[Patch(facecolor="0.45", edgecolor="k", label="isolated (fixed batch)"),
                        Patch(facecolor="0.45", alpha=0.30, edgecolor="0.45",
                              label="+ served (varying batch)")],
               loc="upper center", frameon=False, fontsize=7.8)

    bins = npz["bins"]; ctr = 0.5 * (bins[1:] + bins[:-1])
    for key, lab in [("Qwen3_0p6B", "0.6B"), ("Qwen3_1p7B", "1.7B"), ("Qwen3_8B", "8B"),
                     ("Qwen2p5_7B", "Qwen2.5-7B"), ("Qwen3_30B_MoE", "30B-MoE")]:
        m = {"Qwen3_0p6B": "Qwen3-0.6B", "Qwen3_1p7B": "Qwen3-1.7B", "Qwen3_8B": "Qwen3-8B",
             "Qwen2p5_7B": "Qwen2.5-7B", "Qwen3_30B_MoE": "Qwen3-30B-MoE"}[key]
        axB.plot(ctr, npz[f"pc_{key}"], drawstyle="steps-mid", color=mcol[m], lw=1.3, label=lab)
    axB.axvline(0, color="k", ls=":", lw=.7, alpha=.5)
    axB.set_yscale("log"); axB.set_ylim(5e-2, 60); axB.set_xlim(-0.28, 0.28)
    axB.set_xlabel("single-call log-prob deviation (nat)")
    axB.set_ylabel("density (log scale)")
    axB.set_title("(b)  single-call deviation (cuDNN)")
    axB.legend(loc="upper right", frameon=False, fontsize=8.0)

    KS = [5, 10, 20, 50, 100]
    for m in MODELS:
        sbk = d3[m]["cudnn"]["warm_serve"]["sigma_by_k"]
        s2 = [float(sbk[str(k)]) ** 2 for k in KS]
        axC.loglog(KS, s2, "o-", color=mcol[m], ms=5, lw=1.6,
                   label=m.replace("Qwen3-", ""))
    axC.set_xlabel(r"API top-$k$ truncation")
    axC.set_ylabel(r"relative variance $\sigma^2$  (Def. 3)")
    axC.set_xticks(KS); axC.set_xticklabels(KS)
    axC.set_title(r"(c)  $\sigma^2$ vs. top-$k$")
    axC.legend(loc="upper right", frameon=False, fontsize=8.2)
    axC.grid(True, which="both", alpha=.22)
    fig.tight_layout(w_pad=1.4)
    fig.savefig(OUT_REPORT / "fig1_main.png", bbox_inches="tight")
    fig.savefig(OUT_REPORT / "fig1_main.pdf", bbox_inches="tight")
    if HAVE_PAPER:
        fig.savefig(OUT_PAPER / "fig1_noise.png", bbox_inches="tight")
        fig.savefig(OUT_PAPER / "fig1_noise.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2():
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.05]})
    bins = lad["bins"]; ctr = 0.5 * (bins[1:] + bins[:-1])
    def sl(i): return float(lad[f"s{i}"])
    axA.fill_between(ctr, lad["h0"], step="mid", color="0.6", alpha=.30, lw=0)
    axA.plot(ctr, lad["h0"], drawstyle="steps-mid", color="0.35", lw=1.5,
             label=fr"self: cuDNN vs. itself (std {sl(0):.3f})")
    axA.plot(ctr, lad["h1"], drawstyle="steps-mid", color=KC["cudnn"], lw=1.5, ls=(0, (4, 2)),
             label=fr"auto vs. cuDNN (std {sl(1):.3f})")
    axA.plot(ctr, lad["h2"], drawstyle="steps-mid", color="#d62728", lw=1.5,
             label=fr"isolated vs. served batch (std {sl(2):.3f})")
    axA.plot(ctr, lad["h3"], drawstyle="steps-mid", color=KC["flash"], lw=1.5,
             label=fr"FlashAttn-2 vs. cuDNN (std {sl(3):.3f})")
    axA.plot(ctr, lad["h4"], drawstyle="steps-mid", color=KC["math"], lw=1.5,
             label=fr"math vs. cuDNN (std {sl(4):.3f})")
    axA.axvline(0, color="k", ls=":", lw=.6, alpha=.5)
    axA.set_yscale("log"); axA.set_ylim(3e-2, 60); axA.set_xlim(-.22, .22)
    axA.set_xlabel(r"per-token $\Delta_t=\log\hat\pi_t-\log\hat\mu_t$ (nat),  ref $\mu=$ cuDNN (served)")
    axA.set_ylabel("density (log scale)")
    axA.set_title("(a)  per-token $\\Delta$ against a fixed reference")
    axA.legend(loc="upper left", frameon=False, fontsize=7.6)

    ws = floor["regimes"]["warm_serve"]; sfl = ws["self_floor"]["hf-auto"]; cx = ws["cross_auto_cudnn"]
    r = np.array([p["r"] for p in sfl]); fl = np.array([p["floor"] for p in sfl])
    cr = np.array([p["cross_TV"] for p in cx])
    rr = np.array(dis["r"]); mf = np.array(dis["math_flash"]); plat = float(mf[-1])
    axB.loglog(r, fl, "o-", color="#777", ms=3.2, lw=1.0, label="self-noise floor (cuDNN vs. itself)")
    axB.loglog(r, cr, "s-", color=KC["cudnn"], ms=3.6, lw=1.4,
               label=r"auto vs. cuDNN")
    axB.loglog(rr, mf, "^-", color="#c43d3d", ms=4.2, lw=1.6,
               label=r"math vs. FlashAttn-2")
    axB.loglog([rr[-1], 3000], [plat, plat], "--", color="#c43d3d", lw=1.1, alpha=.6)
    axB.set_xlabel(r"repetitions $r$")
    axB.set_ylabel(r"one-sided TV estimate at $n=500$")
    axB.set_ylim(0.004, 0.55)
    axB.set_title("(b)  one-sided TV estimate vs. repetitions")
    axB.legend(loc="lower left", frameon=False, fontsize=8.4)
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT_REPORT / "fig2_denoise.png", bbox_inches="tight")
    fig.savefig(OUT_REPORT / "fig2_denoise.pdf", bbox_inches="tight")
    if HAVE_PAPER:
        fig.savefig(OUT_PAPER / "fig2_denoise.png", bbox_inches="tight")
        fig.savefig(OUT_PAPER / "fig2_denoise.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    print("wrote", OUT_REPORT / "fig1_main.png",
          "and paper fig1/fig2 png+pdf" if HAVE_PAPER else "(+ fig2 png/pdf)")
