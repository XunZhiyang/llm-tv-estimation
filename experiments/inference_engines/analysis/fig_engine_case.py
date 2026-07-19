"""Fig 3 (engine case study) — matches the paper's existing aesthetic (clean sans,
large fonts, purple signature, soft fills, diamond markers + error bars).
Self-contained: computes the trajectory-level decomposition directly from the
canonical pools.
 (a) TV(n) for one prompt: support mismatch (KL = ∞) (gray, base)
     + shared-support log-prob difference (purple band) = TV (purple, error bars).
 (b) the same decomposition across six prompts at n=500.

TV is the symmetric debiased-mixture estimator; the trajectory-level split is
TV = E[one-sided] + E[tanh(|dlogp|/2) | shared], where a one-sided trajectory has
>=1 token one engine zeroed in its top-k (empirical KL = infinity).

Run (paper Fig 3, panel (a) from the reference v2 pools, grid to n=500):
      python -m experiments.inference_engines.analysis.fig_engine_case
Run (appendix length-sweep figure, single panel, n2000 campaign pools):
      python -m experiments.inference_engines.analysis.fig_engine_case \
          --pools n2000 --panel-a-only --out app_length_sweep
Data: results/vllm_vs_sglang_v2/{pi,mu}_pool.pt  (panel a, --pools v2, default)
      results/depth_2026-06/n2000/{pi,mu}_pool.pt  (panel a, --pools n2000)
      results/engine_multiprompt_2026-06/*_quick_decode_analysis.json  (panel b)
When the raw pools are absent (public code release), panel (a) is re-rendered from
the committed series results/report_2026-06/figdata_v2/fig3_panel_a_<pools>.json,
which this script writes on every from-pools run.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/report_2026-06/figures/paper"
EM = ROOT / "results/engine_multiprompt_2026-06"
FIGDATA = ROOT / "results/report_2026-06/figdata_v2"
POOL_DIRS = {
    "v2": ROOT / "results/vllm_vs_sglang_v2",
    "n2000": ROOT / "results/depth_2026-06/n2000",
}
GRIDS = {
    "v2": [10, 25, 50, 100, 200, 300, 400, 500],
    "n2000": [10, 25, 50, 100, 200, 300, 500, 750, 1000, 1250, 1500, 1750, 2000],
}

PURPLE, GRAY, INK = "#7d5ba6", "#9a9a9a", "#1a1a1a"


def lme(t, d):
    return torch.logsumexp(t, dim=d) - math.log(t.shape[d])


def side_Z(Lp, Lm, n):
    """per-trajectory DM contribution over first-n tokens, + one-sided mask."""
    La = Lp[:, :n].double().sum(1)
    Lb = Lm[:, :n].double().sum(1)
    fa, fb = torch.isfinite(La), torch.isfinite(Lb)
    both, one = fa & fb, fa ^ fb
    Z = torch.where(both, torch.tanh((La - Lb).abs() / 2), torch.zeros_like(La))
    Z = torch.where(one, torch.ones_like(La), Z)
    return Z.numpy(), one.numpy().astype(float)


def panel_a_data(pool_dir, grid):
    pi = torch.load(pool_dir / "pi_pool.pt", map_location="cpu", weights_only=False)
    mu = torch.load(pool_dir / "mu_pool.pt", map_location="cpu", weights_only=False)
    K = pi["meta"]["K_max"]
    Lp_pi, Lm_pi = lme(pi["score_L_pi"].float()[:, :K], 1), lme(pi["score_L_mu"].float()[:, :K], 1)
    Lp_mu, Lm_mu = lme(mu["score_L_pi"].float()[:, :K], 1), lme(mu["score_L_mu"].float()[:, :K], 1)
    TV, SH, MM, SE = [], [], [], []
    for n in grid:
        Zp, op = side_Z(Lp_pi, Lm_pi, n)
        Zm, om = side_Z(Lp_mu, Lm_mu, n)
        tv = 0.5 * Zp.mean() + 0.5 * Zm.mean()
        mm = 0.5 * op.mean() + 0.5 * om.mean()
        se = math.sqrt(0.25 * Zp.var() / len(Zp) + 0.25 * Zm.var() / len(Zm))
        TV.append(tv); MM.append(mm); SH.append(tv - mm); SE.append(se)
    return (np.array(grid), np.array(TV), np.array(SH), np.array(MM), np.array(SE))


def panel_a_series(pools, nmax=None):
    """Full-grid panel-(a) series: computed from the pools when present (and dumped
    to the committed figdata JSON), else re-loaded from that JSON."""
    jp = FIGDATA / f"fig3_panel_a_{pools}.json"
    if (POOL_DIRS[pools] / "pi_pool.pt").exists():
        n, TV, SH, MM, SE = panel_a_data(POOL_DIRS[pools], GRIDS[pools])
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps({"pools": pools, "n": n.tolist(), "TV": TV.tolist(),
                                  "shared": SH.tolist(), "mismatch": MM.tolist(),
                                  "se": SE.tolist()}, indent=1))
    else:
        d = json.loads(jp.read_text())
        n, TV, SH, MM, SE = (np.array(d[k]) for k in ("n", "TV", "shared", "mismatch", "se"))
        print(f"[panel a] pools not found; replotting from {jp}")
    if nmax is not None:
        keep = n <= nmax
        n, TV, SH, MM, SE = n[keep], TV[keep], SH[keep], MM[keep], SE[keep]
    return n, TV, SH, MM, SE


def panel_b_data():
    tags = [("fib", "code (Fibonacci)"), ("gd", "Chinese (grad. desc.)"), ("rj", "literature (R&J)"),
            ("prompt2", "explanation (fridge)"), ("frev", "history (Fr. Rev.)"), ("robot", "story (robot)")]
    rows = []
    for tag, lab in tags:
        e = json.load(open(EM / f"{tag}_quick_decode_analysis.json"))["tv_vs_n"][-1]
        tv, mm = e["TV"], e["mismatch"]
        rows.append((lab, tv, tv - mm, mm))           # label, TV, shared, mismatch
    return sorted(rows, key=lambda r: r[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pools", choices=sorted(POOL_DIRS), default="v2",
                    help="panel-(a) pool campaign (default: v2 reference pools)")
    ap.add_argument("--nmax", type=int, default=None,
                    help="truncate the panel-(a) grid at this n (default: full grid)")
    ap.add_argument("--panel-a-only", action="store_true",
                    help="render a single-panel figure (panel (a) only)")
    ap.add_argument("--out", default="fig3_engine_v2",
                    help="output file stem (default: fig3_engine_v2)")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 12.5, "axes.titlesize": 15, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 10.5,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
        "savefig.dpi": 150, "axes.linewidth": 1.0})
    if args.panel_a_only:
        fig, axA = plt.subplots(figsize=(6.3, 4.4))
        axB = None
    else:
        fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- (a) ----
    n, TV, SH, MM, SE = panel_a_series(args.pools, args.nmax)
    for ni, tvi, mmi, sei in zip(n, TV, MM, SE):
        print(f"[panel a | pools={args.pools}] n={ni:5d}  TV={tvi:.4f} +- {sei:.4f}  "
              f"mismatch={mmi:.4f}  shared={tvi - mmi:.4f}")
    axA.fill_between(n, 0, MM, color=GRAY, alpha=0.40, lw=0, label="support mismatch (KL = ∞)")
    axA.fill_between(n, MM, TV, color=PURPLE, alpha=0.26, lw=0, label="shared-support log-prob difference")
    axA.plot(n, MM, "-o", color=GRAY, lw=1.3, ms=4.5, mfc=GRAY, mec="white", mew=0.6)
    axA.errorbar(n, TV, yerr=SE, fmt="-D", color=PURPLE, lw=1.8, ms=7, mfc=PURPLE,
                 mec="white", mew=0.8, ecolor=PURPLE, elinewidth=1.2, capsize=2.5,
                 label=r"$\mathrm{TV}(\pi,\mu)$")
    n_max = int(n[-1])
    xticks = [t for t in ([0, 500, 1000, 1500, 2000] if n_max > 500 else [0, 100, 200, 300, 400, 500])
              if t <= n_max]
    axA.set_xlim(0, n_max); axA.set_ylim(0, 0.72); axA.set_xticks(xticks)
    axA.set_xlabel(r"sequence length  $n$"); axA.set_ylabel(r"$\mathrm{TV}(\pi,\mu)$")
    axA.set_title("TV vs sequence length" if args.panel_a_only else "(a)  TV vs sequence length")
    handles = [Patch(facecolor=GRAY, alpha=0.40, label="support mismatch (KL = ∞)"),
               Patch(facecolor=PURPLE, alpha=0.26, label="shared-support log-prob difference"),
               plt.Line2D([], [], color=PURPLE, marker="D", ms=7, lw=1.8, mec="white",
                          label=r"$\mathrm{TV}(\pi,\mu)$")]
    axA.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.92,
               edgecolor="0.8", handlelength=1.6)

    # ---- (b) ----
    rows = panel_b_data() if axB is not None else []
    if axB is not None:
        labs = [r[0] for r in rows]; sh = [r[2] for r in rows]; mm = [r[3] for r in rows]
        y = np.arange(len(labs))
        axB.barh(y, mm, color=GRAY, alpha=0.55, height=0.64, edgecolor="white", lw=0.7)
        axB.barh(y, sh, left=mm, color=PURPLE, alpha=0.80, height=0.64, edgecolor="white", lw=0.7)
        for i, (m, s) in enumerate(zip(mm, sh)):
            axB.text(m + s + 0.014, i, f"{m + s:.2f}", va="center", fontsize=11.5, color=INK)
            if m >= 0.06:
                axB.text(m / 2, i, f"{m:.2f}", va="center", ha="center", fontsize=10, color="white")
        axB.set_yticks(y); axB.set_yticklabels(labs)
        ci = labs.index("story (robot)"); axB.get_yticklabels()[ci].set_weight("bold")
        axB.set_xlim(0, 0.70); axB.set_ylim(-0.6, len(labs) - 0.4)
        axB.set_xlabel(r"$\mathrm{TV}(\pi,\mu)$  at  $n=500$")
        axB.set_title("(b)  TV across six prompts")
        axB.grid(axis="y", alpha=0)

    fig.tight_layout(w_pad=2.0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{args.out}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{args.out}.pdf", bbox_inches="tight")
    pfd = ROOT / "paper_final_draft/figures"
    if pfd.exists():   # paper tree, absent in the public code release
        fig.savefig(pfd / f"{args.out}.png", bbox_inches="tight")
        fig.savefig(pfd / f"{args.out}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{args.out}.png", "(+ paper copies)" if pfd.exists() else "")


if __name__ == "__main__":
    main()
