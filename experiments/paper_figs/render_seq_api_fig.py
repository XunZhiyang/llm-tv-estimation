"""Render the appendix figure comparing self-scoring faithfulness of the replay
oracle and the sequence-level log-probability API (paper Appendix B.5).

Values are the acceptance measurements of the 2026-06-12 cross-engine campaign
:
fraction of trajectories whose own sampled token receives zero mass ---
replay: 0 / 0; sequence API: 0.6% (vllm) / 37% (sglang); net effect on the
estimated TV at n=500: 0.586 (replay) -> 0.550 (sequence API).

  python -m experiments.paper_figs.render_seq_api_fig
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_final_draft/figures"
if not OUT.exists():   # public code release: no paper tree
    OUT = ROOT / "results/report_2026-06/figures/paper"
    OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 9.5, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    "savefig.dpi": 150, "axes.linewidth": 0.9})

REPLAY = [0.0, 0.0]
SEQAPI = [0.006, 0.372]

fig, ax = plt.subplots(figsize=(5.6, 3.6))
x = np.arange(2)
w = 0.32
ax.bar(x - w / 2, REPLAY, w, color="#8156b5", label="replay oracle (teacher-forced)")
ax.bar(x + w / 2, SEQAPI, w, color="0.65", label="sequence-logprob API")
for xi, v in zip(x - w / 2, REPLAY):
    ax.text(xi, 0.005, "0", ha="center", fontsize=10)
for xi, v in zip(x + w / 2, SEQAPI):
    ax.text(xi, v + 0.008, f"{v:.1%}" if v >= 0.01 else f"{v:.1%}", ha="center", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(["vllm", "sglang"])
ax.set_ylim(0, 0.45)
ax.set_ylabel("fraction of trajectories with zero mass\non the engine's own sampled token")
ax.text(0.98, 0.55, "net effect on estimated TV ($n=500$):\nreplay 0.586 $\\to$ sequence API 0.550",
        transform=ax.transAxes, ha="right", fontsize=9, color="0.35")
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(OUT / "app_seq_api_selfscore.png", bbox_inches="tight")
fig.savefig(OUT / "app_seq_api_selfscore.pdf", bbox_inches="tight")
print("wrote", OUT / "app_seq_api_selfscore.pdf")
