"""Render report v3 fig 9: top-k union tightness across five models.

Reads the union-vs-r JSONs produced by experiments.topk_union.union_vs_r
(results/figruns_2026-07/union/union_*.json) and plots |union of top-k
indices|/k against repeat count r at k=20.

  python -m experiments.topk_union.plot_union_multimodel
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path("results/figruns_2026-07")
OUT = Path("results/report_2026-06/figures/v3/fig9_union_multimodel.png")

MODELS = [  # (file, display label, color)
    ("union/union_q06.json", "Qwen3-0.6B", "#3b0f70"),
    ("union/union_q17.json", "Qwen3-1.7B", "#414487"),
    ("union/union_q25.json", "Qwen2.5-7B", "#2a9d8f"),
    ("union/union_q8b.json", "Qwen3-8B", "#22726b"),
    ("union/union_moe30.json", "Qwen3-30B-A3B (MoE)", "#8bc34a"),
]


def curve(path):
    d = json.load(open(BASE / path))
    pts = sorted((int(r), v) for r, v in d["union_over_k"]["20"].items())
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=150)
    for path, label, color in MODELS:
        rs, vs = curve(path)
        ax.plot(rs, vs, "o-", ms=4, lw=1.6, color=color, label=label)
    ax.axhline(2.0, color="0.6", ls=":", lw=1)
    ax.text(1.15, 2.02, "2k", color="0.5", fontsize=9, style="italic")
    ax.axhline(1.0, color="0.85", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.93, 2.1)
    ax.set_xlabel("repeated queries $r$")
    ax.set_ylabel(r"|union of top-$k$ indices| / $k$   ($k$=20)")
    ax.set_title("top-$k$ support union under repeated queries (five models)",
                 fontsize=11)
    ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    pfd = Path("paper_final_draft/figures")
    if pfd.exists():   # paper tree, absent in the public code release
        fig.savefig(pfd / "app_topk_union_multimodel.pdf", bbox_inches="tight")
        fig.savefig(pfd / "app_topk_union_multimodel.png", bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
