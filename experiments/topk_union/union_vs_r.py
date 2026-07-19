"""Union-size-vs-r curves from hammer caches (v3 fig 9, multi-model).

For sampled (trajectory, position) cells: |union of top-k indices over the
first r repeats| / k, averaged over cells — the tightness of the top-k
truncation model as a function of repeat depth.

  python -m experiments.topk_union.union_vs_r --cache inc_q06.pt --label q06 \
      --out union_q06.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_cells", type=int, default=1500)
    ap.add_argument("--ks", default="5,20,100")
    a = ap.parse_args()

    j = torch.load(a.cache, map_location="cpu", weights_only=False)
    idx_all = j["topk_idx"].numpy()
    N, K_max, n, top_save = idx_all.shape
    ks = [k for k in (int(x) for x in a.ks.split(",")) if k <= top_save]
    rs = [r for r in [1, 2, 4, 8, 16, 32, 64, 128, 248] if r <= K_max]
    rng = np.random.default_rng(0)
    ci = rng.integers(0, N, size=a.n_cells)
    ct = rng.integers(0, n, size=a.n_cells)
    res = {k: {r: 0.0 for r in rs} for k in ks}
    for p, t in zip(ci, ct):
        cell = idx_all[p, :, t, :]
        for k in ks:
            for r in rs:
                res[k][r] += len(np.unique(cell[:r, :k].ravel())) / k
    out = {"label": a.label, "N": N, "K_max": K_max, "n_cells": a.n_cells,
           "union_over_k": {str(k): {str(r): v / a.n_cells for r, v in d.items()}
                            for k, d in res.items()}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k in ks:
        print(f"[{a.label}] k={k}: " +
              " ".join(f"r={r}:{res[k][r]/a.n_cells:.3f}" for r in rs))


if __name__ == "__main__":
    main()
