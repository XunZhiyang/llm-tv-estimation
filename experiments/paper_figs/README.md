# Paper Figs 1–2: noisy-oracle characterization + denoising

`render_fig1_fig2.py` renders paper Figure 1 ("inference engines as noisy logit
oracles") and Figure 2 ("denoising separates true distance from oracle noise")
from the committed intermediate data in `results/report_2026-06/figdata_v2/` plus
`results/hugeR_floor2/analysis/hugeR_matched_floor.json`. Outputs go to
`results/report_2026-06/figures/v3/`. `render_seq_api_fig.py` renders the
appendix sequence-API self-scoring figure (measured values inlined in the script).

## Data generators (provenance only)

The remaining scripts generated the committed figdata; they ran on the original
compute environment (TACC Vista, GH200) against raw measurement caches that are
not distributed, so they are kept as provenance, not as runnable entry points.
Verified to reproduce the published figure values exactly.

| script | output (committed in `figdata_v2/`) | input caches |
|---|---|---|
| `def3_sigma_multimodel.py` | `def3_sigma.json` — σ (χ² half-split on the top-k union support, k∈{5,10,20,50,100}) per model × {cudnn, flash, math} × {isolated, served} | per-model repeated-query caches (5 Qwen models) |
| `extract_crossmodel_hist.py` | `crossmodel_cudnn_hist{,_ws}.npz` — Fig 1b single-call deviation histograms (`_ws` = served paths) | same caches |
| `fig2a_ladder.py` | `fig2a_ladder{,_ws}.npz` — Fig 2a per-token Δ histograms vs. a fixed cuDNN reference | Qwen3-0.6B repeated-query cache |
| `tv_vs_r_disambig.py` | `tv_vs_r_disambig{,_ws}.json` — Fig 2b TV-vs-r (self-noise floor / auto–cuDNN / math–FlashAttn-2) | SDPA pairwise cache (K=64) |

The `_ws` (warm-serve) variants used by the final figures were derived from the
base scripts by path substitution (`warm_iso` → `warm_serve`); the exact
substitutions are recorded in the module docstrings.

Sanity anchor: `def3_sigma.json` k=20 values reproduce the published Fig-1a bar
heights bar-for-bar (e.g. 0.6B cuDNN 0.045 isolated / 0.059 served; MoE flash
0.107 served).

Figure terminology: **isolated (fixed batch)** = query alone in a fixed batch;
**served (varying batch)** = query batched with varying other requests.
