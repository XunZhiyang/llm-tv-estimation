# Total Variation Distance Estimation in Autoregressive Models

Code and data for the paper *Total Variation Distance Estimation in Autoregressive
Models* (Eric Price, Kevin Tian, Zhiyang Xun, Yusong Zhu).

The paper studies estimating the total variation (TV) distance between two
autoregressive distributions (e.g., two LLM serving stacks with identical weights)
under three access models: sample access, logit access, and noisy logit access.
This repository contains the estimators, the synthetic validation instance, the
measurement infrastructure for HuggingFace and serving-engine oracles, the
cross-engine (vllm vs. sglang) case-study pipeline, and the data and scripts that
reproduce the paper's figures.

## Repository layout

| Path | Contents |
|---|---|
| `experiments/oracle.py` | `HFOracle`: batched KV-cache sampling/scoring for HF models, top-k masking, SDPA attention-backend forcing |
| `experiments/tv_estimate.py` | model loading, log-likelihood helpers, baseline estimators |
| `experiments/fig1/` | pool-cache library: build resumable (π, μ) sample/score pools, estimator implementations (one-sided, symmetric-mixture, coupled MLMC). The name is historical; this is the core infrastructure. |
| `experiments/synthetic/` | CPU synthetic hard instance with true TV known by enumeration; fully reproducible from scratch |
| `experiments/inference_engines/` | vllm/sglang oracles, teacher-forced replay scoring, the phase pipeline (`phase_sample` / `phase_score` / `phase_combine`), and figure/analysis scripts under `analysis/` |
| `experiments/topk_union/` | top-k support-union measurement and σ² calibration |
| `experiments/paper_figs/` | renderers for the noise-characterization figures + the (provenance-only) generators of their intermediate data |
| `scripts/`, `slurm/` | engine venv setup and the production pipeline driver (Slurm templates) |
| `results/` | committed intermediate data: every number plotted in the paper's figures |

## Installation

Analysis and figure rendering (CPU only):

```bash
pip install -r requirements.txt
```

The cross-engine case study additionally needs the two serving engines. They pin
conflicting dependencies, so they live in two separate venvs:

```bash
bash scripts/setup_vllm_venv.sh      # vllm 0.19.1  (paper version)
bash scripts/setup_sglang_venv.sh    # sglang 0.5.10.post1  (paper version)
```

Paper measurement environment (Appendix B.1): NVIDIA GH200 96GB (TACC Vista),
gcc 14.2.0 / CUDA 12.8 / python 3.11.8, bf16. Per-venv versions: vllm 0.19.1 with
torch 2.10.0+cu128 and transformers 5.7.0; sglang 0.5.10.post1 with torch
2.9.1+cu128 and transformers 5.3.0; the HuggingFace attention-backend experiments
use torch 2.10.0+cu128 with transformers 4.57.6. On aarch64,
`scripts/fix_torch_v4.sh` pins the nvidia wheel versions to torch 2.10.0's
requirements (the vllm venv) if the engine install pulled incompatible ones.

## Reproducing the paper figures

All main-text and appendix figures re-render from the committed data in
`results/` — no GPU or raw pools needed:

| Paper figure | Command |
|---|---|
| Fig 1 (noisy-oracle σ) + Fig 2 (denoising) | `python -m experiments.paper_figs.render_fig1_fig2` |
| Fig 3 (case study: TV vs. length; six prompts) | `python -m experiments.inference_engines.analysis.fig_engine_case` |
| Fig 4 (multilevel budget curves) | `python -m experiments.inference_engines.analysis.mlmc_budget_curves --replot` |
| Fig 5 (exact-logit 1/√N convergence) | `python -m experiments.synthetic.exact_logit_convergence` (re-runs the CPU simulation; the committed JSON is the paper run) |
| Fig 6 (synthetic MLMC regimes) | `python -m experiments.synthetic.replot_rho_regime_paper` |
| Fig 7 (top-k union, five models) | `python -m experiments.topk_union.plot_union_multimodel` |
| Fig 8 (sequence-API self-scoring) | `python -m experiments.paper_figs.render_seq_api_fig` |

Outputs are written under `results/report_2026-06/figures/`, except Fig 5 and
Fig 6, which write into their own directories
(`results/synthetic_exact_logit_p040/`, `results/synthetic_rho_regime_p040_4lvl_3col/`);
their re-run/re-simulation commands also rewrite the committed JSON there in
place (the defaults reproduce the committed paper runs). Notes:

- Fig 3 / Fig 4 renderers read the raw score pools when present and otherwise
  fall back to the committed per-figure series
  (`results/report_2026-06/figdata_v2/fig3_panel_a_*.json`,
  `figdata` inside `results/mlmc_seq_2026-07/budget_curves.json`) — both paths
  produce identical figures.
- Fig 6 re-simulation: `python -m experiments.synthetic.plot_rho_regime_convergence`
  (CPU, defaults reproduce the paper run; master seed 10).
- The appendix sequence-length numbers reproduce via
  `python -m experiments.inference_engines.analysis.fig_engine_case --pools n2000 --panel-a-only --out app_length_sweep`.
- The σ² calibration of Appendix B.3 is `experiments/topk_union/sigma_squared_vs_K.py`;
  its output for the paper is committed as `results/topk_union/sigma_squared_vs_K.json`.
- The replay-oracle validation numbers of Appendix B.5 are committed as
  `results/vllm_vs_sglang_v2/{quick_decode_analysis,v2_oracle_comparison}.json`.
- The synthetic access-model comparison of Appendix B.11 (exact logit vs. noisy
  logit vs. sample access) re-renders via
  `python -m experiments.synthetic.replot_three_oracle` from the committed
  `results/three_oracle/three_oracle_runs.json`
  (runner: `experiments/synthetic/three_oracle_comparison.py`).
- Appendix Figs 9–10 are carried as images in the paper source (drawn from an
  internal measurement report) and are not regenerated here.

## Running the estimators yourself

**Synthetic instance (CPU, minutes).** `experiments/synthetic/` is self-contained:
the instance's true TV is computed by enumeration, so estimator error is exact.
See the module docstrings of `plot_rho_regime_convergence.py` and
`exact_logit_convergence.py`.

**Two HF configurations (one GPU).** `experiments/fig1/run_fig1.py` builds
sample/score pools for any pair of `HFOracle` configurations of one shared model
(per-side dtype/quantization, attention implementation, SDPA backend) and runs
the estimator family over them.

**Two serving engines (the case-study pipeline).** The paper's vllm-vs-sglang
measurement is `scripts/run_decode_production.sh` (Slurm template:
`slurm/decode_v2_main.slurm`): sample N trajectories from each engine, score both
pools under both engines with the teacher-forced replay oracle, combine, and
analyze. Engine configuration matches sampling: bf16, `temperature=1.0`,
`top_k=20` applied by the engine itself, `ignore_eos=True`; vllm with
`enforce_eager=True` and prefix caching off, sglang with CUDA graphs off
(each forward must draw fresh kernel noise).

## Measurement caveats

These matter if you extend the measurements; they are discussed in the paper's
appendix:

- **Repeat-axis hygiene.** Serving stacks can be serially correlated along the
  repeat axis (e.g., an engine-level batching period). All repeat-axis variance
  statistics in the analysis scripts therefore apply an independent uniform
  permutation of the repeats at each (trajectory, position) cell before any
  half-split ("per-cell shuffle"); computing on the natural serving order is not
  valid.
- **Serving-environment dependence.** The measured cross-engine TV depends on the
  serving configuration (batch size at scoring, admission behavior, top-k
  truncation, prompt). Values from different campaigns are comparable only within
  the same serving environment; the paper reports this dependence explicitly.
- **Sampling validity.** Estimates are only valid if trajectories are sampled at
  `temperature=1.0` with the oracle's own top-k masking. Do not compute sequence
  log-likelihoods via `model(..., labels=...)` (it returns a batch-averaged
  scalar); use the oracle scoring paths.

## Data

`results/` holds the committed intermediate data behind every figure (JSON/npz,
a few MB total). The raw sample/score pools (`*.pt`, ~GBs) are not distributed
with the repository; the pipeline above regenerates equivalent pools, and the
paper's original pools are archived by the authors — contact us if you need them.
`run_info` blocks inside the JSON records carry the git SHA of the private
development repository at run time; those SHAs do not resolve in this repository
and are kept as provenance only.

## License

MIT (see `LICENSE`).

## Citation

```bibtex
@article{PTXZ2026tv,
  title   = {Total Variation Distance Estimation in Autoregressive Models},
  author  = {Price, Eric and Tian, Kevin and Xun, Zhiyang and Zhu, Yusong},
  year    = {2026},
  journal = {arXiv preprint}
}
```
