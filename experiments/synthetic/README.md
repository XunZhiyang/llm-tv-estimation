# Synthetic experiments

CPU-only experiments on the softened lower-bound autoregressive instance, whose
true TV is known by enumeration. Paper instance: n=128, r=12, p_active=0.4,
α=0.49 → TV = 0.3883; master seed 10.

## Core libraries

- `lower_bound_instance.py` — `LowerBoundConfig` / `LowerBoundTreeInstance`:
  the frozen binary autoregressive π/μ pair with exact TV, sampling, exact
  log-probs, and a prefix oracle.
- `estimators.py` — noisy probability oracles (modes: exact / onehot /
  prob_gaussian / gaussian_logit), the one-sided and symmetric-mixture
  estimators, stratified MLMC, budget formulas.

## Paper-figure scripts (defaults reproduce the paper runs)

- `exact_logit_convergence.py` — runner+plotter for the exact-logit 1/√N sanity
  check (paper Fig 5) → `results/synthetic_exact_logit_p040/`.
- `plot_rho_regime_convergence.py` — simulation runner (despite the name):
  1/2/3/4-level MLMC vs. budget at σ ∈ {0.04, 0.5, 0.8} →
  `results/synthetic_rho_regime_p040_4lvl_3col/rho_regime_convergence.json`.
- `replot_rho_regime_paper.py` — renders paper Fig 6 from the committed JSON;
  no re-simulation.
