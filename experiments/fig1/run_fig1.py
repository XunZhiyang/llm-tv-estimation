"""Fig 1 driver: build (pi_pool, mu_pool) shared cache; run 6 estimators.

Self-TV setup: π = μ = Qwen3-0.6B SDPA bf16 top_k=20 (two independent oracle
instances). Ground truth TV = 0.

Cache cost (default K_max=8, N_pi=100, N_mu=50):
  pi_pool: 1 sample (B=100) + 16 score = 17 batched calls
  mu_pool: 1 sample (B=50)  + 16 score = 17 batched calls
  ~9 min wall (140s warmup + 34 × 13s).

Estimators:
  Method 1 fused, Method 5 fused             — 1 point each (budget ~10⁵)
  Method 4 π-direct K, K=1,2,4,8            — 4 points
  Direct-M K (mixture, no MLMC), K=1,2,4,8  — 4 points
  Method 3 MLMC logit, L=0..3 (4 cumulative levels)
  Method 2 MLMC one-hot, L=0..3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from experiments.oracle import HFOracle
from experiments.fig1.cache_build import build_pool, build_pools_balanced, tokenize_prompt
from experiments.fig1.methods import (
    method1_fused, method5_fused, method4_pi_direct_K,
    direct_M_K, mlmc,
)


def parse_levels(s: str) -> list[tuple[int, int]]:
    """Parse 'N0:r0,N1:r1,...' → [(N0, r0), ...]."""
    out = []
    for tok in s.split(","):
        N_str, r_str = tok.split(":")
        out.append((int(N_str), int(r_str)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--user", default="Tell me a story about a robot learning to paint.")
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--top_k", type=int, default=20)

    p.add_argument("--N_pi", type=int, default=100, help="pi_pool size")
    p.add_argument("--N_mu", type=int, default=50,  help="mu_pool size")
    p.add_argument("--K_max", type=int, default=8, help="fresh score evals per side per traj")

    p.add_argument("--K_sweep", default="1,2,4,8",
                   help="K values for Method4 / Direct-M (comma-sep)")
    p.add_argument("--N_each_directM", type=int, default=50,
                   help="N per side for Direct-M K (uses pi_pool[:N], mu_pool[:N])")
    p.add_argument("--mlmc_levels", default="24:1,14:2,8:4,4:8",
                   help="MLMC levels as N_per_side:r per level (disjoint slices, "
                        "ordered low→high r). Sum N_per_side ≤ N_pi and ≤ N_mu.")

    p.add_argument("--output_dir", default="results/fig1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--quant_pi", default="bf16",
                   help="Dtype/quant for π oracle (bf16/fp16/int8/nf4/fp4)")
    p.add_argument("--quant_mu", default="bf16",
                   help="Dtype/quant for μ oracle. Default = π for self-TV; "
                        "set differently (e.g. fp16) for cross-dtype TV.")
    p.add_argument("--attn_pi", default="sdpa",
                   help="Attention impl for π oracle (sdpa/eager)")
    p.add_argument("--attn_mu", default="sdpa",
                   help="Attention impl for μ oracle")
    p.add_argument("--sdpa_backend_pi", default=None,
                   help="SDPA backend for π (flash/math/mem_efficient/cudnn). "
                        "None = let PyTorch pick (default flash on GH200).")
    p.add_argument("--sdpa_backend_mu", default=None,
                   help="SDPA backend for μ.")
    p.add_argument("--resume_cache", action="store_true",
                   help="Checkpoint pi_pool.pt/mu_pool.pt after each K slice and resume/extend them.")
    p.add_argument("--checkpoint_every", type=int, default=1,
                   help="When --resume_cache is set, save after this many completed K slices.")
    p.add_argument("--score_batch_size", type=int, default=0,
                   help="Optional row chunk size for each score slice. 0 = score all rows at once.")
    p.add_argument("--balanced_cache", action="store_true",
                   help="When resuming, score pi/mu pools round-robin so completed_k stays balanced.")
    p.add_argument("--build_only", action="store_true",
                   help="Only build/resume pools and write config; skip estimators.")
    args = p.parse_args()

    if args.smoke:
        args.N_pi = 12
        args.N_mu = 8
        args.K_max = 4
        args.K_sweep = "1,2,4"
        args.N_each_directM = 6
        args.mlmc_levels = "4:1,2:2,2:4"
        args.n = 200

    K_sweep = [int(x) for x in args.K_sweep.split(",") if x.strip()]
    mlmc_levels = parse_levels(args.mlmc_levels)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # git SHA for reproducibility.
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    print(f"[run] git_sha={git_sha}, n={args.n}, top_k={args.top_k}", flush=True)
    print(f"[run] N_pi={args.N_pi}, N_mu={args.N_mu}, K_max={args.K_max}", flush=True)
    print(f"[run] K_sweep={K_sweep}, N_each_directM={args.N_each_directM}", flush=True)
    print(f"[run] mlmc_levels={mlmc_levels}", flush=True)

    run_config = {
        "model": args.model, "n": args.n, "top_k": args.top_k,
        "N_pi": args.N_pi, "N_mu": args.N_mu, "K_max": args.K_max,
        "K_sweep": K_sweep, "N_each_directM": args.N_each_directM,
        "mlmc_levels": mlmc_levels, "seed": args.seed, "git_sha": git_sha,
        "quant_pi": args.quant_pi, "quant_mu": args.quant_mu,
        "attn_pi": args.attn_pi, "attn_mu": args.attn_mu,
        "sdpa_backend_pi": args.sdpa_backend_pi,
        "sdpa_backend_mu": args.sdpa_backend_mu,
        "resume_cache": args.resume_cache,
        "checkpoint_every": args.checkpoint_every,
        "score_batch_size": args.score_batch_size,
        "balanced_cache": args.balanced_cache,
    }
    out_path = out_dir / "fig1_curves.json"
    with open(out_path, "w") as f:
        json.dump({"config": run_config, "ground_truth_TV": 0.0}, f, indent=2)
    print(f"[run] Wrote config stub to {out_path}", flush=True)

    # ── Load oracles ──
    print(f"\n[run] Loading oracle π ({args.model})...", flush=True)
    oracle_pi = HFOracle(args.model, top_k=args.top_k, quant_type=args.quant_pi,
                         attn_impl=args.attn_pi, sdpa_backend=args.sdpa_backend_pi)
    print(f"[run] Loading oracle μ (independent instance)...", flush=True)
    oracle_mu = HFOracle(args.model, top_k=args.top_k, quant_type=args.quant_mu,
                         attn_impl=args.attn_mu, sdpa_backend=args.sdpa_backend_mu)

    prompt_ids = tokenize_prompt(oracle_pi.tokenizer, args.system, args.user)
    print(f"[run] prompt_len={prompt_ids.shape[0]}, n={args.n}", flush=True)

    # ── Build pools ──
    t_build = time.time()
    score_batch_size = args.score_batch_size if args.score_batch_size > 0 else None
    if args.balanced_cache:
        print("\n=== balanced pi/mu pools ===", flush=True)
        pi_pool, mu_pool = build_pools_balanced(
            oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
            n=args.n, N_pi=args.N_pi, N_mu=args.N_mu, K_max=args.K_max,
            seed=args.seed, output_dir=out_dir, checkpoint_every=args.checkpoint_every,
            score_batch_size=score_batch_size,
        )
    else:
        print(f"\n=== pi_pool (N={args.N_pi}) ===", flush=True)
        pi_pool = build_pool(
            oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
            n=args.n, side="pi", N=args.N_pi, K_max=args.K_max, seed=args.seed,
            cache_path=(out_dir / "pi_pool.pt") if args.resume_cache else None,
            checkpoint_every=args.checkpoint_every,
            score_batch_size=score_batch_size,
        )
        if not args.resume_cache:
            torch.save(pi_pool, out_dir / "pi_pool.pt")

        print(f"\n=== mu_pool (N={args.N_mu}) ===", flush=True)
        mu_pool = build_pool(
            oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
            n=args.n, side="mu", N=args.N_mu, K_max=args.K_max, seed=args.seed + 1,
            cache_path=(out_dir / "mu_pool.pt") if args.resume_cache else None,
            checkpoint_every=args.checkpoint_every,
            score_batch_size=score_batch_size,
        )
        if not args.resume_cache:
            torch.save(mu_pool, out_dir / "mu_pool.pt")
    print(f"\n[run] pools built in {time.time() - t_build:.0f}s", flush=True)

    # Free oracles before running estimators (CPU-only from here on).
    del oracle_pi, oracle_mu
    torch.cuda.empty_cache()

    if args.build_only:
        print("[run] build_only set; skipping estimators", flush=True)
        return

    # ── Estimators ──
    print(f"\n=== Method 1 fused (V_2-LR) ===", flush=True)
    m1 = method1_fused(pi_pool)
    print(f"  estimate={m1['estimate']:.4f}  SE={m1['stderr']:.4f}  budget={m1['budget_forwards']}",
          flush=True)

    print(f"\n=== Method 5 fused (Pinsker) ===", flush=True)
    m5 = method5_fused(pi_pool)
    print(f"  KL_finite={m5['KL_hat_finite']:.4f}  Pinsker={m5['estimate']:.4f}  inf_rate={m5['inf_rate']:.2f}",
          flush=True)

    print(f"\n=== Method 4 π-direct K ===", flush=True)
    method4_results = []
    for K in K_sweep:
        if K > args.K_max:
            continue
        r = method4_pi_direct_K(pi_pool, K=K)
        method4_results.append(r)
        print(f"  K={K}  budget={r['budget_forwards']}  est={r['estimate']:.4f}  SE={r['stderr']:.4f}",
              flush=True)

    print(f"\n=== Direct-M K (mixture, no MLMC) ===", flush=True)
    directM_results = []
    for K in K_sweep:
        if K > args.K_max:
            continue
        r = direct_M_K(pi_pool, mu_pool, K=K, N_each=args.N_each_directM)
        directM_results.append(r)
        print(f"  K={K}  budget={r['budget_forwards']}  est={r['estimate']:.4f}  SE={r['stderr']:.4f}",
              flush=True)

    print(f"\n=== Method 3 MLMC logit ===", flush=True)
    mlmc_logit_results = []
    for L_use in range(len(mlmc_levels)):
        r = mlmc(pi_pool, mu_pool, levels=mlmc_levels, sketch="logit",
                 L_use=L_use, seed=args.seed)
        mlmc_logit_results.append(r)
        Y_str = " + ".join(f"{y['Y']:+.4f}" for y in r["Y_levels"])
        print(f"  L={L_use}  budget={r['budget_forwards']}  est={r['estimate']:.4f}  Y={Y_str}",
              flush=True)

    print(f"\n=== Method 2 MLMC one-hot ===", flush=True)
    mlmc_onehot_results = []
    for L_use in range(len(mlmc_levels)):
        r = mlmc(pi_pool, mu_pool, levels=mlmc_levels, sketch="onehot",
                 L_use=L_use, seed=args.seed)
        mlmc_onehot_results.append(r)
        Y_str = " + ".join(f"{y['Y']:+.4f}" for y in r["Y_levels"])
        print(f"  L={L_use}  budget={r['budget_forwards']}  est={r['estimate']:.4f}  Y={Y_str}",
              flush=True)

    # ── Save curves ──
    curves = {
        "config": run_config,
        "ground_truth_TV": 0.0,
        "method1_fused": m1,
        "method5_fused": m5,
        "method4_pi_direct": method4_results,
        "direct_m_K":       directM_results,
        "mlmc_logit":       mlmc_logit_results,
        "mlmc_onehot":      mlmc_onehot_results,
        "pool_build_time_sec": time.time() - t_build,
    }
    with open(out_path, "w") as f:
        json.dump(curves, f, indent=2)
    print(f"\n[run] Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
