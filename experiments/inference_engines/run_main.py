"""Build (pi_pool, mu_pool) where π=vllm, μ=sglang (or vice versa).

Both engines load Qwen3-0.6B bf16; each occupies ~40 % GPU mem so they
fit side-by-side on a 96 GB GH200. Each generate() call is one fresh
forward → independent SDPA/FlashInfer noise across calls (K-averageable).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from experiments.fig1.cache_build import build_pool, tokenize_prompt
from experiments.inference_engines.vllm_oracle import VLLMOracle
from experiments.inference_engines.sglang_oracle import SGLangOracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--user", default="Tell me a story about a robot learning to paint.")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--N_pi", type=int, default=30)
    p.add_argument("--N_mu", type=int, default=30)
    p.add_argument("--K_max", type=int, default=8)
    p.add_argument("--gpu_mem_each", type=float, default=0.40)
    p.add_argument("--output_dir", default="results/vllm_vs_sglang")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.N_pi = args.N_mu = 8
        args.K_max = 2
        args.n = 64

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        git_sha = "unknown"

    print(f"[engines] git_sha={git_sha}")
    print(f"[engines] n={args.n}, N_pi={args.N_pi}, N_mu={args.N_mu}, K_max={args.K_max}")
    print(f"[engines] gpu_mem each = {args.gpu_mem_each}")

    # Load both engines simultaneously.
    print(f"\n[engines] Loading vLLM (π)...", flush=True)
    t0 = time.time()
    oracle_pi = VLLMOracle(
        args.model, top_k=args.top_k, dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem_each,
        max_model_len=max(2048, args.n + 64),
        seed=args.seed,
    )
    print(f"[engines]   vllm loaded in {time.time()-t0:.0f}s", flush=True)

    print(f"\n[engines] Loading SGLang (μ)...", flush=True)
    t1 = time.time()
    oracle_mu = SGLangOracle(
        args.model, top_k=args.top_k, dtype="bfloat16",
        mem_fraction_static=args.gpu_mem_each,
        max_total_tokens=max(8192, args.N_pi * (args.n + 64) * 2),
        seed=args.seed + 1,
    )
    print(f"[engines]   sglang loaded in {time.time()-t1:.0f}s", flush=True)

    prompt_ids = tokenize_prompt(oracle_pi.tokenizer, args.system, args.user)
    print(f"[engines] prompt_len={prompt_ids.shape[0]}, n={args.n}", flush=True)

    # Build pools.
    t_build = time.time()
    print(f"\n=== pi_pool (sample from vLLM, score under both) ===", flush=True)
    pi_pool = build_pool(
        oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
        n=args.n, side="pi", N=args.N_pi, K_max=args.K_max, seed=args.seed,
    )
    torch.save(pi_pool, out_dir / "pi_pool.pt")
    print(f"[engines] pi_pool saved ({time.time()-t_build:.0f}s)", flush=True)

    t_build = time.time()
    print(f"\n=== mu_pool (sample from SGLang, score under both) ===", flush=True)
    mu_pool = build_pool(
        oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
        n=args.n, side="mu", N=args.N_mu, K_max=args.K_max, seed=args.seed + 1,
    )
    torch.save(mu_pool, out_dir / "mu_pool.pt")
    print(f"[engines] mu_pool saved ({time.time()-t_build:.0f}s)", flush=True)

    cfg = {
        "model": args.model, "n": args.n, "top_k": args.top_k,
        "N_pi": args.N_pi, "N_mu": args.N_mu, "K_max": args.K_max,
        "pi_engine": "vllm", "mu_engine": "sglang",
        "dtype": "bfloat16",
        "seed": args.seed, "git_sha": git_sha,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\n[engines] done; config -> {out_dir}/config.json")


if __name__ == "__main__":
    main()
