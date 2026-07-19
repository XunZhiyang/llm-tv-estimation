"""Phase: sample N trajectories from one engine.

Run inside the venv that has the engine installed.
Output: dict {X: (N, prompt_len+n) int64, sample_lps: (N, n) fp32, meta: ...}.
Compatible with cache_build's pool format (sample-side fields).
"""
from __future__ import annotations
import argparse, json, time, os, sys
from pathlib import Path
import torch


from experiments.inference_engines.phase_score import make_oracle


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--user", default="Tell me a story about a robot learning to paint.")
    p.add_argument("--N", type=int, default=300)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group_size", type=int, default=0,
                   help="micro-batched sampling: trajectories per generate() call, "
                        "cache flushed between groups (0 = single call, May-2026 "
                        "semantics). For decode-path consistency this must equal "
                        "the score phase's --chunk; N must be divisible by it.")
    p.add_argument("--deterministic", action="store_true",
                   help="batch-invariant kernels (vllm VLLM_BATCH_INVARIANT, "
                        "sglang enable_deterministic_inference)")
    p.add_argument("--env", choices=["plain", "replay"], default="plain",
                   help="replay = identical engine config to the replay score "
                        "phase (force processor registered; sglang single-wave "
                        "max_total_tokens) so sample and score share one "
                        "numeric environment")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[sample] engine={args.engine} model={args.model} N={args.N} n={args.n} "
          f"top_k={args.top_k} dtype={args.dtype} seed={args.seed}", flush=True)
    t0 = time.time()
    oracle = make_oracle(args.engine, args.model, args.top_k, args.dtype, args.gpu_mem,
                         args.seed,
                         score_path=("sample" if args.env == "replay" else
                                     ("decode" if args.group_size else "prefill")),
                         deterministic=args.deterministic)
    print(f"[sample] oracle loaded in {time.time()-t0:.0f}s", flush=True)

    from experiments.fig1.cache_build import tokenize_prompt
    prompt_ids = tokenize_prompt(oracle.tokenizer, args.system, args.user)
    print(f"[sample] prompt_len={prompt_ids.shape[0]}", flush=True)

    rng = torch.Generator(device="cpu"); rng.manual_seed(args.seed)
    t0 = time.time()
    if args.group_size:
        if args.N % args.group_size != 0:
            raise ValueError(f"N={args.N} not divisible by group_size={args.group_size}")
        Xs, lps, tis, tvs = [], [], [], []
        for g in range(args.N // args.group_size):
            # cold cache per group == cold cache per score chunk (path symmetry)
            oracle._flush_cache()
            X_g, lp_g, ti_g, tv_g = oracle.sample_trajectories_batched(
                prompt_ids, n_tokens=args.n, B=args.group_size, rng=rng,
                return_topk_logits=True)
            Xs.append(X_g); lps.append(lp_g); tis.append(ti_g); tvs.append(tv_g)
            if (g + 1) % 8 == 0:
                print(f"[sample]   group {g+1}/{args.N // args.group_size} "
                      f"(el={time.time()-t0:.0f}s)", flush=True)
        X = torch.cat(Xs); sample_lps = torch.cat(lps)
        sample_topk_idx = torch.cat(tis); sample_topk_val = torch.cat(tvs)
    else:
        X, sample_lps, sample_topk_idx, sample_topk_val = oracle.sample_trajectories_batched(
            prompt_ids, n_tokens=args.n, B=args.N, rng=rng, return_topk_logits=True)
    print(f"[sample] sample done in {time.time()-t0:.0f}s  X={tuple(X.shape)}", flush=True)

    out = {
        "X": X.cpu(),
        "sample_lps": sample_lps.cpu(),
        "meta": {
            "engine": args.engine, "model": args.model, "N": args.N, "n": args.n,
            "top_k": args.top_k, "dtype": args.dtype, "seed": args.seed,
            "prompt_len": int(prompt_ids.shape[0]),
            "n_total": int(prompt_ids.shape[0]) + args.n,
            "group_size": args.group_size,
            "deterministic": args.deterministic,
            "env": args.env,
        },
    }
    if sample_topk_idx is not None:
        out["sample_topk_idx"] = sample_topk_idx.cpu()
        out["sample_topk_val"] = sample_topk_val.cpu()
    torch.save(out, out_path)
    print(f"[sample] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
