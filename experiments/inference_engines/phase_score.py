"""Phase: score X tensors under one engine for K iterations.

Loads X from a phase_sample output (or existing partial score cache).
Iteratively scores K times and saves (N, K, n) score_L tensor.
Resume-aware: if --out exists with completed_k slices, continues from there.
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import torch


FORCE_PROC = ("experiments.inference_engines.force_tokens_vllm:"
              "ForceTokensLogitsProcessor")


def make_oracle(engine: str, model: str, top_k: int, dtype: str, gpu_mem: float, seed: int,
                score_path: str = "prefill", deterministic: bool = False,
                sdpa_backend: str | None = None):
    if engine == "hf":
        # sigma-study scorer: HF SDPA with a forced backend (None = auto dispatch).
        from experiments.oracle import HFOracle
        quant = {"bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16",
                 "float32": "fp32", "fp32": "fp32"}.get(dtype, "bf16")
        return HFOracle(model_name=model, top_k=top_k, quant_type=quant,
                        attn_impl="sdpa", sdpa_backend=sdpa_backend)
    if engine == "vllm":
        if deterministic:
            import os
            os.environ["VLLM_BATCH_INVARIANT"] = "1"  # before vllm import
        from experiments.inference_engines.vllm_oracle import VLLMOracle
        llm_kwargs = {}
        if deterministic:
            llm_kwargs["attention_backend"] = "FLASH_ATTN"  # required by batch_invariant
        # wide logprob readout for the sigma-study (k_save=256, both replay and
        # seq paths); a harmless higher ceiling for the legacy v2 path too.
        llm_kwargs["max_logprobs"] = max(256, top_k + 8)
        if score_path in ("replay", "sample"):
            # registered for both sample and replay phases => identical engine
            # config on both sides of the measurement
            llm_kwargs["logits_processors"] = [FORCE_PROC]
        return VLLMOracle(
            model_name=model, top_k=top_k, dtype=dtype,
            gpu_memory_utilization=gpu_mem, max_model_len=4096, seed=seed,
            enable_prefix_caching=(score_path == "decode"),
            llm_kwargs=llm_kwargs or None,
        )
    elif engine == "sglang":
        from experiments.inference_engines.sglang_oracle import SGLangOracle
        engine_kwargs = {}
        if deterministic:
            engine_kwargs.update(enable_deterministic_inference=True,
                                 disable_overlap_schedule=True)
        if score_path in ("replay", "sample"):
            engine_kwargs["enable_custom_logit_processor"] = True
        if score_path == "replay":
            # forcing indexes steps via req.output_ids, which lags under the
            # overlap scheduler => overlap must be off while replay-scoring
            engine_kwargs["disable_overlap_schedule"] = True
        if score_path == "decode":
            # radix cache must hold trunk (chunk*(prefix+n)) PLUS the dead
            # 1-token generation leaves (chunk*n) with LRU headroom
            mtt = 262144
        elif score_path in ("replay", "sample"):
            # all 512 trajectories decode concurrently in one wave, matching
            # the sampling-phase scheduler environment
            mtt = 300000
        else:
            mtt = 8192
        return SGLangOracle(
            model_name=model, top_k=top_k, dtype=dtype,
            mem_fraction_static=gpu_mem, seed=seed,
            max_total_tokens=mtt,
            engine_kwargs=engine_kwargs or None,
        )
    else:
        raise ValueError(engine)


def score_once(oracle, score_path, X, prompt_len, chunk=0, return_topk_logits=False):
    """One K-repeat via the requested scoring path; optional B-chunking."""
    import torch as _t
    B = X.shape[0]
    if score_path == "prefill":
        fn = lambda xs, rt: oracle.score_kv_batched(xs, prefix_len=prompt_len,
                                                    return_topk_logits=rt)
    elif score_path == "decode":
        return (oracle.score_decode_batched(X, prefix_len=prompt_len,
                                            chunk=(chunk or 64),
                                            return_topk_logits=return_topk_logits))
    elif score_path == "seq":
        fn = lambda xs, rt: oracle.score_seq_batched(xs, prefix_len=prompt_len,
                                                     return_topk_logits=rt)
    elif score_path == "replay":
        fn = lambda xs, rt: oracle.score_replay_batched(xs, prefix_len=prompt_len,
                                                        return_topk_logits=rt)
    else:
        raise ValueError(score_path)
    if not chunk or chunk >= B:
        return fn(X, return_topk_logits)
    outs = [fn(X[c:c + chunk], return_topk_logits) for c in range(0, B, chunk)]
    if return_topk_logits:
        return (None, _t.cat([o[1] for o in outs]), _t.cat([o[2] for o in outs]))
    return _t.cat(outs)


def lps_from_topk(topk_idx, topk_val, targets):
    """(B,n,k) idx/val + (B,n) targets -> (B,n) target lps (-inf if absent)."""
    import torch as _t
    match = topk_idx == targets[:, :, None]
    val = _t.where(match, topk_val, _t.full_like(topk_val, float("-inf")))
    return val.max(dim=-1).values


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    p.add_argument("--X_path", required=True, help="Output of phase_sample (X + meta)")
    p.add_argument("--K_max", type=int, default=64)
    p.add_argument("--out", required=True, help="Output score tensor file")
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_every", type=int, default=1)
    p.add_argument("--score_path", choices=["prefill", "decode", "seq", "replay"],
                   default="prefill",
                   help="prefill = May-2026 per-step full-prefill; decode = "
                        "incremental KV; seq = standard whole-sequence logprob "
                        "API (one forward per repeat); replay = teacher-forced "
                        "generation (unbiased generation-path oracle)")
    p.add_argument("--chunk", type=int, default=0,
                   help="optional B-chunking (decode: cache-resident chunk, "
                        "default 64; replay/seq: batch-shape diagnostic)")
    p.add_argument("--save_topk_reps", type=int, default=0,
                   help="save full top-k (idx,val) tensors for the first R "
                        "repeats to <out>.topk.pt (sigma/top-k dataset)")
    args = p.parse_args()

    src = torch.load(args.X_path, weights_only=False)
    X = src["X"]
    meta = src["meta"]
    N, L = X.shape
    n = meta["n"]
    prompt_len = meta["prompt_len"]
    print(f"[score] engine={args.engine} N={N} n={n} prompt_len={prompt_len} K_max={args.K_max}", flush=True)

    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        cur = torch.load(out_path, weights_only=False)
        cur_path = cur.get("meta", {}).get("score_path", "prefill")
        if cur_path != args.score_path:
            raise ValueError(
                f"resume score_path mismatch: existing cache was scored with "
                f"'{cur_path}', asked for '{args.score_path}' — mixing path "
                f"semantics in one score_L tensor is invalid")
        score_L = cur["score_L"]
        completed_k = int(cur.get("completed_k", 0))
        print(f"[score] resume: existing completed_k={completed_k}, shape={tuple(score_L.shape)}", flush=True)
        if score_L.shape[0] != N or score_L.shape[1] < args.K_max or score_L.shape[2] != n:
            if score_L.shape[1] < args.K_max:
                ext = torch.full((N, args.K_max, n), float("nan"), dtype=torch.float32)
                ext[:, :score_L.shape[1], :] = score_L
                score_L = ext
                print(f"[score] extended K dim to {args.K_max}", flush=True)
            else:
                raise ValueError(f"shape mismatch: got {tuple(score_L.shape)}, want (N={N}, K={args.K_max}, n={n})")
    else:
        score_L = torch.full((N, args.K_max, n), float("nan"), dtype=torch.float32)
        completed_k = 0

    if completed_k >= args.K_max:
        print(f"[score] already done (completed_k={completed_k} >= K_max={args.K_max})", flush=True)
        return

    t0 = time.time()
    oracle = make_oracle(args.engine, args.model, args.top_k, args.dtype, args.gpu_mem,
                         args.seed, score_path=args.score_path)
    print(f"[score] oracle loaded in {time.time()-t0:.0f}s (score_path={args.score_path})",
          flush=True)
    t0 = time.time()

    topk_store = {}
    targets = X[:, prompt_len:]
    for k in range(completed_k, args.K_max):
        tk = time.time()
        if k < args.save_topk_reps:
            _, ti, tv = score_once(oracle, args.score_path, X, prompt_len,
                                   chunk=args.chunk, return_topk_logits=True)
            topk_store[k] = (ti, tv)
            score_L[:, k, :] = lps_from_topk(ti, tv, targets)
        else:
            score_L[:, k, :] = score_once(oracle, args.score_path, X, prompt_len,
                                          chunk=args.chunk)
        completed_k = k + 1
        print(f"[score]   k={k+1}/{args.K_max} ({time.time()-tk:.1f}s, el={time.time()-t0:.0f}s)",
              flush=True)
        if completed_k == args.K_max or completed_k % args.checkpoint_every == 0:
            torch.save({"score_L": score_L, "completed_k": completed_k,
                        "meta": {**meta, "engine_score": args.engine, "K_max": args.K_max,
                                 "score_path": args.score_path}},
                       out_path)

    torch.save({"score_L": score_L, "completed_k": completed_k,
                "meta": {**meta, "engine_score": args.engine, "K_max": args.K_max,
                                 "score_path": args.score_path}},
               out_path)
    if topk_store:
        tk_path = out_path.with_name(out_path.stem + ".topk.pt")
        torch.save({"topk": topk_store,
                    "meta": {**meta, "engine_score": args.engine,
                             "score_path": args.score_path}}, tk_path)
        print(f"[score] wrote top-k tensors for {len(topk_store)} reps -> {tk_path}",
              flush=True)
    print(f"[score] DONE — wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
