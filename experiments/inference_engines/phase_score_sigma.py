"""sigma-study scorer: one (scorer, source, regime, path) job.

Scores a fixed source pool X under one regime and path, K repeats, storing top-K_save
raw log-probs per (trajectory, repeat, position) into the unified storage schema. The
abstract path maps to the concrete generation-path / whole-sequence oracle method per engine
(get this right — a wrong mapping re-introduces the May "scored prefill while sampling decode"
bug). FRESH regime: one repeat per process via --rep_slot (resumable); WARM: all K in-process.

Usage (warm, all K):
  python -m experiments.inference_engines.phase_score_sigma \
      --engine hf --sdpa_backend auto --path decode --regime warm_iso \
      --X_path .../hf-flash_X.pt --source hf-flash --K 16 --K_save 256 \
      --out_dir .../results/sigma_taxonomy_v1
FRESH: add --regime fresh --K 8 --rep_slot <k>  (call once per k in a fresh process).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import torch

SKIP_UNSUPPORTED = 42   # exit code: this (backend, path) has no available kernel — skip, don't fail
NEG_INF = float("-inf")

from experiments.inference_engines import regime as regime_mod
from experiments.inference_engines import storage
from experiments.inference_engines.phase_score import make_oracle


def target_logp_from_topk(ti, tv, target):
    """logp of each target token from THIS repeat's top-k (ti, tv); -inf if the target is
    outside the returned support. ti:(N,n,w) int, tv:(N,n,w) float, target:(N,n) long ->
    (N,n) float. A tv_only job's stored value is bit-identical to extracting the target
    from a full-topk job (verified by smoke)."""
    tgt = target.view(target.shape[0], target.shape[1], 1)          # (N, n, 1)
    match = (ti.long() == tgt) & torch.isfinite(tv)                 # (N, n, w)
    has = match.any(dim=-1)
    picked = torch.where(match, tv, torch.full_like(tv, NEG_INF)).max(dim=-1).values
    return torch.where(has, picked, torch.full((), NEG_INF, dtype=tv.dtype))


# (engine, abstract path) -> concrete oracle call. decode = generation-path oracle (what X was
# sampled from); prefill = one-shot whole-sequence readout.
def score_topk(oracle, engine, path, batch, prompt_len, k_save, prefill_chunk=64):
    """Return (idx, val) top-k_save raw log-probs for the batch, shape (B, n, k_save)."""
    if engine == "hf":
        if path == "decode":
            _, ti, tv = oracle.score_kv_batched(
                batch, prefix_len=prompt_len, return_topk_logits=True, k_save=k_save)
        elif path == "prefill":
            _, ti, tv = oracle.score_prefill_batched(
                batch, prefix_len=prompt_len, k_save=k_save, return_topk_logits=True,
                chunk=prefill_chunk)
        else:
            raise ValueError((engine, path))
    elif engine in ("vllm", "sglang"):
        if path == "decode":
            _, ti, tv = oracle.score_replay_batched(
                batch, prefix_len=prompt_len, return_topk_logits=True, k_save=k_save)
        elif path == "prefill":
            _, ti, tv = oracle.score_seq_batched(
                batch, prefix_len=prompt_len, return_topk_logits=True, k_save=k_save)
        else:
            raise ValueError((engine, path))
    else:
        raise ValueError(engine)
    return ti, tv


def engine_score_path(path):
    """For engine oracle construction: decode->replay (force proc), prefill->seq."""
    return "replay" if path == "decode" else "seq"


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["hf", "vllm", "sglang"], required=True)
    p.add_argument("--sdpa_backend", choices=["auto", "flash", "cudnn", "memeff", "math"],
                   default=None, help="HF only; 'auto' = None (let PyTorch dispatch)")
    p.add_argument("--path", choices=["decode", "prefill"], required=True)
    p.add_argument("--regime", choices=["fresh", "warm_iso", "warm_serve"], required=True)
    p.add_argument("--X_path", required=True, help="source pool (phase_sample output)")
    p.add_argument("--source", required=True, help="label of the sampling source")
    p.add_argument("--filler_path", default=None, help="pool for warm_serve filler (default: X)")
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--K_save", type=int, default=256)
    p.add_argument("--rep_slot", type=int, default=-1, help="fresh: write only this slot")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full_vocab", action="store_true",
                   help="store dense full-vocab log-softmax (HF only; calibration subset)")
    p.add_argument("--prefill_chunk", type=int, default=64,
                   help="HF prefill: sub-batch size for the whole-seq forward (memory cap; "
                        "lower for big models, e.g. 8 for the 35B MoE)")
    p.add_argument("--tv_only", action="store_true",
                   help="store ONLY target-token logp (N,K,n) not full top-K_save; enables "
                        "K~1e4+ for a low self-TV floor (drops topk_union/sigma-vs-K)")
    p.add_argument("--ckpt_every", type=int, default=1,
                   help="save the job every N repeats (default 1 = every rep). Set higher "
                        "(e.g. 200) for huge K so the growing tensor is not re-written each rep")
    p.add_argument("--n_subset", type=int, default=-1,
                   help="use only the first n_subset trajectories of the source pool "
                        "(full-topk microscope: few cells, big K_save, huge K). filler stays full")
    args = p.parse_args()

    sb = None if args.sdpa_backend in (None, "auto") else args.sdpa_backend
    scorer = f"{args.engine}-{args.sdpa_backend}" if args.engine == "hf" else args.engine

    src = torch.load(args.X_path, weights_only=False)
    X = src["X"]
    meta = src["meta"]
    if args.n_subset > 0:
        X = X[:args.n_subset]               # microscope: few trajectories (filler stays full pool)
    N, L = X.shape
    n = meta["n"]
    prompt_len = meta["prompt_len"]
    filler_pool = X if args.filler_path is None else torch.load(
        args.filler_path, weights_only=False)["X"]
    if filler_pool.shape[1] != L:
        raise ValueError(f"filler L={filler_pool.shape[1]} != source L={L} (prompt must match)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = storage.new_job(scorer=scorer, source=args.source, regime=args.regime,
                          path=args.path, N=N, K=args.K, n=n, K_save=args.K_save,
                          prompt_len=prompt_len, tv_only=args.tv_only)
    key = storage.job_key(job)
    job_file = out_dir / f"{key}.pt"
    if job_file.exists():                       # resume (fresh writes slots across processes;
        job = storage.load_job(out_dir, key)    # warm resumes undone repeats via batch_B below)
    job["X"] = X
    slp = src.get("sample_lps", src.get("sample_logprobs"))   # phase_sample uses "sample_lps"
    if slp is not None:
        job["sample_logprobs"] = (slp[:args.n_subset] if args.n_subset > 0 else slp).half()
    job["meta"].update(model=args.model, dtype=args.dtype, top_k=args.top_k,
                       sdpa_backend_requested=args.sdpa_backend, seed=args.seed,
                       prompt=meta.get("prompt"), full_vocab=bool(args.full_vocab),
                       tv_only=bool(args.tv_only), ckpt_every=int(args.ckpt_every),
                       n_subset=(int(args.n_subset) if args.n_subset > 0 else N))

    t0 = time.time()
    oracle = make_oracle(args.engine, args.model, args.top_k, args.dtype, args.gpu_mem,
                         args.seed, score_path=engine_score_path(args.path), sdpa_backend=sb)
    print(f"[sigma] {scorer}/{args.source}/{args.regime}/{args.path} oracle in "
          f"{time.time()-t0:.0f}s (N={N} n={n} K={args.K} K_save={args.K_save})", flush=True)

    if args.rep_slot >= 0:
        slots = [args.rep_slot]                           # fresh: one repeat per process
    else:                                                 # warm: all K in-process; on resume
        slots = [k for k in range(args.K)                 # skip repeats already scored
                 if int(job["batch_B"][k]) == 0]
    if args.full_vocab:                          # dense HF-only calibration
        if args.engine != "hf":
            raise ValueError("--full_vocab is HF-only")
        V = oracle.vocab_size
        if "logits_full" not in job:
            job["logits_full"] = torch.zeros((N, args.K, n, V), dtype=torch.float16)

    for k in slots:
        tk = time.time()
        bi = regime_mod.build_batch(X, args.regime, k, filler_pool, seed=args.seed)
        batch, tr, B = bi["batch"], bi["target_rows"], bi["B"]
        try:
            if args.full_vocab:
                lp = oracle.full_logsoftmax_prefill(batch, prefix_len=prompt_len)   # (B,n,V)
                job["logits_full"][:, k] = lp[tr]
            else:
                ti, tv = score_topk(oracle, args.engine, args.path, batch, prompt_len,
                                    args.K_save, prefill_chunk=args.prefill_chunk)
                ti, tv = ti[tr], tv[tr]              # (N, n, K_save)
                job["meta"]["logprob_cap"] = int(ti.shape[-1])  # honest: actual returned width
                if args.tv_only:                     # store only the target token's logp
                    target = job["X"][:, prompt_len:prompt_len + n].long()      # (N, n)
                    job["target_logp"][:, k] = target_logp_from_topk(ti, tv, target).half()
                else:
                    w = ti.shape[-1]
                    job["topk_idx"][:, k, :, :w] = ti
                    job["topk_val"][:, k, :, :w] = tv.half()
        except RuntimeError as e:
            # forced SDPA backends (flash/cudnn) reject the batched-prefill mask
            # ("No available kernel"); skip this (scorer, path) cleanly rather than crash.
            if "available kernel" in str(e).lower():
                print(f"[sigma] SKIP {key}: no kernel for ({scorer}, path={args.path}): {e}",
                      flush=True)
                sys.exit(SKIP_UNSUPPORTED)
            raise
        job["batch_B"][k] = int(B)
        job["batch_sig"][k] = f"B={B},m={B-N},tgt@{tr[0]}-{tr[-1]}"
        if (k + 1) % max(1, args.ckpt_every) == 0:   # throttle: avoid re-writing a huge tensor each rep
            storage.save_job(out_dir, job)
        print(f"[sigma]   k={k} B={B} ({time.time()-tk:.1f}s)", flush=True)

    storage.save_job(out_dir, job)
    storage.update_manifest(out_dir, job, md5=md5_of(job_file), status="done")
    print(f"[sigma] DONE {key} -> {job_file}", flush=True)


if __name__ == "__main__":
    main()
