"""Fixed-batch-size scorer: measure an engine's output at a FIXED batch size B,
with the batch COMPOSITION varying across the K repeats. This gives sigma(B) (the
per-batch-size noise) and the per-rep target log-probs needed for cross-engine
TV(B). Strings the whole batch-size axis B = 1 .. ~3N together; B=N reproduces the
warm_iso baseline (control), B=1 is the pure intrinsic floor (single sequence).

Reuses the validated oracle + scoring path from phase_score_sigma (--path decode =>
replay generation-path oracle, enforce_eager / prefix-caching off / disable_cuda_graph),
so the only new thing is the batch construction:
  - B >= N: one batch = N targets + (B-N) fillers (with replacement), scattered.
  - B <  N: chunk the N targets into ceil(N/B) groups of <= B, each padded with
            filler to size B; the grouping + positions are reshuffled every repeat.
  - B == 1: each target scored alone (N single-sequence forwards per repeat).

Run (one (engine, source, B) per process; parallelize across B and engine):
  python -m experiments.inference_engines.phase_score_fixed_B \
     --engine vllm --X_path .../pi_pool.pt --source vllm_pi --fixed_B 256 \
     --K 32 --K_save 20 --tv_only --out_dir $SCRATCH/sigmaB/vllm
"""
from __future__ import annotations
import argparse, hashlib, math, time
from pathlib import Path

import torch

from experiments.inference_engines.phase_score import make_oracle
from experiments.inference_engines.phase_score_sigma import (
    score_topk, target_logp_from_topk, engine_score_path)

NEG_INF = float("-inf")


def build_fixed_B(X, B, k, filler_pool, seed):
    """Return list of chunks covering all N targets at fixed batch size B.
    Each chunk: {batch (B,L), local_tr (positions of targets in the chunk),
    global_tr (target indices into 0..N-1)}. Composition varies with k."""
    N, L = X.shape
    g = torch.Generator().manual_seed(int(seed) * 100003 + int(k))

    def draw_filler(m):
        if m <= 0:
            return None
        idx = torch.randint(0, filler_pool.shape[0], (m,), generator=g)  # WITH replacement
        return filler_pool[idx]

    def make_chunk(tgt_X, global_idx):
        nt = tgt_X.shape[0]
        m = B - nt
        perm = torch.randperm(B, generator=g)
        tpos = sorted(perm[:nt].tolist())
        batch = torch.empty((B, L), dtype=X.dtype)
        batch[tpos] = tgt_X
        if m > 0:
            fpos = perm[nt:].tolist()
            batch[fpos] = draw_filler(m)
        return {"batch": batch, "local_tr": tpos, "global_tr": global_idx.tolist()}

    if B >= N:
        return [make_chunk(X, torch.arange(N))]
    order = torch.randperm(N, generator=g)                      # reshuffle grouping each rep
    return [make_chunk(X[order[s:s + B]], order[s:s + B]) for s in range(0, N, B)]


def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["hf", "vllm", "sglang"], required=True)
    p.add_argument("--sdpa_backend", default=None)
    p.add_argument("--path", choices=["decode", "prefill"], default="decode")
    p.add_argument("--X_path", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--fixed_B", type=int, default=None)
    p.add_argument("--mix_B", default=None,
                   help="comma list of B values; balanced random schedule over the K reps (uniform P)")
    p.add_argument("--fix_composition", action="store_true",
                   help="protocol I: identical batch composition every rep (pure run-to-run noise)")
    p.add_argument("--b1_subset", type=int, default=64,
                   help="in mix mode, B=1 reps score only the first this-many targets")
    p.add_argument("--k_list", default="",
                   help="comma list of score-side top-k truncations to store, e.g. '5,20,100'; "
                        "each k adds target_logp_k{k} (raw logp, -inf outside top-k) and "
                        "logZ_k{k} (logsumexp of the top-k) tensors; requires K_save >= max(k)")
    p.add_argument("--filler_path", default=None)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--K_save", type=int, default=20)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu_mem", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_subset", type=int, default=-1)
    p.add_argument("--ckpt_every", type=int, default=1)
    p.add_argument("--tv_only", action="store_true", help="no-op: this scorer always stores target_logp")
    args = p.parse_args()

    src = torch.load(args.X_path, weights_only=False)
    X = src["X"]
    meta = src["meta"]
    if args.n_subset > 0:
        X = X[:args.n_subset]
    N, L = X.shape
    n, plen = meta["n"], meta["prompt_len"]
    if args.mix_B:
        grid = sorted(int(x) for x in args.mix_B.split(","))
        assert args.K % len(grid) == 0, "K must be a multiple of len(mix_B) for a balanced schedule"
        sched = torch.tensor(grid * (args.K // len(grid)), dtype=torch.long)
        g = torch.Generator().manual_seed(args.seed * 7919 + 1)
        sched = sched[torch.randperm(args.K, generator=g)]
        tag = "mixB" + "-".join(str(b) for b in grid)
    else:
        assert args.fixed_B is not None, "need --fixed_B or --mix_B"
        sched = torch.full((args.K,), args.fixed_B, dtype=torch.long)
        tag = f"fixedB{args.fixed_B}"
    if args.fix_composition:
        tag += "_fixcomp"
    kl = [int(x) for x in args.k_list.split(",") if x.strip()] if args.k_list else []
    assert not kl or max(kl) <= args.K_save, "k_list entries must be <= K_save"
    filler = X if args.filler_path is None else torch.load(args.filler_path, weights_only=False)["X"]
    sb = None if args.sdpa_backend in (None, "auto") else args.sdpa_backend
    scorer = f"{args.engine}-{args.sdpa_backend}" if args.engine == "hf" else args.engine

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    key = f"{scorer}__{args.source}__{tag}__{args.path}"
    job_file = out_dir / f"{key}.pt"
    if job_file.exists():
        job = torch.load(job_file, weights_only=False)
        if "rep_B" in job:
            assert torch.equal(job["rep_B"], sched), "resume with a different B schedule"
            assert job.get("k_list", []) == kl, "resume with a different k_list"
    else:
        job = {"scorer": scorer, "source": args.source, "fixed_B": args.fixed_B if args.fixed_B else -1,
               "rep_B": sched, "b1_subset": args.b1_subset, "k_list": kl,
               "fix_composition": args.fix_composition, "path": args.path,
               "X": X, "target_logp": torch.full((N, args.K, n), NEG_INF, dtype=torch.float16),
               "done": torch.zeros(args.K, dtype=torch.bool),
               "meta": {**meta, "engine": args.engine, "tag": tag, "K": args.K, "N": N, "n": n,
                        "top_k": args.top_k, "K_save": args.K_save,
                        "seed": args.seed, "model": args.model}}
        for k_ in kl:
            job[f"target_logp_k{k_}"] = torch.full((N, args.K, n), NEG_INF, dtype=torch.float16)
            job[f"logZ_k{k_}"] = torch.zeros((N, args.K, n), dtype=torch.float16)

    t0 = time.time()
    oracle = make_oracle(args.engine, args.model, args.top_k, args.dtype, args.gpu_mem,
                         args.seed, score_path=engine_score_path(args.path), sdpa_backend=sb)
    print(f"[{tag}] {scorer}/{args.source} oracle in {time.time()-t0:.0f}s "
          f"(N={N} n={n} K={args.K} k_list={kl})", flush=True)
    target = job["X"][:, plen:plen + n].long()                       # (N,n)

    for k in range(args.K):
        if bool(job["done"][k]):
            continue
        tk = time.time()
        B_k = int(sched[k])
        Xk = X[:args.b1_subset] if (args.mix_B and B_k == 1) else X
        chunks = build_fixed_B(Xk, B_k, 0 if args.fix_composition else k, filler, args.seed)
        tl_k = torch.full((N, n), NEG_INF, dtype=torch.float32)
        buf_tl = {k_: torch.full((N, n), NEG_INF, dtype=torch.float32) for k_ in kl}
        buf_z = {k_: torch.zeros((N, n), dtype=torch.float32) for k_ in kl}
        for ch in chunks:
            ti, tv = score_topk(oracle, args.engine, args.path, ch["batch"], plen, args.K_save)
            gtr = ch["global_tr"]
            ti_t, tv_t = ti[ch["local_tr"]], tv[ch["local_tr"]]      # (nt, n, K_save)
            tl_k[gtr] = target_logp_from_topk(ti_t, tv_t, target[gtr]).float()
            for k_ in kl:
                buf_tl[k_][gtr] = target_logp_from_topk(ti_t[..., :k_], tv_t[..., :k_],
                                                        target[gtr]).float()
                buf_z[k_][gtr] = torch.logsumexp(tv_t[..., :k_].float(), dim=-1)
        job["target_logp"][:, k] = tl_k.half()
        for k_ in kl:
            job[f"target_logp_k{k_}"][:, k] = buf_tl[k_].half()
            job[f"logZ_k{k_}"][:, k] = buf_z[k_].half()
        job["done"][k] = True
        if (k + 1) % max(1, args.ckpt_every) == 0:
            torch.save(job, job_file)
        print(f"[{tag}]   k={k} B={B_k} chunks={len(chunks)} ({time.time()-tk:.1f}s)", flush=True)

    torch.save(job, job_file)
    print(f"[{tag}] DONE {key} -> {job_file}  md5={md5_of(job_file)[:8]}", flush=True)


if __name__ == "__main__":
    main()
