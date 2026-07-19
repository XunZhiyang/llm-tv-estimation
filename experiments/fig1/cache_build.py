"""Cache builder for Figure 1 (Self-TV) — shared stratified design.

Two pools, one per sampling distribution:
  pi_pool: X ~ π,  scored K_max times under π (fresh) + K_max times under μ
  mu_pool: X ~ μ,  scored K_max times under π + K_max times under μ

Stored quantities are SCALAR target-token logprobs only (no top-k logits):
  X            : (N, prompt_len + n)  int64       — sampled trajectories
  sample_L_self: (N, n)                fp32       — sample-time logprob (fused)
  score_L_pi   : (N, K_max, n)         fp32       — fresh evals under π
  score_L_mu   : (N, K_max, n)         fp32       — fresh evals under μ

All Fig-1 estimators (Method 1, 4, 5; Direct-M K; MLMC logit/one-hot)
operate on these two pools — no per-method caches.

Cost: per pool, 1 sample call + 2 K_max score calls = 1 + 2 K_max batched
forward passes at B=N. With K_max=8, N_pi=100, N_mu=50: 17 + 17 = 34 batched
calls; ~13s/call warm + 140s warmup ≈ 9 min wall.
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Literal

import torch

from experiments.oracle import HFOracle


def tokenize_prompt(tokenizer, system: str, user: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    out = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if not torch.is_tensor(out):  # newer transformers return a BatchEncoding
        out = out["input_ids"]
    return out.squeeze(0)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    """Write a checkpoint without corrupting the previous one if killed mid-save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _extend_score_tensor(old: torch.Tensor, K_max: int, n: int) -> torch.Tensor:
    """Return a (N, K_max, n) tensor, copying existing K slices."""
    N = old.shape[0]
    if old.shape[1] == K_max and old.shape[2] == n:
        return old
    if old.shape[1] > K_max or old.shape[2] != n:
        raise ValueError(
            f"Cannot reuse score tensor with shape {tuple(old.shape)} for K_max={K_max}, n={n}"
        )
    out = torch.full((N, K_max, n), float("nan"), dtype=torch.float32)
    out[:, : old.shape[1], :] = old
    return out


def _row_completed(pool: dict) -> torch.Tensor:
    """Per-row completed K counts, backfilled for older uniform-cache files."""
    if "row_completed_k" not in pool:
        N = int(pool["meta"]["N"])
        completed = int(pool["meta"].get("completed_k", pool["meta"]["K_max"]))
        pool["row_completed_k"] = torch.full((N,), completed, dtype=torch.int16)
    return pool["row_completed_k"]


def _infer_row_completed_from_scores(pool: dict) -> torch.Tensor:
    """Infer completed K prefix per row from non-NaN score slices."""
    score_pi = pool["score_L_pi"]
    score_mu = pool["score_L_mu"]
    if score_pi.shape != score_mu.shape:
        raise ValueError(
            f"score_L_pi shape {tuple(score_pi.shape)} != "
            f"score_L_mu shape {tuple(score_mu.shape)}"
        )

    valid = (~torch.isnan(score_pi).any(dim=-1)) & (~torch.isnan(score_mu).any(dim=-1))
    # Completion is the length of the initial valid prefix. Later valid slices
    # after a hole are ignored, because estimators require contiguous K repeats.
    return valid.to(torch.int16).cumprod(dim=1).sum(dim=1).to(torch.int16)


def _repair_row_completed_from_scores(pool: dict, *, side: str) -> None:
    """Make resume metadata match the actual saved score tensors."""
    inferred = _infer_row_completed_from_scores(pool)
    old = pool.get("row_completed_k")
    if old is None or old.shape != inferred.shape or not torch.equal(old.to(torch.int16), inferred):
        if old is None:
            print(f"[pool/{side}]   inferred row_completed_k from score tensors", flush=True)
        else:
            old_min = int(old.min().item()) if old.numel() else 0
            old_max = int(old.max().item()) if old.numel() else 0
            print(
                f"[pool/{side}]   repaired row_completed_k "
                f"old=({old_min},{old_max}) "
                f"new=({int(inferred.min().item())},{int(inferred.max().item())})",
                flush=True,
            )
        pool["row_completed_k"] = inferred
    _refresh_completed_meta(pool)
    pool["meta"]["complete"] = pool["meta"]["completed_k"] >= int(pool["meta"]["K_max"])


def _refresh_completed_meta(pool: dict) -> None:
    row_k = _row_completed(pool)
    pool["meta"]["completed_k"] = int(row_k.min().item())
    pool["meta"]["max_completed_k"] = int(row_k.max().item())


def _append_rows(
    *,
    pool: dict,
    seqs_new: torch.Tensor,
    sample_lps_new: torch.Tensor,
    sample_key: str,
    K_max: int,
    n: int,
) -> dict:
    """Append newly sampled rows with zero completed score slices."""
    old_N = int(pool["meta"]["N"])
    new_N = old_N + seqs_new.shape[0]
    old_K = pool["score_L_pi"].shape[1]

    score_pi = torch.full((new_N, K_max, n), float("nan"), dtype=torch.float32)
    score_mu = torch.full((new_N, K_max, n), float("nan"), dtype=torch.float32)
    score_pi[:old_N, :old_K, :] = pool["score_L_pi"]
    score_mu[:old_N, :old_K, :] = pool["score_L_mu"]

    pool["X"] = torch.cat([pool["X"], seqs_new], dim=0)
    pool[sample_key] = torch.cat([pool[sample_key], sample_lps_new], dim=0)
    pool["score_L_pi"] = score_pi
    pool["score_L_mu"] = score_mu
    pool["row_completed_k"] = torch.cat(
        [_row_completed(pool), torch.zeros(seqs_new.shape[0], dtype=torch.int16)],
        dim=0,
    )
    pool["meta"]["N"] = new_N
    _refresh_completed_meta(pool)
    return pool


def _prepare_pool(
    *,
    oracle_pi: HFOracle,
    oracle_mu: HFOracle,
    prompt_ids: torch.Tensor,
    n: int,
    side: Literal["pi", "mu"],
    N: int,
    K_max: int,
    seed: int,
    cache_path: str | Path | None,
) -> tuple[dict, int, float, Path | None]:
    """Load/resume a pool or sample fresh trajectories, but do not score K slices."""
    prompt_len = prompt_ids.shape[0]
    n_total = prompt_len + n
    sampler = oracle_pi if side == "pi" else oracle_mu
    t0 = time.time()

    cache = Path(cache_path) if cache_path is not None else None
    sample_key = f"sample_L_{side}"
    pool: dict | None = None
    completed_k = 0

    if cache is not None and cache.exists():
        print(f"[pool/{side}] resume candidate: {cache}", flush=True)
        pool = torch.load(cache, map_location="cpu", weights_only=False)
        meta = pool.get("meta", {})
        expected = {
            "side": side,
            "n": n,
            "prompt_len": prompt_len,
            "n_total": n_total,
            "seed": seed,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"Cache {cache} has meta[{key}]={meta.get(key)!r}, expected {value!r}. "
                    "Use a new output_dir or delete the stale cache."
                )
        old_N = int(meta.get("N", pool["X"].shape[0]))
        if old_N > N:
            raise ValueError(f"Cache {cache} has N={old_N}, larger than requested N={N}")
        if pool["X"].shape != (old_N, n_total):
            raise ValueError(f"Cache {cache} has bad X shape {tuple(pool['X'].shape)}")
        if pool[sample_key].shape != (old_N, n):
            raise ValueError(
                f"Cache {cache} has bad {sample_key} shape {tuple(pool[sample_key].shape)}"
            )
        for score_key in ("score_L_pi", "score_L_mu"):
            if pool[score_key].shape[0] != old_N or pool[score_key].shape[2] != n:
                raise ValueError(
                    f"Cache {cache} has bad {score_key} shape "
                    f"{tuple(pool[score_key].shape)}"
                )

        old_K = int(pool["score_L_pi"].shape[1])
        _repair_row_completed_from_scores(pool, side=side)
        pool["score_L_pi"] = _extend_score_tensor(pool["score_L_pi"], K_max, n)
        pool["score_L_mu"] = _extend_score_tensor(pool["score_L_mu"], K_max, n)
        pool["meta"] = {**meta, "N": old_N, "K_max": K_max}
        _repair_row_completed_from_scores(pool, side=side)
        print(
            f"[pool/{side}]   resumed rows={old_N}/{N} "
            f"completed_k={pool['meta']['completed_k']} "
            f"max_completed_k={pool['meta']['max_completed_k']} "
            f"(cached_K={old_K}, target_K={K_max})",
            flush=True,
        )

    if pool is None:
        rng = torch.Generator(device="cpu"); rng.manual_seed(seed)
        print(f"[pool/{side}] sample (B={N}, n={n})...", flush=True)
        seqs, sample_lps = sampler.sample_trajectories_batched(
            prompt_ids, n_tokens=n, B=N, rng=rng,
        )
        t_sample = time.time() - t0
        print(f"[pool/{side}]   sample done ({t_sample:.1f}s)", flush=True)

        pool = {
            "X": seqs,                       # (N, prompt_len + n)
            sample_key: sample_lps,          # (N, n)
            "score_L_pi": torch.full((N, K_max, n), float("nan"), dtype=torch.float32),
            "score_L_mu": torch.full((N, K_max, n), float("nan"), dtype=torch.float32),
            "meta": {
                "side": side, "N": N, "K_max": K_max, "n": n,
                "prompt_len": prompt_len, "n_total": n_total,
                "seed": seed, "completed_k": 0,
                "build_time_sec": time.time() - t0,
            },
            "row_completed_k": torch.zeros(N, dtype=torch.int16),
        }
        if cache is not None:
            _atomic_torch_save(pool, cache)
            print(f"[pool/{side}]   checkpointed sample -> {cache}", flush=True)

    if int(pool["meta"]["N"]) < N:
        old_N = int(pool["meta"]["N"])
        append_N = N - old_N
        append_seed = seed + 1_000_003 + old_N
        rng = torch.Generator(device="cpu"); rng.manual_seed(append_seed)
        print(
            f"[pool/{side}] append sample rows {old_N}:{N} "
            f"(B={append_N}, n={n}, seed={append_seed})...",
            flush=True,
        )
        seqs_new, sample_lps_new = sampler.sample_trajectories_batched(
            prompt_ids, n_tokens=n, B=append_N, rng=rng,
        )
        pool = _append_rows(
            pool=pool, seqs_new=seqs_new, sample_lps_new=sample_lps_new,
            sample_key=sample_key, K_max=K_max, n=n,
        )
        pool["meta"]["append_seed"] = append_seed
        pool["meta"]["build_time_sec"] = time.time() - t0
        if cache is not None:
            _atomic_torch_save(pool, cache)
            print(f"[pool/{side}]   checkpointed append -> {cache}", flush=True)

    return pool, prompt_len, t0, cache


def _score_pool_k(
    *,
    pool: dict,
    side: Literal["pi", "mu"],
    oracle_pi: HFOracle,
    oracle_mu: HFOracle,
    prompt_len: int,
    k: int,
    K_max: int,
    t0: float,
    cache: Path | None,
    checkpoint_every: int,
    score_batch_size: int | None = None,
) -> None:
    """Score one K slice under both oracles and optionally checkpoint."""
    tk = time.time()
    row_k = _row_completed(pool)
    idx = torch.nonzero(row_k == k, as_tuple=False).flatten()
    if len(idx) == 0:
        _refresh_completed_meta(pool)
        return
    if score_batch_size is not None and score_batch_size > 0 and len(idx) > score_batch_size:
        chunks = list(idx.split(score_batch_size))
    else:
        chunks = [idx]
    for ci, chunk_idx in enumerate(chunks, start=1):
        seqs = pool["X"][chunk_idx]
        pool["score_L_pi"][chunk_idx, k, :] = oracle_pi.score_kv_batched(
            seqs, prefix_len=prompt_len
        )
        pool["score_L_mu"][chunk_idx, k, :] = oracle_mu.score_kv_batched(
            seqs, prefix_len=prompt_len
        )
        if len(chunks) > 1:
            print(
                f"[pool/{side}]     score k={k+1}/{K_max} "
                f"chunk={ci}/{len(chunks)} rows={len(chunk_idx)}",
                flush=True,
            )
    row_k[idx] = k + 1
    _refresh_completed_meta(pool)
    pool["meta"]["complete"] = pool["meta"]["completed_k"] == K_max
    pool["meta"]["build_time_sec"] = time.time() - t0
    print(
        f"[pool/{side}]   score k={k+1}/{K_max} rows={len(idx)} "
        f"({time.time() - tk:.1f}s, el={time.time() - t0:.0f}s, "
        f"completed={pool['meta']['completed_k']}, "
        f"max={pool['meta']['max_completed_k']})",
        flush=True,
    )
    if cache is not None and (
        pool["meta"]["completed_k"] == K_max
        or pool["meta"]["completed_k"] % checkpoint_every == 0
    ):
        _atomic_torch_save(pool, cache)


def build_pool(
    *,
    oracle_pi: HFOracle,
    oracle_mu: HFOracle,
    prompt_ids: torch.Tensor,
    n: int,
    side: Literal["pi", "mu"],
    N: int,
    K_max: int,
    seed: int,
    cache_path: str | Path | None = None,
    checkpoint_every: int = 1,
    score_batch_size: int | None = None,
) -> dict:
    """Sample N trajectories from `side` oracle; score under both.

    If cache_path is provided, the pool is checkpointed after sampling and every
    checkpoint_every completed K slices. Re-running with the same path resumes
    from the latest completed slice; re-running with a larger K_max extends the
    cache.
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    pool, prompt_len, t0, cache = _prepare_pool(
        oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
        n=n, side=side, N=N, K_max=K_max, seed=seed, cache_path=cache_path,
    )
    completed_k = int(pool["meta"].get("completed_k", 0))
    for k in range(completed_k, K_max):
        _score_pool_k(
            pool=pool, side=side, oracle_pi=oracle_pi, oracle_mu=oracle_mu,
            prompt_len=prompt_len, k=k, K_max=K_max, t0=t0, cache=cache,
            checkpoint_every=checkpoint_every, score_batch_size=score_batch_size,
        )

    pool["meta"]["completed_k"] = K_max
    pool["meta"]["complete"] = True
    pool["meta"]["build_time_sec"] = time.time() - t0
    if cache is not None:
        _atomic_torch_save(pool, cache)
        print(f"[pool/{side}]   complete checkpoint -> {cache}", flush=True)
    return pool


def build_pools_balanced(
    *,
    oracle_pi: HFOracle,
    oracle_mu: HFOracle,
    prompt_ids: torch.Tensor,
    n: int,
    N_pi: int,
    N_mu: int,
    K_max: int,
    seed: int,
    output_dir: str | Path,
    checkpoint_every: int = 1,
    score_batch_size: int | None = None,
) -> tuple[dict, dict]:
    """Build pi/mu pools by balancing completed K across the two sides.

    This is useful for short resumable jobs: after any partial run, both pools
    have comparable completed_k, so CPU post-processing can already produce
    balanced M4/Mixture diagnostics at K <= min(completed_k_pi, completed_k_mu).
    """
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pi_pool, pi_prompt_len, pi_t0, pi_cache = _prepare_pool(
        oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
        n=n, side="pi", N=N_pi, K_max=K_max, seed=seed,
        cache_path=out_dir / "pi_pool.pt",
    )
    mu_pool, mu_prompt_len, mu_t0, mu_cache = _prepare_pool(
        oracle_pi=oracle_pi, oracle_mu=oracle_mu, prompt_ids=prompt_ids,
        n=n, side="mu", N=N_mu, K_max=K_max, seed=seed + 1,
        cache_path=out_dir / "mu_pool.pt",
    )
    if pi_prompt_len != mu_prompt_len:
        raise ValueError("pi/mu prompt lengths differ")

    pools = {"pi": pi_pool, "mu": mu_pool}
    t0s = {"pi": pi_t0, "mu": mu_t0}
    caches = {"pi": pi_cache, "mu": mu_cache}

    while True:
        completed = {side: int(pool["meta"].get("completed_k", 0)) for side, pool in pools.items()}
        if completed["pi"] >= K_max and completed["mu"] >= K_max:
            break
        side = min(
            (s for s in ("pi", "mu") if completed[s] < K_max),
            key=lambda s: (completed[s], 0 if s == "pi" else 1),
        )
        k = completed[side]
        _score_pool_k(
            pool=pools[side], side=side, oracle_pi=oracle_pi, oracle_mu=oracle_mu,
            prompt_len=pi_prompt_len, k=k, K_max=K_max, t0=t0s[side],
            cache=caches[side], checkpoint_every=checkpoint_every,
            score_batch_size=score_batch_size,
        )

    for side, pool in pools.items():
        pool["meta"]["completed_k"] = K_max
        pool["meta"]["complete"] = True
        pool["meta"]["build_time_sec"] = time.time() - t0s[side]
        if caches[side] is not None:
            _atomic_torch_save(pool, caches[side])
            print(f"[pool/{side}]   complete checkpoint -> {caches[side]}", flush=True)
    return pi_pool, mu_pool
