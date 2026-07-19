"""idev: cuDNN per-call self-noise histogram for 1.7B/8B/7B/MoE (warm_iso).
Tiny npz back. flash/math are sigma=0 (delta) across all models (Fig 1a), so one
representative spike (0.6B) suffices."""
import math
import numpy as np
import torch

NEG_INF = float("-inf")
BASE = "/path/to/scratch/sigma-taxonomy/results"
POOLS = {
    "Qwen3-1.7B":  f"{BASE}/breadth_qwen17b/hf-cudnn__hf-flash__warm_iso__decode.pt",
    "Qwen3-8B":    f"{BASE}/breadth_m8b/hf-cudnn__hf-flash__warm_iso__decode.pt",
    "Qwen2.5-7B":  f"{BASE}/breadth_m25_7b/hf-cudnn__hf-flash__warm_iso__decode.pt",
    "Qwen3-30B-MoE": f"{BASE}/sigma_taxonomy_moe30/hf-cudnn__hf-flash__warm_iso__decode.pt",
}
OUT = "/path/to/scratch/crossmodel_cudnn_hist.npz"
BINS = np.linspace(-0.30, 0.30, 161)


def target_logprob(job):
    idx = job["topk_idx"]; val = job["topk_val"].float()
    N, K, n, K_save = idx.shape
    plen = job["meta"]["prompt_len"]
    cap = max(1, min(int(job["meta"].get("logprob_cap", K_save)), K_save))
    idx = idx[..., :cap]; val = val[..., :cap]
    tgt = job["X"][:, plen:plen + n].long().view(N, 1, n, 1)
    match = (idx.long() == tgt) & torch.isfinite(val)
    has = match.any(-1)
    picked = torch.where(match, val, torch.full_like(val, NEG_INF)).max(-1).values
    return torch.where(has, picked, torch.full((), NEG_INF))


out = {"bins": BINS.astype(np.float32)}
for name, path in POOLS.items():
    print("loading", name, flush=True)
    job = torch.load(path, weights_only=False)
    L = target_logprob(job)
    K = L.shape[1]
    cells = L.permute(0, 2, 1).reshape(-1, K)
    kept = cells[torch.isfinite(cells).all(1)].numpy()
    eps = (kept - kept.mean(1, keepdims=True)).ravel()
    eps = eps[np.isfinite(eps)]
    key = name.replace("-", "_").replace(".", "p")
    out[f"pc_{key}"] = np.histogram(eps, bins=BINS, density=True)[0].astype(np.float32)
    out[f"pc_{key}_std"] = np.float32(eps.std())
    out[f"pc_{key}_K"] = np.int32(K)
    print(f"  {name}: K={K} per-call std={eps.std():.4f}", flush=True)
    del job, L, cells, kept, eps

np.savez(OUT, **out)
print("wrote", OUT, flush=True)
