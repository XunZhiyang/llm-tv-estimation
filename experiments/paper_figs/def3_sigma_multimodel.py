"""Vista: Def-3 sigma (Definition 3 noisy-oracle parameter) for all 5 models,
via the half-split chi^2 method (same as topk_union/sigma_squared_vs_K.py, the
Figure-4 methodology). For each (model, kernel, regime) full-topk pool, compute
sigma^2(top-k, single-forward) = chi2[K_avg=1] / (1 + 1/K_half), renormalized to
the top-k union. Outputs a tiny JSON: {model: {kernel: {regime: {sigma_def3 (k=20),
sigma2_by_k}}}}. Subsamples cells for speed."""
import json
import numpy as np
import torch

BASE = "/path/to/scratch/sigma-taxonomy/results"
# (tag, label, dir): 0.6B + MoE already have top-k; 1.7B/8B/7B from the re-run
MODELS = [
    ("Qwen3-0.6B",    f"{BASE}/sigma_taxonomy_v1"),
    ("Qwen3-1.7B",    f"{BASE}/breadth_topk_qwen17b"),
    ("Qwen3-8B",      f"{BASE}/breadth_topk_m8b"),
    ("Qwen2.5-7B",    f"{BASE}/breadth_topk_m25_7b"),
    ("Qwen3-30B-MoE", f"{BASE}/sigma_taxonomy_moe30"),
]
KERNELS = ["cudnn", "flash", "math"]
REGIMES = ["warm_iso", "warm_serve"]
TOP_KS = [5, 10, 20, 50, 100]
N_CELLS = 1500


def per_iter_trunc_prob(idx_i, val_i, k, tok2pos, S):
    toks = idx_i[:k]
    lg = val_i[:k].astype(np.float64)
    fin = np.isfinite(lg)
    toks, lg = toks[fin], lg[fin]
    if lg.size == 0:
        return np.zeros(S)
    lg -= lg.max()
    p = np.exp(lg); p /= p.sum()
    out = np.zeros(S)
    for j, t in enumerate(toks):
        out[tok2pos[int(t)]] += p[j]
    return out


def sigma_for_pool(path):
    pool = torch.load(path, weights_only=False)
    idx = pool["topk_idx"].numpy()                    # (N,K,n,Ks)
    val = pool["topk_val"].float().numpy()
    N, K, n, Ks = idx.shape
    K_half = K // 2
    rng = np.random.default_rng(0)
    cp = rng.integers(0, N, N_CELLS); ct = rng.integers(0, n, N_CELLS)
    chi2 = {k: [] for k in TOP_KS}
    for ip, it in zip(cp, ct):
        ic = idx[ip, :, it, :]; vc = val[ip, :, it, :]      # (K,Ks)
        for k in TOP_KS:
            if k > Ks:
                continue
            union = np.unique(ic[:, :k].ravel())
            union = union[union >= 0]
            tok2pos = {int(t): j for j, t in enumerate(union)}
            S = union.size
            ip_probs = np.stack([per_iter_trunc_prob(ic[i], vc[i], k, tok2pos, S)
                                 for i in range(K)])         # (K,S)
            p_ref = ip_probs[K_half:].mean(0)
            p_full = ip_probs.mean(0)
            p_full = np.where(p_full > 0, p_full, 1.0)        # avoid /0 (0 only off-union)
            p1 = ip_probs[0]                                   # single forward (K_avg=1)
            chi2[k].append(float(np.sum((p1 - p_ref) ** 2 / p_full)))
    out = {}
    for k in TOP_KS:
        if not chi2[k]:
            continue
        c = float(np.mean(chi2[k]))
        s2 = c / (1.0 + 1.0 / K_half)                         # K_avg=1
        out[str(k)] = round(float(np.sqrt(max(s2, 0))), 5)    # report sigma (=sqrt sigma^2)
    return out, K, Ks


res = {}
for label, d in MODELS:
    res[label] = {}
    for ker in KERNELS:
        res[label][ker] = {}
        for reg in REGIMES:
            path = f"{d}/hf-{ker}__hf-flash__{reg}__decode.pt"
            import os
            if not os.path.exists(path):
                print(f"MISSING {label} {ker} {reg}", flush=True); continue
            try:
                svk, K, Ks = sigma_for_pool(path)
                res[label][ker][reg] = {"sigma_by_k": svk, "K": K, "Ks": Ks}
                print(f"{label:14} {ker:5} {reg:10} K={K} Ks={Ks}  sigma(k=20)={svk.get('20')}", flush=True)
            except Exception as e:
                print(f"ERR {label} {ker} {reg}: {type(e).__name__} {str(e)[:80]}", flush=True)

json.dump(res, open("/path/to/scratch/def3_sigma.json", "w"), indent=1)
print("wrote /path/to/scratch/def3_sigma.json", flush=True)
