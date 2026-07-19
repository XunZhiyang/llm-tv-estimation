"""idev: Fig 2a ladder of per-token Delta distributions against a fixed
reference mu = cuDNN warm_iso (Qwen3-0.6B). Five rungs:
  self (cuDNN vs an independent half of itself) -> floor spike
  auto warm_iso (auto vs cuDNN)                 -> identity spike (auto == cuDNN)
  cuDNN warm_serve                              -> regime/batch difference
  flash warm_iso                                -> kernel difference
  math  warm_iso                                -> kernel difference
Delta_t = logmeanexp_r(pi) - logmeanexp_r(mu), r=8 (K=16 half), on the shared
hf-flash trajectories. Outputs a tiny npz (histograms + std)."""
# NOTE (provenance): the *_ws.npz/json variants used by the final figures were
# derived from this script by path substitution (recorded 2026-06-24):
#   warm_iso -> warm_serve on the reference/rung cache names; output filename + _ws.

import math
import numpy as np
import torch

NEG_INF = float("-inf")
D = "/path/to/scratch/sigma-taxonomy/results/sigma_taxonomy_v1"
R = 8
BINS = np.linspace(-0.25, 0.25, 141)


def logmeanexp(t, dim):
    return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])


def target_logprob(job):
    if "target_logp" in job:
        return job["target_logp"].float()
    idx = job["topk_idx"]; val = job["topk_val"].float()
    N, K, n, Ks = idx.shape
    plen = job["meta"]["prompt_len"]
    cap = max(1, min(int(job["meta"].get("logprob_cap", Ks)), Ks))
    idx = idx[..., :cap]; val = val[..., :cap]
    tgt = job["X"][:, plen:plen + n].long().view(N, 1, n, 1)
    match = (idx.long() == tgt) & torch.isfinite(val)
    has = match.any(-1)
    picked = torch.where(match, val, torch.full_like(val, NEG_INF)).max(-1).values
    return torch.where(has, picked, torch.full((), NEG_INF))


def shuffle_K(s, seed):
    a = s.numpy(); N, K, n = a.shape
    c = np.transpose(a, (0, 2, 1)).reshape(N * n, K)
    idx = np.argsort(np.random.default_rng(seed).random((N * n, K)), axis=1)
    sh = np.take_along_axis(c, idx, axis=1)
    return torch.from_numpy(np.ascontiguousarray(sh.reshape(N, n, K).transpose(0, 2, 1)))


def load_lp(name, seed):
    job = torch.load(f"{D}/{name}.pt", weights_only=False)
    return shuffle_K(target_logprob(job), seed)


print("loading mu = cuDNN warm_iso ...", flush=True)
mu = load_lp("hf-cudnn__hf-flash__warm_iso__decode", 1)
Lmu = logmeanexp(mu[:, :R], 1)                       # reference, first half
Lmu2 = logmeanexp(mu[:, R:2 * R], 1)                 # independent second half (for self)

RUNGS = [
    ("self (cuDNN vs itself)", None),                # uses Lmu vs Lmu2
    ("auto warm_iso", "hf-auto__hf-flash__warm_iso__decode"),
    ("cuDNN warm_serve", "hf-cudnn__hf-flash__warm_serve__decode"),
    ("flash warm_iso", "hf-flash__hf-flash__warm_iso__decode"),
    ("math warm_iso", "hf-math__hf-flash__warm_iso__decode"),
]
out = {"bins": BINS.astype(np.float32)}
labels = []
for i, (lab, name) in enumerate(RUNGS):
    if name is None:
        d = (Lmu - Lmu2).numpy().ravel()
    else:
        Lpi = logmeanexp(load_lp(name, 10 + i)[:, :R], 1)
        d = (Lpi - Lmu).numpy().ravel()
    d = d[np.isfinite(d)]
    out[f"h{i}"] = np.histogram(d, bins=BINS, density=True)[0].astype(np.float32)
    out[f"s{i}"] = np.float32(np.std(d))
    labels.append(lab)
    print(f"  rung{i} {lab:28} std={np.std(d):.4f}", flush=True)

out["labels"] = np.array(labels)
np.savez("/path/to/scratch/fig2a_ladder.npz", **out)
print("wrote /path/to/scratch/fig2a_ladder.npz", flush=True)
