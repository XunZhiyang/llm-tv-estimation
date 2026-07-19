"""idev: TV-vs-r to disambiguate the histogram coincidences. From sdpa_pairwise_k64
(K=64, warm_iso, all 4 kernels on flash-source trajectories), compute the symmetric
DM TV at n=500 as a function of repetition r for:
  self-floor : cuDNN vs an independent half of itself  (the noise floor)
  auto-cudnn : tracks the floor -> auto == cuDNN
  math-flash : stays far above the floor -> math != flash
Both auto/cudnn and math/flash coincide in the marginal Delta histogram; the TV
estimator vs r tells them apart. Outputs a tiny JSON."""
# NOTE (provenance): the *_ws.npz/json variants used by the final figures were
# derived from this script by path substitution (recorded 2026-06-24):
#   warm_iso -> warm_serve on the reference/rung cache names; output filename + _ws.

import math, json
import numpy as np, torch

NEG_INF = float("-inf")
D = "/path/to/scratch/sigma-taxonomy/results/sdpa_pairwise_k64"
RG = [1, 2, 4, 8, 16, 32]


def lme(t, d): return torch.logsumexp(t, dim=d) - math.log(t.shape[d])


def tlp(j):
    idx = j["topk_idx"]; val = j["topk_val"].float(); N, K, n, Ks = idx.shape
    plen = j["meta"]["prompt_len"]; tgt = j["X"][:, plen:plen + n].long().view(N, 1, n, 1)
    m = (idx.long() == tgt) & torch.isfinite(val)
    return torch.where(m.any(-1), torch.where(m, val, torch.full_like(val, NEG_INF)).max(-1).values,
                       torch.full((), NEG_INF))


def shuf(s, seed):
    a = s.numpy(); N, K, n = a.shape; c = np.transpose(a, (0, 2, 1)).reshape(N * n, K)
    i = np.argsort(np.random.default_rng(seed).random((N * n, K)), 1)
    sh = np.take_along_axis(c, i, 1)
    return torch.from_numpy(np.ascontiguousarray(sh.reshape(N, n, K).transpose(0, 2, 1)))


def load(name, seed):
    return shuf(tlp(torch.load(f"{D}/{name}.pt", weights_only=False)), seed)


def Lsum(lp, lo, hi):
    """trajectory logp: sum over tokens of logmeanexp over reps[lo:hi]. (N,)"""
    return torch.where(torch.isfinite(lp[:, lo:hi]), lp[:, lo:hi],
                       torch.full_like(lp[:, lo:hi], NEG_INF)).pipe if False else \
        lme(lp[:, lo:hi], 1).sum(1)


def dm_tv(La, Lb):
    """symmetric DM on shared support: Z = tanh(|La-Lb|/2) finite, 1 one-sided, 0 both -inf."""
    fa, fb = torch.isfinite(La), torch.isfinite(Lb)
    z = torch.zeros_like(La)
    both = fa & fb
    z = torch.where(both, torch.tanh((La - Lb).abs() / 2), z)
    z = torch.where(fa ^ fb, torch.ones_like(La), z)
    return float(z.mean())


print("loading kernels (K=64 warm_iso) ...", flush=True)
auto = load("hf-auto__hf-flash__warm_iso__decode", 1)
cud = load("hf-cudnn__hf-flash__warm_iso__decode", 2)
mat = load("hf-math__hf-flash__warm_iso__decode", 3)
fla = load("hf-flash__hf-flash__warm_iso__decode", 4)

res = {"r": RG, "floor": [], "auto_cudnn": [], "math_flash": []}
for r in RG:
    # cross: r-avg each kernel over first r reps
    La = lme(auto[:, :r], 1).sum(1); Lc = lme(cud[:, :r], 1).sum(1)
    Lm = lme(mat[:, :r], 1).sum(1); Lf = lme(fla[:, :r], 1).sum(1)
    # self-floor: cudnn first-r vs last-r (need 2r <= 64)
    Lc2 = lme(cud[:, r:2 * r], 1).sum(1) if 2 * r <= 64 else lme(cud[:, -r:], 1).sum(1)
    res["floor"].append(round(dm_tv(Lc, Lc2), 5))
    res["auto_cudnn"].append(round(dm_tv(La, Lc), 5))
    res["math_flash"].append(round(dm_tv(Lm, Lf), 5))
    print(f"  r={r:>3}  floor={res['floor'][-1]:.4f}  auto-cudnn={res['auto_cudnn'][-1]:.4f}  "
          f"math-flash={res['math_flash'][-1]:.4f}", flush=True)

json.dump(res, open("/path/to/scratch/tv_vs_r_disambig.json", "w"), indent=1)
print("wrote /path/to/scratch/tv_vs_r_disambig.json", flush=True)
