"""Sequential pilot-MLMC, SOLID evaluation (v2, 2026-07-02).

Protocol (realizable):
  1. PILOT = first P trajectories, deep-scored: estimate level variances V_l,
     increment variances, and the pilot bias curve bias_hat(r) = |mean Z_r - mean Z_rmax|.
  2. CHOOSE ladder (incl. top level) + Giles allocation for B_eff = B - P*r_max.
  3. Evaluate on the REST against a HELD-OUT deep reference.

v2 evaluation fixes (the v1 replay produced artifacts in BOTH directions):
  - Reference theta comes from a HELD-OUT half of the reps (deepest r there), never
    from the reps the estimator draws from (v1's same-data reference flattered MLMC).
  - Replay track draws WITHOUT replacement and is reported ONLY inside the validity
    window (every level's draw count <= half its stored stock); outside the window a
    with-replacement replay fabricates calls that were never made and its "RMSE" is
    dominated by finite-stock fluctuations (~sqrt(Var/(N_ev*n_blocks)) ~ 0.01 here),
    which is neither arm's true error.
  - FORMULA track (valid at every budget): RMSE = sqrt(bias_pop(r_top)^2 + G/B_eff)
    with population bias measured against the held-out reference and population
    level variances; the pilot only picks the ladder/allocation. This is the primary
    number; the windowed replay is its empirical check where available.

  python -m experiments.inference_engines.analysis.mlmc_pilot_sequential \
      --pt <scorerA.pt> --pt2 <scorerB.pt> [--key target_logp_k5] --out out.json
"""
from __future__ import annotations
import argparse, itertools, json, math
from pathlib import Path

import numpy as np
import torch


def z_blocks(tl_a, tl_b, r):
    """Coupled DM Z per (traj, block) between two scorers' r-averages on disjoint r-blocks."""
    N, K, n = tl_a.shape
    nb = K // r

    def seq(tl):
        t = tl[:, :nb * r].view(N, nb, r, n)
        return (torch.logsumexp(t, 2) - math.log(r)).sum(-1)
    La, Lb = seq(tl_a), seq(tl_b)
    fa, fb = torch.isfinite(La), torch.isfinite(Lb)
    Z = torch.where(fa & fb, torch.tanh((La - Lb).abs() / 2), torch.zeros_like(La))
    return torch.where(fa ^ fb, torch.ones_like(La), Z).numpy()


def y_blocks(tl_a, tl_b, r_f, r_c):
    """Coupled increment Y = Z_{r_f}(block) - Z_{r_c}(prefix of the same block)."""
    N, K, n = tl_a.shape
    nb = K // r_f

    def z_at(r):
        ta = tl_a[:, :nb * r_f].view(N, nb, r_f, n)[:, :, :r]
        tb = tl_b[:, :nb * r_f].view(N, nb, r_f, n)[:, :, :r]
        La = (torch.logsumexp(ta, 2) - math.log(r)).sum(-1)
        Lb = (torch.logsumexp(tb, 2) - math.log(r)).sum(-1)
        fa, fb = torch.isfinite(La), torch.isfinite(Lb)
        Z = torch.where(fa & fb, torch.tanh((La - Lb).abs() / 2), torch.zeros_like(La))
        return torch.where(fa ^ fb, torch.ones_like(La), Z)
    return (z_at(r_f) - z_at(r_c)).numpy()


def giles_alloc(Vs, ladder, budget):
    w = [math.sqrt(max(v, 1e-12) / r) for v, r in zip(Vs, ladder)]
    s = sum(wi * r for wi, r in zip(w, ladder))
    return [max(int(budget * wi / s), 1) for wi in w]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True); ap.add_argument("--pt2", required=True)
    ap.add_argument("--key", default="target_logp")
    ap.add_argument("--pilot", type=int, default=64)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--out", default=None); ap.add_argument("--label", default="")
    a = ap.parse_args()

    def load(p):
        j = torch.load(p, map_location="cpu", weights_only=False)
        d = j.get("done")
        return j[a.key][:, d if d is not None else slice(None)].float()
    ta, tb = load(a.pt), load(a.pt2)
    K = min(ta.shape[1], tb.shape[1]); ta, tb = ta[:, :K], tb[:, :K]
    N = ta.shape[0]; Kh = K // 2
    ta_e, tb_e, ta_r, tb_r = ta[:, :Kh], tb[:, :Kh], ta[:, Kh:], tb[:, Kh:]

    rg = [r for r in [1, 2, 4, 8, 16, 32, 64, 128] if r <= Kh // 2]
    rmax = rg[-1]
    Zb = {r: z_blocks(ta_e, tb_e, r) for r in rg}
    Yb = {(rf, rc): y_blocks(ta_e, tb_e, rf, rc) for rf in rg for rc in rg if rc < rf}
    rref = [r for r in [1, 2, 4, 8, 16, 32, 64, 128, 256] if r <= Kh][-1]
    P = a.pilot
    ev = np.arange(P, N)
    theta = z_blocks(ta_r, tb_r, rref)[ev].mean()                    # held-out deep ref

    pm = {r: Zb[r][:P].mean() for r in rg}
    bh = {r: abs(pm[r] - pm[rmax]) for r in rg}                       # pilot bias curve
    pv = {r: Zb[r][:P].var(ddof=1) for r in rg}
    pY = {k: Yb[k][:P].var(ddof=1) for k in Yb}
    em_ = {r: Zb[r][ev].mean() for r in rg}
    evr = {r: Zb[r][ev].var(ddof=1) for r in rg}
    eY = {k: Yb[k][ev].var(ddof=1) for k in Yb}
    bias_pop = {r: abs(em_[r] - theta) for r in rg}
    cands = [list(c) + [rt] for rt in rg for m in range(3)
             for c in itertools.combinations([r for r in rg if r < rt], m)]
    pilot_cost = P * rmax
    rng = np.random.default_rng(20260702)
    rows = []
    for mult in [2, 3, 4, 6, 8, 12, 16, 24, 32]:
        B = pilot_cost * mult; Beff = B - pilot_cost

        def ps(L):
            Vs = [pv[L[0]] if i == 0 else pY[(L[i], L[i - 1])] for i in range(len(L))]
            return bh[L[-1]] ** 2 + sum(math.sqrt(max(v, 1e-12) * r)
                                        for v, r in zip(Vs, L)) ** 2 / Beff
        Lm = min(cands, key=ps)
        rs = min(rg, key=lambda r: bh[r] ** 2 + pv[r] * r / Beff)
        Ns = giles_alloc([pv[Lm[0]] if i == 0 else pY[(Lm[i], Lm[i - 1])]
                          for i in range(len(Lm))], Lm, Beff)
        # FORMULA track (primary): population bias/variances, pilot's choices
        Vs_pop = [evr[Lm[0]] if i == 0 else eY[(Lm[i], Lm[i - 1])] for i in range(len(Lm))]
        g = sum(math.sqrt(max(v, 1e-12) * r) for v, r in zip(Vs_pop, Lm)) ** 2
        rm_f = math.sqrt(bias_pop[Lm[-1]] ** 2 + g / Beff)
        rs_f = math.sqrt(bias_pop[rs] ** 2 + evr[rs] * rs / Beff)
        # REPLAY track (empirical check, only inside the validity window)
        stocks = [len(ev) * Zb[Lm[0]].shape[1]] + \
                 [len(ev) * Yb[(Lm[i], Lm[i - 1])].shape[1] for i in range(1, len(Lm))]
        ms = max(int(Beff // rs), 1)
        ok = all(nd <= 0.5 * s for nd, s in zip(Ns, stocks)) and \
            ms <= 0.5 * len(ev) * Zb[rs].shape[1]
        rm_r = rs_r = None
        if ok:
            def dnr(arr, m):
                st = arr[ev].reshape(-1)
                return st[rng.choice(len(st), min(m, len(st)), replace=False)].mean()
            em2, es2 = [], []
            for _ in range(a.reps):
                est = dnr(Zb[Lm[0]], Ns[0])
                for i in range(1, len(Lm)):
                    est += dnr(Yb[(Lm[i], Lm[i - 1])], Ns[i])
                em2.append(est - theta); es2.append(dnr(Zb[rs], ms) - theta)
            rm_r = float(np.sqrt(np.mean(np.square(em2))))
            rs_r = float(np.sqrt(np.mean(np.square(es2))))
        rows.append({"B": float(B), "ladder": Lm, "single_r": rs, "alloc": Ns,
                     "adv_formula": rm_f / rs_f,
                     "adv_replay": (rm_r / rs_r) if ok else None, "in_window": ok})

    advf = [r["adv_formula"] for r in rows]
    out = {"label": a.label or Path(a.pt).stem, "key": a.key, "K": K, "N": N, "pilot": P,
           "theta_heldout": float(theta), "r_grid": rg,
           "bias_pop": {str(r): float(bias_pop[r]) for r in rg},
           "VarZ": {str(r): float(evr[r]) for r in rg}, "rows": rows,
           "best_adv_formula": float(min(advf))}
    print(f"[{out['label']}] key={a.key} K={K} N={N} theta(held-out)={theta:.4f}")
    print(f"{'B':>9} {'ladder':>14} {'single':>7} {'adv_formula':>12} {'adv_replay':>11}")
    for r in rows:
        rp = f"{r['adv_replay']:.3f}" if r["adv_replay"] else "  out-of-window"
        print(f"{r['B']:>9.0f} {str(r['ladder']):>14} {r['single_r']:>7} "
              f"{r['adv_formula']:>12.3f} {rp:>11}")
    print(f"  best formula adv = {out['best_adv_formula']:.3f} "
          f"(1.0 = pilot judged single-level optimal)")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1)); print("wrote", a.out)


if __name__ == "__main__":
    main()
