"""Quick acceptance analysis for the decode-path cross-engine pools.

Reads {pi,mu}_pool.pt from --dir and reports, per pool and combined:
  - own-side own-zero rates (acceptance: == 0 — sample/score path consistency)
  - sample_lps vs own-side score rep0 bit-agreement (acceptance: == 1)
  - DM TV estimate with the exact support-mismatch decomposition, vs n
  - self-TV: DM between the two halves of the own-side K repeats (truth 0)

Writes <dir>/quick_decode_analysis.json and prints a summary.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch


def logmeanexp(t, dim):
    return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])


def decomp(L_pi, L_mu):
    fin_pi, fin_mu = torch.isfinite(L_pi), torch.isfinite(L_mu)
    one_sided = fin_pi ^ fin_mu
    both = fin_pi & fin_mu
    Z = torch.zeros_like(L_pi)
    Z[one_sided] = 1.0
    Z[both] = torch.tanh((L_pi[both] - L_mu[both]).abs() / 2)
    return {
        "dm": Z.mean().item(),
        "se": (Z.var(unbiased=True) / Z.shape[0]).sqrt().item(),
        "mismatch": (Z * one_sided.float()).mean().item(),
        "shared": (Z * both.float()).mean().item(),
        "frac_one_sided": one_sided.float().mean().item(),
        "frac_both_zero": (~fin_pi & ~fin_mu).float().mean().item(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--n_grid", default="10,25,50,100,200,300,400,500")
    args = p.parse_args()
    d = Path(args.dir)
    out = {"pools": {}}

    pools = {}
    for side in ["pi", "mu"]:
        pool = torch.load(d / f"{side}_pool.pt", weights_only=False)
        pools[side] = pool
        own = pool[f"score_L_{side}"] if f"score_L_{side}" in pool else None
        own_scores = pool["score_L_pi"] if side == "pi" else pool["score_L_mu"]
        sample_lps = pool.get(f"sample_L_{side}")
        K = own_scores.shape[1]
        rep = {}
        rep["K"] = K
        rep["own_zero_cell_rep0"] = (~torch.isfinite(own_scores[:, 0, :])).float().mean().item()
        rep["own_zero_traj_rep0"] = (~torch.isfinite(own_scores[:, 0, :])).any(-1).float().mean().item()
        if sample_lps is not None:
            both_fin = torch.isfinite(own_scores[:, 0, :]) & torch.isfinite(sample_lps)
            rep["bit_agree_sample_vs_score0"] = (
                (own_scores[:, 0, :] == sample_lps)[both_fin].float().mean().item()
                if both_fin.any() else float("nan"))
        if K >= 2:
            ha = logmeanexp(own_scores[:, :K // 2, :], 1).sum(-1)
            hb = logmeanexp(own_scores[:, K // 2:, :], 1).sum(-1)
            rep["self_tv_halves"] = decomp(ha, hb)["dm"]
            rep["reps_bit_identical"] = bool(
                (own_scores[:, 0, :] == own_scores[:, 1, :]).all().item())
        out["pools"][side] = rep

    n_eval = pools["pi"]["meta"]["n"]
    rows = []
    for n in [int(x) for x in args.n_grid.split(",") if int(x) <= n_eval]:
        per = {}
        for side, pool in pools.items():
            K = pool["score_L_pi"].shape[1]
            L_pi = logmeanexp(pool["score_L_pi"], 1)[:, :n].sum(-1)
            L_mu = logmeanexp(pool["score_L_mu"], 1)[:, :n].sum(-1)
            per[side] = decomp(L_pi, L_mu)
        rows.append({
            "n": n,
            "TV": 0.5 * (per["pi"]["dm"] + per["mu"]["dm"]),
            "TV_se": 0.5 * math.hypot(per["pi"]["se"], per["mu"]["se"]),
            "mismatch": 0.5 * (per["pi"]["mismatch"] + per["mu"]["mismatch"]),
            "per_pool": per,
        })
    out["tv_vs_n"] = rows

    (d / "quick_decode_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["pools"], indent=1))
    for r in rows:
        print(f"n={r['n']:4d}  TV={r['TV']:.4f} ± {r['TV_se']:.4f}  "
              f"mismatch={r['mismatch']:.4f} ({r['mismatch']/max(r['TV'],1e-9):.1%})")
    print(f"[analysis] wrote {d}/quick_decode_analysis.json")


if __name__ == "__main__":
    main()
