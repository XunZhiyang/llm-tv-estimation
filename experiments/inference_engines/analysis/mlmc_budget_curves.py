"""RMSE-vs-budget curves for fixed MLMC configs vs single-level, on real pairs.

Population formula track (same machinery as mlmc_pilot_sequential):
  reps split in half -> first half provides Z/Y blocks (estimation stock),
  second half provides the deep reference theta;
  rmse_single(r, B)  = sqrt(bias_pop(r)^2 + Var[Z_r] * r / B_draws)
  rmse_ladder(L, B)  = sqrt(bias_pop(L[-1])^2 + (sum_i sqrt(V_i r_i))^2 / B_draws)
with Giles-optimal continuous allocation. x-axis reported as prefix-oracle
queries per side (= B_draws * n).

Repeat-axis hygiene (--kshuffle, default "percell"): serving stacks can be
serially correlated along the repeat axis, so all repeat-axis statistics are
computed after an independent uniform permutation of the repeats at each
(trajectory, position) cell, averaging the per-configuration RMSE curves over
--kshuffle-seeds independent permutations (declared protocol of the paper's
"Repeat-axis hygiene" paragraph). --kshuffle none reproduces the raw
serving-order computation (the pre-2026-07-11 behavior).

  python -m experiments.inference_engines.analysis.mlmc_budget_curves \
      --pt A.pt --pt2 B.pt --key target_logp --single 16 --ladder 16,64 \
      --label "cuDNN-flash warm-served" --out fig.png

Every from-pools run stores the plotted curves under "figdata" in --json;
--replot re-renders the figure from that committed JSON without the raw pools
(public code release path).

Readout convention (paper Appendix B.6): the default --key target_logp is the
engines' raw log-softmax on the stored top-100 support — the paper's "k = 100
readout". Renormalizing over that support (target_logp_k100 - logZ_k100)
changes the estimand by < 0.002 on these pools; the k = 20 renormalized
readout (target_logp_k20 - logZ_k20) is a different, mismatch-dominated
estimand and is deliberately not used here.
"""
from __future__ import annotations
import argparse, json, math, zlib
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import itertools

from experiments.inference_engines.analysis.mlmc_pilot_sequential import z_blocks, y_blocks

KSHUFFLE_MASTER_SEED = 20260710


def _load(p, key):
    j = torch.load(p, map_location="cpu", weights_only=False)
    d = j.get("done"); t = j[key]
    return (t[:, d] if d is not None else t).float()


def pair_seeds(pt, pt2, n_seeds, master=KSHUFFLE_MASTER_SEED):
    """2*n_seeds child seeds (one per side per shuffle replicate), tagged by the
    pool file names so every (pair, master) combination has its own stream."""
    tag = f"kshuffle|{Path(pt).name}|{Path(pt2).name}"
    base = np.random.SeedSequence([master, zlib.crc32(tag.encode())])
    return [int(s.generate_state(1)[0]) for s in base.spawn(2 * n_seeds)]


def shuffle_per_cell(t, seed):
    """Independent uniform permutation of the repeat axis at each (trajectory,
    position) cell — the paper's declared repeat-axis hygiene."""
    N, K, n = t.shape
    g = torch.Generator().manual_seed(int(seed) % (2**63 - 1))
    idx = torch.argsort(torch.rand(N, n, K, generator=g), dim=-1)   # (N, n, K)
    return torch.gather(t, 1, idx.permute(0, 2, 1))


def prep(pt, pt2, key):
    ta = pt if torch.is_tensor(pt) else _load(pt, key)
    tb = pt2 if torch.is_tensor(pt2) else _load(pt2, key)
    K = min(ta.shape[1], tb.shape[1]); ta, tb = ta[:, :K], tb[:, :K]
    n_tok = ta.shape[-1]
    Kh = K // 2
    ta_e, tb_e, ta_r, tb_r = ta[:, :Kh], tb[:, :Kh], ta[:, Kh:], tb[:, Kh:]
    rg = [r for r in [1, 2, 4, 8, 16, 32, 64] if r <= Kh // 2]
    rref = [r for r in [1, 2, 4, 8, 16, 32, 64, 128] if r <= Kh][-1]
    zref = np.asarray(z_blocks(ta_r, tb_r, rref))
    theta = float(zref.mean())
    bias = {}
    for r in rg:
        d = np.asarray(z_blocks(ta_e, tb_e, r)) - zref          # paired per-trajectory
        se = float(d.std(ddof=1)) / math.sqrt(len(d))
        bias[r] = max(abs(float(d.mean())), se)                 # can't claim bias < its SE
    vz = {r: float(np.asarray(z_blocks(ta_e, tb_e, r)).var(ddof=1)) for r in rg}
    vy = {(rf, rc): float(np.asarray(y_blocks(ta_e, tb_e, rf, rc)).var(ddof=1))
          for rf in rg for rc in rg if rc < rf}
    return rg, theta, bias, vz, vy, n_tok


def rmse_single(r, B, bias, vz):
    return math.sqrt(bias[r] ** 2 + vz[r] * r / B)


def rmse_ladder(L, B, bias, vz, vy):
    Vs = [vz[L[0]] if i == 0 else vy[(L[i], L[i - 1])] for i in range(len(L))]
    g = sum(math.sqrt(max(v, 1e-12) * r) for v, r in zip(Vs, L)) ** 2
    return math.sqrt(bias[L[-1]] ** 2 + g / B)


def pilot_adaptive(pt, pt2, key, Bs, P=64, draws=40):
    """mean RMSE of the pilot-chosen config (single OR ladder) at each budget."""
    ta = pt if torch.is_tensor(pt) else _load(pt, key)
    tb = pt2 if torch.is_tensor(pt2) else _load(pt2, key)
    K = min(ta.shape[1], tb.shape[1]); Kh = K // 2
    ta_e, tb_e, ta_r, tb_r = ta[:, :Kh], tb[:, :Kh], ta[:, Kh:], tb[:, Kh:]
    rg = [r for r in [1, 2, 4, 8, 16, 32, 64] if r <= Kh // 2]
    rmax = rg[-1]
    rref = [r for r in [1, 2, 4, 8, 16, 32, 64, 128] if r <= Kh][-1]
    zref = np.asarray(z_blocks(ta_r, tb_r, rref))
    Zb = {r: np.asarray(z_blocks(ta_e, tb_e, r)) for r in rg}
    Yb = {(rf, rc): np.asarray(y_blocks(ta_e, tb_e, rf, rc))
          for rf in rg for rc in rg if rc < rf}
    N = len(zref)
    bias, evr, eY = {}, {r: Zb[r].var(ddof=1) for r in rg}, {k: v.var(ddof=1) for k, v in Yb.items()}
    for r in rg:
        d = Zb[r] - zref
        bias[r] = max(abs(float(d.mean())), float(d.std(ddof=1)) / math.sqrt(N))
    cands = [list(c) + [rt] for rt in rg for m in range(3)
             for c in itertools.combinations([r for r in rg if r < rt], m)]
    def rmse_pop(L, B):
        Vs = [evr[L[0]] if i == 0 else eY[(L[i], L[i - 1])] for i in range(len(L))]
        g = sum(math.sqrt(max(v, 1e-12) * r) for v, r in zip(Vs, L)) ** 2
        return math.sqrt(bias[L[-1]] ** 2 + g / B)
    rng = np.random.default_rng(20260703)
    accL = np.zeros(len(Bs)); accS = np.zeros(len(Bs)); cnt = np.zeros(len(Bs))
    for _ in range(draws):
        pi = rng.choice(N, P, replace=False)
        pm = {r: Zb[r][pi].mean() for r in rg}
        bh = {r: abs(pm[r] - pm[rmax]) for r in rg}
        pv = {r: Zb[r][pi].var(ddof=1) for r in rg}
        pY = {k: Yb[k][pi].var(ddof=1) for k in Yb}
        for i, B in enumerate(Bs):
            Beff = B          # pilot draws are recycled into the estimate
            if Beff < P * rmax:   # need at least the pilot itself
                continue
            def ps(L):
                Vs = [pv[L[0]] if j == 0 else pY[(L[j], L[j - 1])] for j in range(len(L))]
                return bh[L[-1]] ** 2 + sum(math.sqrt(max(v, 1e-12) * r)
                                            for v, r in zip(Vs, L)) ** 2 / Beff
            Lc = min(cands, key=ps)
            rc = min(rg, key=lambda r: bh[r] ** 2 + pv[r] * r / Beff)
            accL[i] += rmse_pop(Lc, Beff); accS[i] += rmse_pop([rc], Beff); cnt[i] += 1
    good = cnt > 0
    return (np.where(good, accL / np.maximum(cnt, 1), np.nan),
            np.where(good, accS / np.maximum(cnt, 1), np.nan))


def peak_saving(base, xs_base, ld, xs_ld):
    """Peak equal-accuracy budget saving of curve `ld` vs curve `base`
    (max over matched-RMSE targets of budget_base / budget_ld)."""
    fin = np.isfinite(np.asarray(ld))
    ldv = [float(v) for v, f in zip(ld, fin) if f]
    xv = [float(x) for x, f in zip(xs_ld, fin) if f]
    finb = np.isfinite(np.asarray(base))
    env = [float(v) for v, f in zip(base, finb) if f]
    xb = [float(x) for x, f in zip(xs_base, finb) if f]
    lo = max(min(ldv), min(env)); hi = min(max(ldv), max(env))
    save = 0.0
    for t in np.geomspace(lo * 1.01, hi * 0.99, 40):
        bx = np.interp(math.log(t), [math.log(v) for v in env[::-1]],
                       [math.log(x) for x in xb[::-1]])
        gx = np.interp(math.log(t), [math.log(v) for v in ldv[::-1]],
                       [math.log(x) for x in xv[::-1]])
        save = max(save, math.exp(bx) / math.exp(gx))
    return save


def panel_data(pt, pt2, key, singles, ladder, title, bmax=262144, ylog=False,
               kshuffle="percell", kseeds=20, kmaster=KSHUFFLE_MASTER_SEED):
    ta, tb = _load(pt, key), _load(pt2, key)
    Bs = np.geomspace(2048, bmax, 15) * 1.0001
    n_reps = kseeds if kshuffle == "percell" else 1
    seeds = pair_seeds(pt, pt2, kseeds, kmaster) if kshuffle == "percell" else []
    runs = []
    for i in range(n_reps):
        va, vb = ((shuffle_per_cell(ta, seeds[2 * i]), shuffle_per_cell(tb, seeds[2 * i + 1]))
                  if kshuffle == "percell" else (ta, tb))
        rg, theta_i, bias_i, vz_i, vy_i, n_tok = prep(va, vb, key)
        sing_i = {r: [rmse_single(r, B, bias_i, vz_i) for B in Bs] for r in singles}
        env_i = [min(v) for v in zip(*sing_i.values())]
        plL_i, plS_i = pilot_adaptive(va, vb, key, Bs)
        runs.append({"theta": theta_i, "bias": bias_i, "vz": vz_i, "sing": sing_i,
                     "env": env_i, "plL": plL_i, "plS": plS_i,
                     "save": peak_saving(env_i, Bs, plL_i, Bs)})
    # pointwise mean of the per-seed RMSE curves (identity for kshuffle="none")
    theta = float(np.mean([u["theta"] for u in runs]))
    bias = {r: float(np.mean([u["bias"][r] for u in runs])) for r in runs[0]["bias"]}
    vz = {r: float(np.mean([u["vz"][r] for u in runs])) for r in runs[0]["vz"]}
    sing = {r: np.mean([u["sing"][r] for u in runs], axis=0) for r in singles}
    env = [min(v) for v in zip(*sing.values())]
    plL = np.mean([u["plL"] for u in runs], axis=0)
    plS = np.mean([u["plS"] for u in runs], axis=0)
    xs = Bs * n_tok / 1e6
    # PEAK equal-accuracy saving: pilot multilevel vs the single-family envelope
    save = peak_saving(env, Bs, plL, Bs)
    save_ps = peak_saving(plS, Bs, plL, Bs)
    fin = np.isfinite(plL) & np.isfinite(plS)
    ratio_min = float((plL[fin] / plS[fin]).min())
    print(f"[{title}] theta={theta:.4f} bias={ {r: round(b,4) for r,b in bias.items()} } saving={save:.3f}x")
    summary = {"theta": theta, "bias": bias, "VarZ": vz, "saving_equal_acc_vs_best": save,
               "saving_vs_pilot_single": save_ps, "min_rmse_ratio_vs_pilot_single": ratio_min,
               "kshuffle": {"mode": kshuffle, "n_seeds": n_reps if kshuffle == "percell" else 0,
                            "master_seed": kmaster if kshuffle == "percell" else None,
                            "per_seed_saving_vs_best": [u["save"] for u in runs]},
               "singles": singles, "ladder": ladder, "n_tok": n_tok}
    figdata = {"title": title, "ylog": ylog, "singles": [int(r) for r in singles],
               "xs": [float(v) for v in xs],
               "sing": {str(r): [float(v) for v in sing[r]] for r in singles},
               "plL": [float(v) for v in plL], "save": float(save)}
    return summary, figdata


def draw_panel(ax, fd):
    xs, singles = np.asarray(fd["xs"]), fd["singles"]
    greys = plt.cm.Greys(np.linspace(0.45, 0.95, len(singles)))
    for r, c in zip(singles, greys):
        ax.plot(xs, fd["sing"][str(r)], "--o", color=c, lw=1.2, ms=3.2,
                label=f"single-level $r$={r}")
    ax.plot(xs, fd["plL"], "-^", color="#2a9d5c", lw=1.9, ms=5,
            label="multilevel (pilot-chosen per budget)")
    ax.set_xscale("log")
    if fd["ylog"]:
        ax.set_yscale("log")
    ax.set_xlabel(r"oracle queries / side ($\times 10^6$)")
    ax.set_ylabel("RMSE of the estimate")
    ax.set_title(fd["title"], fontsize=10)
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    msg = (f"peak equal-accuracy saving vs best single ≈ {fd['save']:.2f}×" if fd["save"] >= 1.05
           else "tracks the best fixed $r$ at every budget (without knowing it)")
    ax.annotate(msg, (0.03, 0.05), xycoords="axes fraction", fontsize=8, color="#2a9d5c")


SCHED = [([16], "#2a9d5c", "{16} (single)"), ([4, 16], "#3a6ea5", "{4,16}"),
         ([1, 4, 16], "#8a5fb0", "{1,4,16}"), ([1, 16], "#c07f30", "{1,16}")]


def ladder_data(pt, pt2, key="target_logp",
                kshuffle="percell", kseeds=20, kmaster=KSHUFFLE_MASTER_SEED):
    """estimate (mean + 1 s.d. upper edge) vs budget for different ladders:
    same top level -> same answer; inner levels only change the speed."""
    ta0, tb0 = _load(pt, key), _load(pt2, key)
    K = min(ta0.shape[1], tb0.shape[1]); Kh = K // 2
    rg = [r for r in [1, 2, 4, 8, 16, 32] if r <= Kh // 2]
    rref = [r for r in [1, 2, 4, 8, 16, 32, 64] if r <= Kh][-1]
    Bs = np.geomspace(2048, 524288, 40); x = Bs * 500 / 1e6

    def curves(ta, tb):
        ta_e, tb_e, ta_r, tb_r = ta[:, :Kh], tb[:, :Kh], ta[:, Kh:], tb[:, Kh:]
        Zb = {r: np.asarray(z_blocks(ta_e, tb_e, r)) for r in rg}
        Yb = {(rf, rc): np.asarray(y_blocks(ta_e, tb_e, rf, rc))
              for rf in rg for rc in rg if rc < rf}
        theta = float(np.asarray(z_blocks(ta_r, tb_r, rref)).mean())
        em = {r: float(Zb[r].mean()) for r in rg}
        evr = {r: float(Zb[r].var(ddof=1)) for r in rg}
        eY = {k: float(v.var(ddof=1)) for k, v in Yb.items()}

        def bandw(L, B):
            Vs = [evr[L[0]] if i == 0 else eY[(L[i], L[i - 1])] for i in range(len(L))]
            return math.sqrt(sum(math.sqrt(max(v, 1e-12) * r) for v, r in zip(Vs, L)) ** 2 / B)

        cs = {tuple(L): np.array([em[L[-1]] + bandw(L, B) for B in Bs]) for L, _, _ in SCHED}
        cs[(2,)] = np.array([em[2] + bandw([2], B) for B in Bs])
        return theta, em, cs

    n_reps = kseeds if kshuffle == "percell" else 1
    seeds = pair_seeds(pt, pt2, kseeds, kmaster) if kshuffle == "percell" else []
    runs = []
    for i in range(n_reps):
        va, vb = ((shuffle_per_cell(ta0, seeds[2 * i]), shuffle_per_cell(tb0, seeds[2 * i + 1]))
                  if kshuffle == "percell" else (ta0, tb0))
        runs.append(curves(va, vb))
    theta = float(np.mean([u[0] for u in runs]))
    em = {r: float(np.mean([u[1][r] for u in runs])) for r in rg}
    cs = {k: np.mean([u[2][k] for u in runs], axis=0) for k in runs[0][2]}
    print(f"[ladder panel] theta={theta:.4f} em={ {r: round(v, 4) for r, v in em.items()} }")
    return {"x": [float(v) for v in x],
            "curves": {",".join(map(str, L)): [float(v) for v in cs[tuple(L)]] for L, _, _ in SCHED},
            "curve2": [float(v) for v in cs[(2,)]],
            "em2": em[2], "em16": em[16], "theta": theta}


def draw_ladder(ax, fd):
    x, em2, em16 = np.asarray(fd["x"]), fd["em2"], fd["em16"]
    for L, c, lab in SCHED:
        ax.plot(x, fd["curves"][",".join(map(str, L))], color=c, lw=1.6, label=lab)
    ax.axhline(em16, color="0.3", lw=0.9, ls=":")
    ax.plot(x, fd["curve2"], color="#b04a4a", ls="--", lw=1.3,
            alpha=0.8, label="{2} (shallow top)")
    ax.axhline(em2, color="#b04a4a", lw=0.8, ls=":", alpha=0.6)
    ax.text(0.02, em2 + 0.0008, r"biased plateau $\theta_2$ (top level $r=2$)",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color="#b04a4a")
    ax.text(0.98, em16 + 0.0005, r"$\theta_{16}$ (shared top level $r=16$)",
            transform=ax.get_yaxis_transform(), ha="right", fontsize=7.5, color="0.25")
    ax.set_xscale("log")
    ax.set_xlabel(r"oracle queries / side ($\times 10^6$)")
    ax.set_ylabel(r"estimated $\theta$ (mean + 1 s.d.)")
    ax.set_title("(b) schedules sharing a top level ($B=256$ rescoring)", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right", title="schedule", title_fontsize=7.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/report_2026-06/figures/v3/fig4c_budget_curves.png")
    ap.add_argument("--json", default="results/mlmc_seq_2026-07/budget_curves.json")
    ap.add_argument("--kshuffle", default="percell", choices=["none", "percell"],
                    help="repeat-axis hygiene: 'percell' (declared protocol, default) "
                         "shuffles the repeats independently at each (trajectory, position) "
                         "cell and averages RMSE curves over --kshuffle-seeds permutations; "
                         "'none' reproduces the raw serving-order computation")
    ap.add_argument("--kshuffle-seeds", type=int, default=20)
    ap.add_argument("--kshuffle-master-seed", type=int, default=KSHUFFLE_MASTER_SEED)
    ap.add_argument("--replot", action="store_true",
                    help="re-render the figure from the 'figdata' section of --json "
                         "(no raw pools needed)")
    a = ap.parse_args()
    if a.replot:
        res = json.loads(Path(a.json).read_text())
        fdA, fdB = res["figdata"]["panel_a"], res["figdata"]["panel_b"]
        print(f"[replot] curves loaded from {a.json}")
    else:
        ks = dict(kshuffle=a.kshuffle, kseeds=a.kshuffle_seeds, kmaster=a.kshuffle_master_seed)
        res = {"kshuffle": {"mode": a.kshuffle, "n_seeds": a.kshuffle_seeds,
                            "master_seed": a.kshuffle_master_seed}}
        res["engine_n2000"], fdA = panel_data(
            "results/figruns_2026-07/n2000_deep/vllm__vllm_n2kpi__fixedB512__decode.pt",
            "results/figruns_2026-07/n2000_deep/sglang__sglang_n2kpi__fixedB512__decode.pt",
            "target_logp", [1, 2, 4, 8, 16], [1, 8],
            "(a) budget efficiency: vllm vs sglang, n=2000", bmax=2097152, ylog=True, **ks)
        fdB = ladder_data(
            "results/figruns_2026-07/mlmc_engine_deep/vllm__vllm_pi_deep__fixedB256__decode.pt",
            "results/figruns_2026-07/mlmc_engine_deep/sglang__sglang_pi_deep__fixedB256__decode.pt",
            **ks)
        res["figdata"] = {"panel_a": fdA, "panel_b": fdB}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    draw_panel(axes[0], fdA)
    draw_ladder(axes[1], fdB)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=180)
    fig.savefig(Path(a.out).with_suffix(".pdf"))
    pfd = Path("paper_final_draft/figures")
    if pfd.exists():   # paper tree, absent in the public code release
        fig.savefig(pfd / "fig4_mlmc.png", dpi=180)
        fig.savefig(pfd / "fig4_mlmc.pdf")
    if not a.replot:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))
    print("wrote", a.out, "and" if not a.replot else "(replot;", a.json, ")" if a.replot else "")


if __name__ == "__main__":
    main()
