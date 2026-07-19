"""Validate the correction-variance diagnostic with convergence curves.

For several synthetic noise levels, compare:
  - one-level fixed K=256 mixture estimator,
  - two-level MLMC 32 -> 256,
  - three-level MLMC 8 -> 32 -> 256.

The figure annotates the coupled correction-variance ratios so we can check
whether rho predicts when adding levels helps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from experiments.synthetic.estimators import (
    _paired_Z,
    budget_direct_m,
    budget_mlmc,
    direct_m_estimate,
    mlmc_estimate,
    symmetric_Z,
)
from experiments.synthetic.lower_bound_instance import LowerBoundConfig, LowerBoundTreeInstance


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def summarize(vals: list[float], truth: float) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    err = np.abs(arr - truth)
    return {
        "estimate_mean": float(arr.mean()),
        "estimate_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "estimate_se": float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        "abs_error_mean": float(err.mean()),
        "abs_error_std": float(err.std(ddof=1)) if len(err) > 1 else 0.0,
        "abs_error_se": float(err.std(ddof=1) / np.sqrt(len(err))) if len(err) > 1 else 0.0,
    }


def scaled_levels(r_values: list[int], ratios: list[int], scale: int) -> list[tuple[int, int]]:
    return [(max(1, int(round(scale * ratio))), r) for ratio, r in zip(ratios, r_values)]


def correction_rhos(
    inst: LowerBoundTreeInstance,
    *,
    sigma: float,
    mode: str,
    rng: np.random.Generator,
    N_each: int,
) -> dict[str, float]:
    X = np.concatenate([inst.sample("p", N_each, rng), inst.sample("q", N_each, rng)], axis=0)
    Lp, Lq = _paired_Z(inst, X, 256, rng, mode=mode, sigma=sigma)
    Z1 = symmetric_Z(Lp[:, 0], Lq[:, 0])
    Z8 = symmetric_Z(Lp[:, 7], Lq[:, 7])
    Z32 = symmetric_Z(Lp[:, 31], Lq[:, 31])
    Z256 = symmetric_Z(Lp[:, 255], Lq[:, 255])

    def ratio(low: np.ndarray, high: np.ndarray) -> float:
        var_high = float(np.var(high, ddof=1))
        if var_high == 0.0:
            return 0.0
        return float(np.var(high - low, ddof=1) / var_high)

    return {
        "rho_1_8": ratio(Z1, Z8),
        "rho_8_32": ratio(Z8, Z32),
        "rho_32_256": ratio(Z32, Z256),
        "mean_correction_1_8": float(np.mean(Z8 - Z1)),
        "mean_correction_8_32": float(np.mean(Z32 - Z8)),
        "mean_correction_32_256": float(np.mean(Z256 - Z32)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results/synthetic_rho_regime_p040_4lvl_3col")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--r", type=int, default=12)
    p.add_argument("--p_active", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.49)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--mode", default="prob_gaussian", choices=["prob_gaussian", "gaussian_logit", "onehot", "exact"])
    p.add_argument("--sigmas", default="0.04,0.5,0.8")
    p.add_argument("--scales", default="1,2,4,8,16,32")
    p.add_argument("--reps", type=int, default=24)  # 24 = the committed paper run
    p.add_argument("--diag_N_each", type=int, default=4096)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = LowerBoundConfig(
        n=args.n,
        r=args.r,
        p=args.p_active,
        alpha=args.alpha,
        seed=args.seed,
    )
    inst = LowerBoundTreeInstance(cfg)
    sigmas = parse_floats(args.sigmas)
    scales = parse_ints(args.scales)
    rng_master = np.random.default_rng(args.seed + 202605061)

    methods = {
        "one_level_K256": {
            "label": "1-level, K=256",
            "color": "#4e79a7",
            "marker": "o",
        },
        "two_level_32_256": {
            "label": "2-level, 32→256",
            "color": "#f28e2b",
            "marker": "s",
            "r_values": [32, 256],
            "ratios": [8, 1],
        },
        "three_level_8_32_256": {
            "label": "3-level, 8→32→256",
            "color": "#d62728",
            "marker": "*",
            "r_values": [8, 32, 256],
            "ratios": [30, 8, 1],
        },
        "four_level_1_8_32_256": {
            "label": "4-level, 1→8→32→256",
            "color": "#59a14f",
            "marker": "D",
            "r_values": [1, 8, 32, 256],
            "ratios": [171, 30, 8, 1],
        },
    }

    rows: list[dict] = []
    rho_rows: list[dict] = []

    print(
        f"[rho-conv] true_tv={inst.true_tv:.6f}, mode={args.mode}, reps={args.reps}",
        flush=True,
    )
    for sigma in sigmas:
        print(f"[rho-conv] sigma={sigma:g}", flush=True)
        rhos = correction_rhos(
            inst,
            sigma=sigma,
            mode=args.mode,
            rng=np.random.default_rng(rng_master.integers(2**32)),
            N_each=args.diag_N_each,
        )
        rho_rows.append({"sigma": sigma, **rhos})

        for scale in scales:
            for key, method in methods.items():
                if key == "one_level_K256":
                    # Match the one-level budget to the 2-level scale grid.
                    target_budget = budget_mlmc(
                        inst,
                        scaled_levels(
                            methods["two_level_32_256"]["r_values"],
                            methods["two_level_32_256"]["ratios"],
                            scale,
                        ),
                    )
                    N_each = max(1, target_budget // (inst.n * 2 * (1 + 2 * 256)))
                    budget = budget_direct_m(inst, N_each, 256)
                    vals = [
                        direct_m_estimate(
                            inst,
                            N_each,
                            256,
                            np.random.default_rng(rng_master.integers(2**32)),
                            mode=args.mode,
                            sigma=sigma,
                        )
                        for _ in range(args.reps)
                    ]
                    rows.append(
                        {
                            "sigma": sigma,
                            "method": key,
                            "scale": scale,
                            "K": 256,
                            "N_each": int(N_each),
                            "budget": int(budget),
                            "reps": args.reps,
                            **summarize(vals, inst.true_tv),
                        }
                    )
                    continue

                levels = scaled_levels(method["r_values"], method["ratios"], scale)
                budget = budget_mlmc(inst, levels)
                vals = [
                    mlmc_estimate(
                        inst,
                        levels,
                        np.random.default_rng(rng_master.integers(2**32)),
                        mode=args.mode,
                        sigma=sigma,
                    )
                    for _ in range(args.reps)
                ]
                rows.append(
                    {
                        "sigma": sigma,
                        "method": key,
                        "scale": scale,
                        "levels": levels,
                        "budget": int(budget),
                        "reps": args.reps,
                        **summarize(vals, inst.true_tv),
                    }
                )

    payload = {
        "run_info": {
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": _git_value(["rev-parse", "HEAD"]),
            "git_branch": _git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
            "device": "cpu",
        },
        "config": {
            **cfg.__dict__,
            "true_tv": inst.true_tv,
            "active_blocks": int(inst.active.sum()),
            "mode": args.mode,
            "sigmas": sigmas,
            "scales": scales,
            "reps": args.reps,
            "diag_N_each": args.diag_N_each,
            "methods": methods,
        },
        "rho_rows": rho_rows,
        "rows": rows,
    }
    with open(out_dir / "rho_regime_convergence.json", "w") as f:
        json.dump(payload, f, indent=2)

    fig, axes = plt.subplots(1, len(sigmas), figsize=(4.4 * len(sigmas), 4.0), sharex=False)
    if len(sigmas) == 1:
        axes = np.asarray([axes])

    for col, sigma in enumerate(sigmas):
        rho = next(r for r in rho_rows if r["sigma"] == sigma)
        title = (
            f"$\\sigma$={sigma:g}\n"
            + r"$\rho_{1\to8}$="
            + f"{rho['rho_1_8']:.3f}, "
            + r"$\rho_{8\to32}$="
            + f"{rho['rho_8_32']:.3f}, "
            + r"$\rho_{32\to256}$="
            + f"{rho['rho_32_256']:.3f}"
        )
        for key, method in methods.items():
            mr = sorted(
                [r for r in rows if r["sigma"] == sigma and r["method"] == key],
                key=lambda x: x["budget"],
            )
            axes[col].errorbar(
                [r["budget"] for r in mr],
                [r["abs_error_mean"] for r in mr],
                yerr=[r["abs_error_se"] for r in mr],
                color=method["color"],
                marker=method["marker"],
                linewidth=2.0,
                capsize=2,
                label=method["label"],
            )

        axes[col].set_title(title, fontsize=11)
        axes[col].set_xscale("log")
        axes[col].grid(True, alpha=0.3)
        axes[col].set_xlabel("Oracle-call budget proxy")

    axes[0].set_ylabel("Mean absolute error")
    axes[-1].legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle("Synthetic rho-regime validation: level count vs budget")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_regime_convergence.png", dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / "rho_regime_convergence.pdf", bbox_inches="tight")

    print(f"[rho-conv] wrote {out_dir / 'rho_regime_convergence.json'}")
    print(f"[rho-conv] wrote {out_dir / 'rho_regime_convergence.png'}")
    print("\n[summary]")
    for sigma in sigmas:
        rho = next(r for r in rho_rows if r["sigma"] == sigma)
        print(
            f"sigma={sigma:g}: rho1->8={rho['rho_1_8']:.4f}, "
            f"rho8->32={rho['rho_8_32']:.4f}, "
            f"rho32->256={rho['rho_32_256']:.4f}"
        )
        for key, method in methods.items():
            mr = [r for r in rows if r["sigma"] == sigma and r["method"] == key]
            best = min(mr, key=lambda x: x["abs_error_mean"])
            print(
                f"  {method['label']:<18s} best mae={best['abs_error_mean']:.5f} "
                f"B={best['budget']:.2e}"
            )


if __name__ == "__main__":
    main()
