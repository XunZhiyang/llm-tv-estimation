"""Exact-logit convergence sanity check for the synthetic lower-bound instance.

With exact logits there is no sketch noise.  The Algorithm-1 estimator is just
an average of bounded i.i.d. trajectory contributions, so the mean absolute
error should scale as 1/sqrt(N).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from experiments.synthetic.estimators import exact_v2_lr, v2_lr_from_logs
from experiments.synthetic.lower_bound_instance import LowerBoundConfig, LowerBoundTreeInstance


def _git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def exact_single_sample_variance(inst: LowerBoundTreeInstance) -> float:
    """Compute Var(R(X)) exactly for X~P and R=(1-Q(X)/P(X))_+."""
    block_ids = np.arange(inst.num_blocks, dtype=np.int64)
    probs_p = inst.final_probs_for_blocks(block_ids, "p")

    # Build the two possible full trajectories in each block.
    prefix = inst.block_bits(block_ids)
    parent = inst.hidden_parent[block_ids]
    x0 = np.concatenate([prefix, parent, np.zeros((inst.num_blocks, 1), dtype=np.int8)], axis=1)
    x1 = np.concatenate([prefix, parent, np.ones((inst.num_blocks, 1), dtype=np.int8)], axis=1)

    r0 = v2_lr_from_logs(inst.log_prob(x0, "p"), inst.log_prob(x0, "q"))
    r1 = v2_lr_from_logs(inst.log_prob(x1, "p"), inst.log_prob(x1, "q"))
    mean = float(np.mean(probs_p[:, 0] * r0 + probs_p[:, 1] * r1))
    second = float(np.mean(probs_p[:, 0] * r0**2 + probs_p[:, 1] * r1**2))
    return max(0.0, second - mean**2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="results/synthetic_exact_logit_p040")
    p.add_argument("--n", type=int, default=128)
    p.add_argument("--r", type=int, default=12)
    p.add_argument("--p_active", type=float, default=0.4)
    p.add_argument("--alpha", type=float, default=0.49)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--N_grid", default="32,64,128,256,512,1024,2048,4096,8192,16384")
    p.add_argument("--reps", type=int, default=512)
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
    N_grid = parse_ints(args.N_grid)
    rng_master = np.random.default_rng(args.seed + 20260506)
    single_var = exact_single_sample_variance(inst)

    rows: list[dict] = []
    for N in N_grid:
        vals = [
            exact_v2_lr(inst, N, np.random.default_rng(rng_master.integers(2**32)))
            for _ in range(args.reps)
        ]
        arr = np.asarray(vals, dtype=np.float64)
        err = np.abs(arr - inst.true_tv)
        rows.append(
            {
                "N": N,
                "budget_2nN": int(2 * inst.n * N),
                "estimate_mean": float(arr.mean()),
                "estimate_std": float(arr.std(ddof=1)),
                "abs_error_mean": float(err.mean()),
                "abs_error_std": float(err.std(ddof=1)),
                "abs_error_se": float(err.std(ddof=1) / np.sqrt(args.reps)),
                "theory_abs_error_normal": float(np.sqrt(2.0 / np.pi) * np.sqrt(single_var / N)),
            }
        )
        print(
            f"[exact] N={N:6d} mae={rows[-1]['abs_error_mean']:.6f} "
            f"theory={rows[-1]['theory_abs_error_normal']:.6f}",
            flush=True,
        )

    xs = np.log(np.asarray([r["N"] for r in rows], dtype=np.float64))
    ys = np.log(np.asarray([r["abs_error_mean"] for r in rows], dtype=np.float64))
    slope, intercept = np.polyfit(xs, ys, deg=1)

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
            "N_grid": N_grid,
            "reps": args.reps,
            "single_sample_variance": single_var,
            "fit_loglog_slope": float(slope),
            "fit_loglog_intercept": float(intercept),
        },
        "rows": rows,
    }
    with open(out_dir / "exact_logit_convergence.json", "w") as f:
        json.dump(payload, f, indent=2)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x_plot = [1.0 / np.sqrt(r["N"]) for r in rows]
    ax.errorbar(
        x_plot,
        [r["abs_error_mean"] for r in rows],
        yerr=[r["abs_error_se"] for r in rows],
        marker="o",
        linewidth=2.2,
        capsize=3,
        label="Exact-logit estimator",
    )
    ax.plot(
        x_plot,
        [r["theory_abs_error_normal"] for r in rows],
        linestyle="--",
        color="black",
        linewidth=1.8,
        label=r"Normal reference: $\sqrt{2/\pi}\sqrt{\mathrm{Var}(R)/N}$",
    )
    ax.set_xlabel(r"$1/\sqrt{N}$")
    ax.set_ylabel("Mean absolute error")
    ax.set_title(f"Exact-logit convergence, true TV={inst.true_tv:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "exact_logit_convergence.png", dpi=190)
    fig.savefig(out_dir / "exact_logit_convergence.pdf")

    print(f"[exact] fitted log-log slope = {slope:.3f} (theory: -0.5)")
    print(f"[exact] wrote {out_dir / 'exact_logit_convergence.json'}")
    print(f"[exact] wrote {out_dir / 'exact_logit_convergence.png'}")


if __name__ == "__main__":
    main()
