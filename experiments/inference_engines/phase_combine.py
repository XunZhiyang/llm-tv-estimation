"""Phase: combine X + score_L_pi + score_L_mu into a {pi,mu}_pool.pt
compatible with cache_build's pool format and rerun_estimators.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--X_path", required=True, help="phase_sample output")
    p.add_argument("--score_pi_path", required=True, help="phase_score output for pi side")
    p.add_argument("--score_mu_path", required=True, help="phase_score output for mu side")
    p.add_argument("--side", choices=["pi", "mu"], required=True,
                   help="which sampling side this pool represents")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    src = torch.load(args.X_path, weights_only=False)
    pi_score = torch.load(args.score_pi_path, weights_only=False)
    mu_score = torch.load(args.score_mu_path, weights_only=False)
    X = src["X"]
    sample_lps = src["sample_lps"]
    meta_x = src["meta"]
    K_pi = pi_score["completed_k"]
    K_mu = mu_score["completed_k"]
    K_min = min(K_pi, K_mu)

    pool = {
        "X": X,
        f"sample_L_{args.side}": sample_lps,
        "score_L_pi": pi_score["score_L"][:, :K_min, :].contiguous(),
        "score_L_mu": mu_score["score_L"][:, :K_min, :].contiguous(),
        "row_completed_k": torch.full((meta_x["N"],), K_min, dtype=torch.int16),
        "meta": {
            **meta_x,
            "side": args.side,
            "K_max": K_min,
            "completed_k": K_min,
            "max_completed_k": K_min,
            "engine_pi": pi_score["meta"].get("engine_score", "?"),
            "engine_mu": mu_score["meta"].get("engine_score", "?"),
            "score_path": pi_score["meta"].get("score_path", "prefill"),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(pool, args.out)
    print(f"[combine] wrote {args.out}: side={args.side}, K_min={K_min}, "
          f"engine_pi={pool['meta']['engine_pi']}, engine_mu={pool['meta']['engine_mu']}")


if __name__ == "__main__":
    main()
