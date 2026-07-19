"""Toy autoregressive instance based on the paper's Section 3.2 lower bound.

The sequence is x = u z.  The prefix u selects a block, and each block hides
one parent Z_u deep in the suffix.  Only at that parent can P and Q differ.
This file implements a softened shared-support version so KL remains finite.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Side = str


@dataclass(frozen=True)
class LowerBoundConfig:
    n: int = 128
    r: int = 12
    p: float = 0.05
    alpha: float = 0.40
    shift: str = "symmetric"
    eta: float = 0.01
    delta: float = 0.20
    seed: int = 10
    hard_support: bool = False

    def __post_init__(self) -> None:
        if not (1 <= self.r <= self.n - 2):
            raise ValueError("Need 1 <= r <= n-2.")
        if not (0.0 <= self.p <= 1.0):
            raise ValueError("p must be in [0, 1].")
        if self.hard_support:
            if not (0.0 <= self.alpha <= 0.5):
                raise ValueError("alpha is ignored for hard_support but kept in [0, 0.5].")
        elif not (0.0 <= self.alpha < 0.5):
            raise ValueError("Soft shared-support alpha must be in [0, 0.5).")
        if self.shift not in {"symmetric", "q_rare"}:
            raise ValueError("shift must be 'symmetric' or 'q_rare'.")
        if not (0.0 < self.eta < 1.0):
            raise ValueError("eta must be in (0, 1).")
        if not (0.0 <= self.delta <= 1.0 - self.eta):
            raise ValueError("delta must be in [0, 1-eta].")


class LowerBoundTreeInstance:
    """Frozen binary autoregressive P,Q pair with exact TV.

    For each block u, X_u decides whether the hidden parent is active.
    If inactive, P=Q=(1/2, 1/2) at the final sibling split.
    If active and hard_support=False:
        shift="symmetric": P(B_u)=1/2+alpha, Q(B_u)=1/2-alpha.
        shift="q_rare": Q puts extra mass delta on a P-rare child.
    """

    def __init__(self, cfg: LowerBoundConfig):
        self.cfg = cfg
        self.n = cfg.n
        self.r = cfg.r
        self.t = cfg.n - cfg.r
        self.num_blocks = 1 << cfg.r
        self.rng = np.random.default_rng(cfg.seed)

        self.hidden_parent = self.rng.integers(
            0, 2, size=(self.num_blocks, self.t - 1), dtype=np.int8
        )
        self.orientation = self.rng.integers(0, 2, size=self.num_blocks, dtype=np.int8)
        self.active = (self.rng.random(self.num_blocks) < cfg.p).astype(np.int8)

        self.true_tv = self.local_tv * float(self.active.mean())

    @property
    def local_tv(self) -> float:
        if self.cfg.hard_support:
            return 1.0
        if self.cfg.shift == "q_rare":
            return self.cfg.delta
        return 2.0 * self.cfg.alpha

    def block_bits(self, block_ids: np.ndarray) -> np.ndarray:
        shifts = np.arange(self.r - 1, -1, -1, dtype=np.int64)
        return ((block_ids[:, None] >> shifts[None, :]) & 1).astype(np.int8)

    def bits_to_block(self, bits: np.ndarray) -> np.ndarray:
        bits = np.asarray(bits, dtype=np.int8)
        weights = (1 << np.arange(self.r - 1, -1, -1, dtype=np.int64))
        return (bits[:, : self.r].astype(np.int64) * weights[None, :]).sum(axis=1)

    def final_probs_for_blocks(self, block_ids: np.ndarray, side: Side) -> np.ndarray:
        """Return final-step probabilities with shape (N, 2)."""
        active = self.active[block_ids].astype(bool)
        orient = self.orientation[block_ids].astype(np.int64)
        probs = np.full((len(block_ids), 2), 0.5, dtype=np.float64)

        if self.cfg.hard_support:
            if side == "p":
                probs[active, :] = 0.0
                probs[active, orient[active]] = 1.0
            elif side == "q":
                probs[active, :] = 0.0
                probs[active, 1 - orient[active]] = 1.0
            else:
                raise ValueError("side must be 'p' or 'q'.")
            return probs

        if side not in {"p", "q"}:
            raise ValueError("side must be 'p' or 'q'.")

        if self.cfg.shift == "q_rare":
            rare = orient
            common = 1 - orient
            if side == "p":
                probs[active, rare[active]] = self.cfg.eta
                probs[active, common[active]] = 1.0 - self.cfg.eta
            else:
                probs[active, rare[active]] = self.cfg.eta + self.cfg.delta
                probs[active, common[active]] = 1.0 - self.cfg.eta - self.cfg.delta
            return probs

        hi = 0.5 + self.cfg.alpha
        lo = 0.5 - self.cfg.alpha
        if side == "p":
            probs[active, orient[active]] = hi
            probs[active, 1 - orient[active]] = lo
        else:
            probs[active, orient[active]] = lo
            probs[active, 1 - orient[active]] = hi
        return probs

    def sample(self, side: Side, N: int, rng: np.random.Generator) -> np.ndarray:
        """Sample N full trajectories from P or Q. Returns int8 array (N, n)."""
        block_ids = rng.integers(0, self.num_blocks, size=N, dtype=np.int64)
        prefix = self.block_bits(block_ids)
        parent = self.hidden_parent[block_ids]
        probs = self.final_probs_for_blocks(block_ids, side)
        final = (rng.random(N) >= probs[:, 0]).astype(np.int8)
        return np.concatenate([prefix, parent, final[:, None]], axis=1)

    def log_prob(self, x: np.ndarray, side: Side) -> np.ndarray:
        """Exact sequence log-probability under P or Q."""
        x = np.asarray(x, dtype=np.int8)
        if x.ndim != 2 or x.shape[1] != self.n:
            raise ValueError(f"Expected x shape (N, {self.n}).")

        block_ids = self.bits_to_block(x)
        parent = self.hidden_parent[block_ids]
        parent_ok = (x[:, self.r : self.n - 1] == parent).all(axis=1)
        probs = self.final_probs_for_blocks(block_ids, side)
        final = x[:, -1].astype(np.int64)
        final_prob = probs[np.arange(len(x)), final]

        out = np.full(len(x), -np.inf, dtype=np.float64)
        ok = parent_ok & (final_prob > 0)
        out[ok] = -self.r * math.log(2.0) + np.log(final_prob[ok])
        return out

    def target_probs_along_path(self, x: np.ndarray, side: Side) -> np.ndarray:
        """True conditional probability of each realized next token.

        Shape: (N, n).  The first r bits are uniform block bits, the middle
        t-1 bits follow the hidden parent deterministically, and the final bit
        carries the active-block signal.
        """
        x = np.asarray(x, dtype=np.int8)
        N = x.shape[0]
        target = np.ones((N, self.n), dtype=np.float64)
        target[:, : self.r] = 0.5

        block_ids = self.bits_to_block(x)
        parent_ok = (x[:, self.r : self.n - 1] == self.hidden_parent[block_ids]).all(axis=1)
        probs = self.final_probs_for_blocks(block_ids, side)
        final = x[:, -1].astype(np.int64)
        target[:, -1] = probs[np.arange(N), final]

        if not parent_ok.all():
            target[~parent_ok, self.r : self.n - 1] = 0.0
            target[~parent_ok, -1] = 0.0
        return target

    def next_probs_from_prefix(self, prefix: np.ndarray, side: Side) -> np.ndarray:
        """Exact next-token probabilities for an arbitrary prefix.

        This is useful for spot checks and for making the object behave like a
        prefix oracle.  Off-support prefixes return uniform probabilities.
        """
        prefix = np.asarray(prefix, dtype=np.int8)
        m = len(prefix)
        if m < self.r:
            return np.array([0.5, 0.5], dtype=np.float64)

        block_id = int(self.bits_to_block(prefix[None, : self.r])[0])
        j = m - self.r
        if j < self.t - 1:
            if j > 0 and not np.array_equal(prefix[self.r : m], self.hidden_parent[block_id, :j]):
                return np.array([0.5, 0.5], dtype=np.float64)
            probs = np.zeros(2, dtype=np.float64)
            probs[int(self.hidden_parent[block_id, j])] = 1.0
            return probs

        if j == self.t - 1:
            if not np.array_equal(prefix[self.r : m], self.hidden_parent[block_id]):
                return np.array([0.5, 0.5], dtype=np.float64)
            return self.final_probs_for_blocks(np.array([block_id]), side)[0]

        raise ValueError("Prefix is already a full trajectory.")
