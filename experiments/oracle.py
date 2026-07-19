"""
Oracle abstraction for TV distance estimation experiments.

Single primitive:  next_token_logits(prefix_ids) -> (vocab,) fp32 logits.

Derived helpers (concrete, not overridden):
    next_token_logprobs(prefix)        -> log_softmax(logits)
    complete(prefix, n, rng)           -> (sequence, sample-time logprobs)
    score(sequence, prefix_len)        -> per-token logprobs

Top-k truncation is enforced inside next_token_logits — derived helpers
honor it automatically.

Cache-build helper (NOT a primitive, NOT counted as one call):
    full_sequence_logits(sequence, n_passes)
        Runs n_passes independent forward passes on the full sequence.
        Returns (n_passes, seq_len, vocab) sparse top-k logits.
        Each pass's noise is independent (separate kernel launches).
        Within a single pass, positions share noise (correlated).
        Used to populate Cache A / Cache B efficiently — equivalent to
        n_passes * seq_len calls to next_token_logits at each prefix
        IF we only consume one position per pass.
"""
from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class Oracle(ABC):
    """Single-primitive autoregressive oracle."""

    def __init__(self, top_k: int = 20):
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k

    @property
    @abstractmethod
    def vocab_size(self) -> int: ...

    @property
    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def next_token_logits(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """ONLY primitive. prefix_ids shape (L,) int64. Returns (vocab,) fp32.
        For top-k oracles, entries outside top-k support are -inf.
        Repeated calls on same prefix are i.i.d. realizations."""

    # ── Derived helpers (concrete; subclasses must NOT override) ──────────

    def next_token_logprobs(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.next_token_logits(prefix_ids), dim=-1)

    def complete(
        self,
        prefix_ids: torch.Tensor,
        n_tokens: int,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample n_tokens autoregressively. Returns (full_sequence, sampled_logprobs).
        Exactly n_tokens primitive calls.
        sampled_logprobs[t] is log P(token_t | prefix..token_{t-1}) at sample time.
        """
        seq = prefix_ids.clone()
        lps = []
        for _ in range(n_tokens):
            lp = self.next_token_logprobs(seq)        # (vocab,)
            probs = lp.exp()
            tok = torch.multinomial(probs, 1, generator=rng)   # (1,)
            seq = torch.cat([seq, tok], dim=-1)
            lps.append(lp[tok].squeeze(-1))           # scalar
        return seq, torch.stack(lps)                  # (n_tokens,)

    def score(self, sequence: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """Per-position logprobs of fixed sequence. Exactly len-prefix_len calls.
        Returns (n_tokens,) where n_tokens = len(sequence) - prefix_len.
        """
        n_tokens = sequence.shape[-1] - prefix_len
        if n_tokens < 1:
            raise ValueError("sequence must extend past prefix_len")
        lps = []
        for t in range(n_tokens):
            prefix = sequence[: prefix_len + t]
            lp = self.next_token_logprobs(prefix)     # (vocab,)
            target = sequence[prefix_len + t]
            lps.append(lp[target])
        return torch.stack(lps)                       # (n_tokens,)


# ── HuggingFace concrete implementation ──────────────────────────────────────

class HFOracle(Oracle):
    """HF causal-LM as a single-primitive oracle.

    Implementation notes:
      - next_token_logits(prefix) does ONE full forward pass on the prefix
        (use_cache=False), reads logits at the last position, applies top-k.
      - This is the contract: every call is a fresh kernel launch → SDPA
        non-determinism gives independent noise per call.
      - For efficient cache build, full_sequence_logits(seq, n_passes) batches
        n_passes copies of `seq` through the model in ONE launch only if
        the user requests batched mode; default is n_passes separate launches
        which guarantees IID noise across passes.
    """

    def __init__(
        self,
        model_name: str,
        top_k: int = 20,
        quant_type: str = "bf16",
        attn_impl: str = "sdpa",
        sdpa_backend: str | None = None,
    ):
        super().__init__(top_k=top_k)
        from experiments.tv_estimate import load_model, load_tokenizer
        self.model_name = model_name
        self.quant_type = quant_type
        self.attn_impl = attn_impl
        self.sdpa_backend = sdpa_backend
        self._tokenizer = load_tokenizer(model_name)
        self.model = load_model(model_name, quant_type=quant_type, attn_impl=attn_impl)
        self._device = self.model.get_input_embeddings().weight.device
        self._vocab_size = self.model.config.vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def tokenizer(self):
        return self._tokenizer

    def _ctx(self):
        """Fresh sdpa-kernel context manager (sdpa_kernel is single-use)."""
        if self.sdpa_backend is None or self.attn_impl != "sdpa":
            return contextlib.nullcontext()
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend_map = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "math": SDPBackend.MATH,
            "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
            "memeff": SDPBackend.EFFICIENT_ATTENTION,   # sigma-study alias
            "cudnn": SDPBackend.CUDNN_ATTENTION,
        }
        return sdpa_kernel([backend_map[self.sdpa_backend]])

    def _apply_topk(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k truncation: keep top-k logits, set rest to -inf.
        logits: (..., vocab). Returns same shape.
        """
        if self.top_k >= self._vocab_size:
            return logits
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        masked = torch.full_like(logits, float("-inf"))
        masked.scatter_(-1, top_idx, top_vals)
        return masked

    # ── Batched incremental decoding (B trajectories at once) ────────────
    @torch.inference_mode()
    def score_kv_batched(
        self,
        sequences: torch.Tensor,    # (B, L) int64
        prefix_len: int,
        return_topk_logits: bool = False,
        k_save: int | None = None,
    ):
        """Like score_kv but for B sequences in parallel.
        Each batch element shares the kernel launch but has independent
        token stream → noise consistent with single-traj score_kv.
        Returns (B, n_tokens) lps; optionally (top_k_idx, top_k_val) of shape (B, n_tokens, top_k).

        sigma-study mode (``k_save`` set): records the top-``k_save`` **log-probs**
        (log-softmax over the full vocab — directly comparable to vllm/sglang
        raw_logprobs), not raw logits. ``k_save=None`` preserves the legacy
        behavior exactly (raw logits, width ``top_k``).
        """
        ids = sequences.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        n_tokens = L - prefix_len
        if n_tokens < 1:
            raise ValueError("sequences must extend past prefix_len")
        ksave = k_save if k_save is not None else self.top_k

        topk_vals_steps = []
        topk_idx_steps = []
        with self._ctx():
            out = self.model(ids[:, :prefix_len], use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float()                # (B, vocab)

            for t in range(n_tokens):
                if k_save is not None:                            # sigma-study: store log-probs
                    lp = logits - torch.logsumexp(logits, -1, keepdim=True)
                    top_vals, top_idx = lp.topk(ksave, dim=-1)    # (B, ksave) log-softmax
                else:
                    top_vals, top_idx = logits.topk(self.top_k, dim=-1)   # (B, top_k) raw logits
                topk_vals_steps.append(top_vals)
                topk_idx_steps.append(top_idx)
                if t < n_tokens - 1:
                    next_in = ids[:, prefix_len + t : prefix_len + t + 1]
                    out = self.model(next_in, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    logits = out.logits[:, -1, :].float()

        TV = torch.stack(topk_vals_steps, dim=1)                  # (B, n_tokens, top_k)
        TI = torch.stack(topk_idx_steps,  dim=1)

        if return_topk_logits:
            return None, TI.cpu().int(), TV.cpu()

        targets = ids[:, prefix_len:]                             # (B, n_tokens)
        log_Z = torch.logsumexp(TV, dim=-1)                        # (B, n_tokens)
        match = TI == targets.unsqueeze(-1)                        # (B, n_tokens, top_k)
        any_match = match.any(dim=-1)
        target_vals = torch.where(
            any_match, (TV * match).sum(dim=-1), torch.full_like(log_Z, float("-inf")),
        )
        lps_gpu = torch.where(
            any_match, target_vals - log_Z, torch.full_like(log_Z, float("-inf")),
        )
        return lps_gpu.cpu()

    @torch.inference_mode()
    def score_prefill_batched(
        self,
        sequences: torch.Tensor,    # (B, L) int64
        prefix_len: int,
        k_save: int | None = None,
        return_topk_logits: bool = False,
        chunk: int = 64,
    ):
        """PREFILL-path score: a batched forward over the whole sequence (all
        positions in one attention pass), then per-position top-``k_save``
        log-probs. The one-shot counterpart to the per-position KV-decode
        ``score_kv_batched`` — same noise question, different kernel path.

        Chunked over B: a single B=256/n=500 forward would materialize a
        (B, n, vocab~152k) fp32 tensor (~80 GB) and OOM, so we sub-batch by
        ``chunk``. NOTE: the forward batch is therefore <= chunk, so HF prefill
        is measured at batch <= chunk (a documented cap; decode runs at full B).

        Returns (B, n_tokens) lps; or (None, idx, val) of (B, n_tokens, k_save)
        when ``return_topk_logits``. Stored values are log-softmax (full vocab).
        """
        ids = sequences.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        n_tokens = L - prefix_len
        if n_tokens < 1:
            raise ValueError("sequences must extend past prefix_len")
        ksave = k_save if k_save is not None else self.top_k

        TI_parts, TV_parts, lp_parts = [], [], []
        with self._ctx():
            for c0 in range(0, B, chunk):
                sub = ids[c0 : c0 + chunk]
                out = self.model(sub, use_cache=False)
                logits = out.logits[:, prefix_len - 1 : L - 1, :].float()   # (cb, n, vocab)
                lp = logits - torch.logsumexp(logits, -1, keepdim=True)     # log-softmax
                tv, ti = lp.topk(ksave, dim=-1)                            # (cb, n, ksave)
                ti = ti.cpu().int(); tv = tv.cpu()
                TI_parts.append(ti); TV_parts.append(tv)
                if not return_topk_logits:
                    tgt = sub[:, prefix_len:].cpu()
                    match = ti == tgt.unsqueeze(-1)
                    am = match.any(dim=-1)
                    lp_parts.append(torch.where(
                        am, (tv * match).sum(dim=-1),
                        torch.full(am.shape, float("-inf"))))
                del out, logits, lp
        TI = torch.cat(TI_parts, dim=0)
        TV = torch.cat(TV_parts, dim=0)
        if return_topk_logits:
            return None, TI, TV
        return torch.cat(lp_parts, dim=0)

    @torch.inference_mode()
    def full_logsoftmax_prefill(self, sequences: torch.Tensor, prefix_len: int):
        """Full-vocab log-softmax per position via ONE forward — for the
        full-vocab calibration subset only (certifies top-256 loses no mass).
        Returns (B, n_tokens, vocab) fp16 on CPU."""
        ids = sequences.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        B, L = ids.shape
        with self._ctx():
            out = self.model(ids, use_cache=False)
        logits = out.logits[:, prefix_len - 1 : L - 1, :].float()
        lp = logits - torch.logsumexp(logits, -1, keepdim=True)
        return lp.half().cpu()

    @torch.inference_mode()
    def sample_trajectories_batched(
        self,
        prompt_ids: torch.Tensor,   # (L_prompt,) int64
        n_tokens: int,
        B: int,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample B independent trajectories from the same prompt via one
        batched KV-cache decode. Returns:
            sequences: (B, L_prompt + n_tokens) int64 (CPU)
            sampled_lps: (B, n_tokens) fp32 (CPU) — sample-time logprobs.

        Multinomial sampling stays on GPU (uses a CUDA Generator seeded from
        the supplied CPU rng). One CPU transfer at the end. ~10× faster than
        the per-step .cpu() version.
        """
        if prompt_ids.dim() != 1:
            raise ValueError("prompt_ids must be 1D")
        ids = prompt_ids.to(self._device).unsqueeze(0).expand(B, -1).contiguous()  # (B, L_prompt)

        # Derive a CUDA-side generator from the CPU rng (one int seed; the per-
        # step sampling uses this CUDA generator going forward).
        seed_val = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        cuda_rng = torch.Generator(device=self._device)
        cuda_rng.manual_seed(seed_val)

        sampled_tokens_steps = []
        sampled_lps_steps = []
        with self._ctx():
            out = self.model(ids, use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :].float()                   # (B, vocab)

            for t in range(n_tokens):
                # Top-k truncate FIRST, do softmax/multinomial on (B, top_k).
                # Avoids 4 vocab-sized GPU ops per step (log_softmax + exp + ...).
                top_vals, top_idx = logits.topk(self.top_k, dim=-1)  # (B, top_k)
                log_Z = torch.logsumexp(top_vals, dim=-1, keepdim=True)
                top_probs = (top_vals - log_Z).exp()                 # (B, top_k)
                pos = torch.multinomial(top_probs, 1, generator=cuda_rng)  # (B, 1) in [0, top_k)
                tok = top_idx.gather(1, pos).squeeze(-1)              # (B,) actual token id
                lp_at_tok = (top_vals.gather(1, pos) - log_Z).squeeze(-1)  # (B,)
                sampled_tokens_steps.append(tok)
                sampled_lps_steps.append(lp_at_tok)

                if t < n_tokens - 1:
                    out = self.model(tok.unsqueeze(-1), past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    logits = out.logits[:, -1, :].float()

        sampled_tokens = torch.stack(sampled_tokens_steps, dim=1)   # (B, n) GPU
        sampled_lps = torch.stack(sampled_lps_steps, dim=1)         # (B, n) GPU
        full_seq = torch.cat([ids, sampled_tokens], dim=1).cpu()    # (B, L_prompt + n) CPU
        return full_seq, sampled_lps.cpu()

    @torch.inference_mode()
    def next_token_logits(self, prefix_ids: torch.Tensor) -> torch.Tensor:
        """Single forward pass; return (vocab,) fp32 top-k-masked logits."""
        ids = prefix_ids.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)                    # (1, L)
        with self._ctx():
            out = self.model(ids, use_cache=False)
        logits = out.logits[0, -1, :].float()         # (vocab,)
        return self._apply_topk(logits).cpu()

    # ── Cache-build helper (efficient, NOT a primitive) ──────────────────

    @torch.inference_mode()
    def full_sequence_logits(
        self,
        sequence: torch.Tensor,
        n_passes: int,
    ) -> torch.Tensor:
        """Run n_passes independent forward passes on `sequence`; return logits
        at every position from each pass.

        sequence: (L,) int64 token ids.
        Returns: (n_passes, L, vocab) fp32 top-k-masked logits.

        Each pass is a SEPARATE forward call (separate kernel launch) →
        independent SDPA noise across passes. Within a single pass, positions
        share noise (correlated; but that's fine — multilevel/K-avg algorithms
        only average across PASSES at fixed position, not across positions).

        Cost: n_passes * (cost of one forward at length L).
        """
        ids = sequence.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)                    # (1, L)
        all_logits = []
        with self._ctx():
            for _ in range(n_passes):
                out = self.model(ids, use_cache=False)
                logits = out.logits[0].float()        # (L, vocab)
                all_logits.append(self._apply_topk(logits).cpu())
        return torch.stack(all_logits, dim=0)         # (n_passes, L, vocab)

    # ── Sampling helper using KV-cache (for trajectory generation) ───────

    # ── Incremental score (KV-cache; matches noise of sample_trajectory) ──
    @torch.inference_mode()
    def score_kv(
        self,
        sequence: torch.Tensor,
        prefix_len: int,
        return_topk_logits: bool = False,
    ):
        """Per-position logprob of `sequence` under top-k truncated distribution,
        via incremental KV-cache decode (1 prefill + n_tokens-1 decode steps).

        This path matches the noise structure of sample_trajectory: each
        forward step is its own kernel launch, exhibiting bf16 SDPA
        cross-call non-determinism at sequence length L ≥ ~300.

        sequence: (L,) or (1, L) int64.
        Returns lps: (n_tokens,) fp32 (-inf if token out of top-k).
        Optionally also returns top-k (idx, val) per position for sketch caching.

        Per-step work is kept on GPU; one CPU transfer at the end.
        """
        ids = sequence.to(self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        n_tokens = ids.shape[-1] - prefix_len
        if n_tokens < 1:
            raise ValueError("sequence must extend past prefix_len")

        # Collect (top_k vals, top_k idx) per step on GPU. Avoid per-step .item().
        topk_vals_steps = []      # list of (top_k,) GPU tensors (sparse fp32)
        topk_idx_steps = []       # list of (top_k,) GPU int64 tensors

        with self._ctx():
            out = self.model(ids[:, :prefix_len], use_cache=True)
            past_kv = out.past_key_values
            logits = out.logits[0, -1, :].float()                  # (vocab,)

            for t in range(n_tokens):
                top_vals, top_idx = logits.topk(self.top_k)        # GPU
                topk_vals_steps.append(top_vals)
                topk_idx_steps.append(top_idx)
                if t < n_tokens - 1:
                    next_in = ids[:, prefix_len + t : prefix_len + t + 1]
                    out = self.model(next_in, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
                    logits = out.logits[0, -1, :].float()

        # Stack on GPU.
        TV = torch.stack(topk_vals_steps, dim=0)                   # (n_tokens, top_k) fp32
        TI = torch.stack(topk_idx_steps,  dim=0)                   # (n_tokens, top_k) int64
        targets = ids[0, prefix_len:]                              # (n_tokens,) int64

        # Compute target logprob: target_logit - logsumexp(top-k logits).
        log_Z = torch.logsumexp(TV, dim=-1)                         # (n_tokens,)
        # Find target's position in top-k (or -1 if not present).
        match = (TI == targets.unsqueeze(-1))                      # (n_tokens, top_k) bool
        any_match = match.any(dim=-1)                              # (n_tokens,)
        # Pick target value where matched; placeholder -inf otherwise.
        target_vals = torch.where(
            any_match,
            (TV * match).sum(dim=-1),                              # 0 if not matched, target_logit if matched
            torch.full_like(log_Z, float("-inf")),
        )
        lps_gpu = torch.where(
            any_match, target_vals - log_Z, torch.full_like(log_Z, float("-inf")),
        )
        lps = lps_gpu.cpu()

        if return_topk_logits:
            return lps, TI.cpu().int(), TV.cpu()
        return lps

    @torch.inference_mode()
    def sample_trajectory(
        self,
        prompt_ids: torch.Tensor,
        n_tokens: int,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """KV-cache-accelerated trajectory sampling.

        Equivalent to Oracle.complete(prompt_ids, n_tokens, rng) but uses
        past_key_values for O(1)-per-token decode instead of O(L) full
        re-prefill. The output is statistically equivalent (same noise model)
        because each step is still a single forward call.

        prompt_ids: (L_prompt,) int64.
        Returns:
            full_sequence: (L_prompt + n_tokens,) int64
            sampled_logprobs: (n_tokens,) fp32 — log P at sample time.
        """
        # Convenience wrapper: just call batched version with B=1 then squeeze.
        full_seq, lps = self.sample_trajectories_batched(
            prompt_ids if prompt_ids.dim() == 1 else prompt_ids.squeeze(0),
            n_tokens=n_tokens, B=1, rng=rng,
        )
        return full_seq.squeeze(0), lps.squeeze(0)
