"""VLLMOracle: vllm.LLM as a drop-in oracle compatible with cache_build.build_pool.

Each sample_trajectories_batched / score_kv_batched call wraps one
vllm.LLM.generate() call → one fresh forward pass → independent noise across
calls (suitable for K-averaging).

Two scoring paths:
  - score_kv_batched: per-step generate(max_tokens=1) on the growing prefix.
    With enable_prefix_caching=False (recorded production config) every step
    re-prefills the whole prefix → estimates the PREFILL-path distribution
    pi_prefill, which differs from the sample-time decode-path distribution
    pi_decode at ~6e-4 of cells (top-k boundary flips) — the source of the
    own-zero artifact in the May 2026 pools.
  - score_decode_batched: same per-step loop but relies on prefix caching to
    reuse the KV built by earlier steps of the same call, approximating the
    decode path generation uses. Requires enable_prefix_caching=True;
    K-repeat independence is restored by flushing the cache between calls.

Sample uses vllm's native autoregressive generation (decode path),
with logprobs at each generated step.
"""
from __future__ import annotations

import math

import torch


def _lazy_import():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    return LLM, SamplingParams, TokensPrompt


class VLLMOracle:
    def __init__(
        self,
        model_name: str,
        top_k: int = 20,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.40,
        max_model_len: int = 2048,
        seed: int = 0,
        enable_prefix_caching: bool = False,
        llm_kwargs: dict | None = None,
    ):
        LLM, _, _ = _lazy_import()
        self.model_name = model_name
        self.top_k = top_k
        self.dtype = dtype
        self.enable_prefix_caching = bool(enable_prefix_caching)
        kwargs = dict(
            model=model_name,
            dtype=dtype,
            # False = recorded May-2026 production config (prefill-path scoring).
            # True is required for score_decode_batched, which handles K-repeat
            # independence itself via _flush_cache().
            enable_prefix_caching=self.enable_prefix_caching,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
            enforce_eager=True,  # disable CUDA graphs so each forward draws fresh noise
        )
        if llm_kwargs:
            kwargs.update(llm_kwargs)
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self._base_seed = int(seed)

    @property
    def vocab_size(self) -> int:
        return getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)

    @property
    def device(self):
        return torch.device("cuda")

    def _flush_cache(self):
        """Drop all cached KV blocks (independence between K-repeats)."""
        fn = getattr(self.llm, "reset_prefix_cache", None) or getattr(
            getattr(self.llm, "llm_engine", None), "reset_prefix_cache", None)
        if fn is None:
            raise RuntimeError("this vllm version exposes no reset_prefix_cache()")
        if fn() is False:  # returns False instead of raising when it cannot flush
            raise RuntimeError("reset_prefix_cache() failed — K-repeats would share KV")

    @torch.inference_mode()
    def sample_trajectories_batched(
        self, prompt_ids: torch.Tensor, n_tokens: int, B: int, rng: torch.Generator,
        return_topk_logits: bool = False,
    ):
        """Returns (full_seq: (B, prompt_len+n) int64 CPU,
                     sample_lps: (B, n) fp32 CPU — log p_vllm at sample time).
        NOTE sample_lps are vllm's RAW readout (logprobs_mode=raw_logprobs:
        full-vocab log-softmax computed BEFORE the top-k filter) — they are NOT
        renormalized over the top-k support the token was actually drawn from.
        Comparable to score_replay_batched's raw readout, not to its returned
        (renormalized) lps.
        With return_topk_logits also returns (topk_idx, topk_val) of shape
        (B, n, top_k): the sample-time (decode-path) top-k distribution —
        the exact reference for validating decode-path scoring."""
        _, SamplingParams, TokensPrompt = _lazy_import()
        prompt_list = prompt_ids.tolist()
        prompts = [TokensPrompt(prompt_token_ids=prompt_list) for _ in range(B)]
        seed_off = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        sps = [SamplingParams(
            max_tokens=n_tokens, top_k=self.top_k, temperature=1.0,
            logprobs=self.top_k, seed=seed_off + i, ignore_eos=True,
        ) for i in range(B)]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        full = torch.zeros((B, len(prompt_list) + n_tokens), dtype=torch.int64)
        sample_lps = torch.full((B, n_tokens), float("-inf"), dtype=torch.float32)
        if return_topk_logits:
            topk_idx = torch.full((B, n_tokens, self.top_k), -1, dtype=torch.int32)
            topk_val = torch.full((B, n_tokens, self.top_k), float("-inf"), dtype=torch.float32)
        for i, out in enumerate(outs):
            gen = out.outputs[0]
            toks = gen.token_ids
            full[i, :len(prompt_list)] = torch.tensor(prompt_list, dtype=torch.int64)
            full[i, len(prompt_list):] = torch.tensor(toks, dtype=torch.int64)
            for t, tok_id in enumerate(toks):
                lp_dict = gen.logprobs[t]
                if tok_id in lp_dict:
                    sample_lps[i, t] = float(lp_dict[tok_id].logprob)
                if return_topk_logits:
                    items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)[:self.top_k]
                    for k, (tid, lp) in enumerate(items):
                        topk_idx[i, t, k] = int(tid)
                        topk_val[i, t, k] = float(lp.logprob)
        if return_topk_logits:
            return full, sample_lps, topk_idx, topk_val
        return full, sample_lps

    @torch.inference_mode()
    def score_kv_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        flush_each_step: bool = False,
    ):
        """PREFILL-path score (the May-2026 production semantics): for each
        step t in 0..n, submit a batched generate(max_tokens=1) with prefix
        length prefix_len+t. With enable_prefix_caching=False every step
        re-prefills the full prefix. If the oracle was built with caching ON,
        pass flush_each_step=True to preserve the prefill-path semantics.

        Returns (B, n_tokens) fp32 lps on CPU.
        """
        _, SamplingParams, TokensPrompt = _lazy_import()
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        if return_topk_logits:
            topk_idx = torch.full((B, n, self.top_k), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, self.top_k), float("-inf"), dtype=torch.float32)

        for t in range(n):
            if flush_each_step:
                self._flush_cache()
            # Build B prefixes of length prefix_len + t.
            seq_len = prefix_len + t
            prompts = [TokensPrompt(prompt_token_ids=sequences[i, :seq_len].tolist())
                       for i in range(B)]
            sps = [SamplingParams(
                max_tokens=1, top_k=self.top_k, temperature=1.0,
                logprobs=self.top_k,
            ) for _ in range(B)]
            outs = self.llm.generate(prompts, sps, use_tqdm=False)
            for i, out in enumerate(outs):
                lp_dict = out.outputs[0].logprobs[0]
                target = int(sequences[i, seq_len].item())
                if target in lp_dict:
                    lps[i, t] = float(lp_dict[target].logprob)
                if return_topk_logits:
                    items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)[:self.top_k]
                    for k, (tok_id, lp) in enumerate(items):
                        topk_idx[i, t, k] = int(tok_id)
                        topk_val[i, t, k] = float(lp.logprob)

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_replay_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        k_save: int | None = None,
    ):
        """Teacher-forced replay scoring — the unbiased generation-path oracle.

        The engine truly GENERATES (decode kernel, same code path and batch
        shape as sampling) while ForceTokensLogitsProcessor pins every step to
        the given sequence. vllm computes returned logprobs from the raw
        logits BEFORE custom processors run (logprobs_mode=raw_logprobs), so
        they are unpolluted — validated 2026-06-12: the RAW replay readout is
        bit-identical to the raw sample-time readout at the same batch shape,
        with zero own-zero cells, at batch sizes where incremental scoring
        fails. The default (k_save=None) return value renormalizes that
        readout over the tie-inclusive top-k support, so it is NOT bit-equal
        to sample_lps (which stay raw).

        Requires the oracle constructed with
            llm_kwargs={"logits_processors":
                ["experiments.inference_engines.force_tokens_vllm:"
                 "ForceTokensLogitsProcessor"]}

        One K-repeat per call. Returns (B, n_tokens) fp32 lps on CPU.
        """
        _, SamplingParams, TokensPrompt = _lazy_import()
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        width = k_save if k_save is not None else self.top_k + 8
        if return_topk_logits:
            topk_idx = torch.full((B, n, width), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, width), float("-inf"), dtype=torch.float32)

        prompts = [TokensPrompt(prompt_token_ids=sequences[i, :prefix_len].tolist())
                   for i in range(B)]
        # report beyond top_k: the sampler top-k filter is threshold semantics
        # (bf16 ties with the k-th value survive), so the true support can
        # exceed top_k. Requires max_logprobs >= top_k+8 at engine build.
        k_rep = k_save if k_save is not None else self.top_k + 8
        sps = [SamplingParams(
            max_tokens=n, temperature=1.0, top_k=self.top_k,
            logprobs=k_rep, ignore_eos=True,
            extra_args={"force_tokens": sequences[i, prefix_len:].tolist()},
        ) for i in range(B)]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        for i, out in enumerate(outs):
            gen = out.outputs[0]
            if list(gen.token_ids)[:n] != sequences[i, prefix_len:].tolist():
                raise RuntimeError(f"teacher forcing failed for request {i}")
            for t in range(n):
                lp_dict = gen.logprobs[t]
                target = int(sequences[i, prefix_len + t].item())
                items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)
                if k_save is not None:
                    # sigma-study: store RAW top-k_save log-probs (raw_logprobs =
                    # log-softmax over full vocab); no truncation/renorm here — all
                    # support reconstruction happens offline at analysis time.
                    for k, (tid, lp) in enumerate(items[:k_save]):
                        if int(tid) == target:
                            lps[i, t] = float(lp.logprob)
                        if return_topk_logits:
                            topk_idx[i, t, k] = int(tid)
                            topk_val[i, t, k] = float(lp.logprob)
                    continue
                kth = items[min(self.top_k, len(items)) - 1][1].logprob
                sup = [(tid, lp.logprob) for tid, lp in items if lp.logprob >= kth]
                m = max(lp for _, lp in sup)
                log_z = m + math.log(sum(math.exp(lp - m) for _, lp in sup))
                for k, (tid, lp) in enumerate(sup):
                    if int(tid) == target:
                        lps[i, t] = float(lp - log_z)
                    if return_topk_logits and k < topk_idx.shape[-1]:
                        topk_idx[i, t, k] = int(tid)
                        topk_val[i, t, k] = float(lp - log_z)
        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_seq_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        k_save: int | None = None,
    ):
        """Whole-sequence single-call scoring via prompt_logprobs — the
        engine's standard "score this text" API. All positions of a repeat
        come from ONE forward, so per-call noise does not accumulate along
        positions (unlike generation, where each step's noise enters the KV
        cache and propagates) — measured property, see calibration.

        Top-k handling: prompt_logprobs always includes the actual token even
        when its rank > top_k; we enforce the top-k oracle model by keeping
        the top_k best entries only and renormalizing them client-side
        (prompt logprobs are raw log-softmax, no sampler renormalization).

        Returns (B, n_tokens) fp32 lps on CPU.
        """
        _, SamplingParams, TokensPrompt = _lazy_import()
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        width = k_save if k_save is not None else self.top_k
        if return_topk_logits:
            topk_idx = torch.full((B, n, width), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, width), float("-inf"), dtype=torch.float32)

        prompts = [TokensPrompt(prompt_token_ids=sequences[i].tolist()) for i in range(B)]
        sps = [SamplingParams(
            max_tokens=1, temperature=1.0, top_k=self.top_k,
            prompt_logprobs=width, detokenize=False,
        ) for _ in range(B)]
        outs = self.llm.generate(prompts, sps, use_tqdm=False)
        for i, out in enumerate(outs):
            pls = out.prompt_logprobs  # entry j: dist of token j given < j; [0] is None
            for t in range(n):
                lp_dict = pls[prefix_len + t]
                if lp_dict is None:
                    continue
                if k_save is not None:
                    # sigma-study: store RAW top-k_save log-probs (prompt_logprobs
                    # are raw log-softmax); no client-side renorm.
                    items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)[:k_save]
                    target = int(sequences[i, prefix_len + t].item())
                    for k, (tok_id, lp) in enumerate(items):
                        if int(tok_id) == target:
                            lps[i, t] = float(lp.logprob)
                        if return_topk_logits:
                            topk_idx[i, t, k] = int(tok_id)
                            topk_val[i, t, k] = float(lp.logprob)
                    continue
                items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)[:self.top_k]
                log_z = items[0][1].logprob + math.log(sum(
                    math.exp(lp.logprob - items[0][1].logprob) for _, lp in items))
                target = int(sequences[i, prefix_len + t].item())
                for k, (tok_id, lp) in enumerate(items):
                    if int(tok_id) == target:
                        lps[i, t] = float(lp.logprob - log_z)
                    if return_topk_logits:
                        topk_idx[i, t, k] = int(tok_id)
                        topk_val[i, t, k] = float(lp.logprob - log_z)

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_decode_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        chunk: int = 128,
    ):
        """DECODE-path score (one K-repeat per call). Same per-step loop as
        score_kv_batched, but with prefix caching ON: step t reuses the KV
        blocks built by steps < t of THIS call, so the engine processes the
        prefix the way generation does instead of re-prefilling it. The cache
        is flushed at call start and between chunks, so separate calls are
        independent K-repeats.

        Caveats:
          - vllm prefix caching is block-granular (16 tokens): each step
            recomputes the tail of the last partial block with the extend
            kernel, so this approximates (not bit-exactly reproduces) the
            decode path. Validation check: the own-zero rate on self-sampled
            trajectories must drop from ~6e-4/cell (prefill path) to ~0.
          - chunk * (prefix_len + n) tokens of KV must fit in the paged cache
            with headroom, otherwise eviction silently degrades steps back to
            full prefill. 128 seqs * 530 tok * ~115KB/tok (Qwen3-0.6B) ≈ 8 GB.

        Returns (B, n_tokens) fp32 lps on CPU.
        """
        if not self.enable_prefix_caching:
            raise RuntimeError(
                "score_decode_batched requires the oracle to be constructed with "
                "enable_prefix_caching=True")
        _, SamplingParams, TokensPrompt = _lazy_import()
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        if return_topk_logits:
            topk_idx = torch.full((B, n, self.top_k), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, self.top_k), float("-inf"), dtype=torch.float32)

        for c0 in range(0, B, chunk):
            self._flush_cache()
            rows = range(c0, min(c0 + chunk, B))
            for t in range(n):
                seq_len = prefix_len + t
                prompts = [TokensPrompt(prompt_token_ids=sequences[i, :seq_len].tolist())
                           for i in rows]
                sps = [SamplingParams(
                    max_tokens=1, top_k=self.top_k, temperature=1.0,
                    logprobs=self.top_k,
                ) for _ in rows]
                outs = self.llm.generate(prompts, sps, use_tqdm=False)
                for i, out in zip(rows, outs):
                    lp_dict = out.outputs[0].logprobs[0]
                    target = int(sequences[i, seq_len].item())
                    if target in lp_dict:
                        lps[i, t] = float(lp_dict[target].logprob)
                    if return_topk_logits:
                        items = sorted(lp_dict.items(), key=lambda x: -x[1].logprob)[:self.top_k]
                        for k, (tok_id, lp) in enumerate(items):
                            topk_idx[i, t, k] = int(tok_id)
                            topk_val[i, t, k] = float(lp.logprob)

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps
