"""SGLangOracle: sglang.Engine as a drop-in oracle compatible with build_pool.

Mirrors VLLMOracle interface. Each sample/score call is one fresh
sglang Engine forward → independent noise across calls.

sglang's Engine supports:
  - .generate(prompt, sampling_params): autoregressive sample with logprobs
  - .generate(prompt, return_logprob=True, top_logprobs_num=K): logprobs at every prompt position
"""
from __future__ import annotations

import torch


def _lazy_import():
    import sglang as sgl
    return sgl


class SGLangOracle:
    def __init__(
        self,
        model_name: str,
        top_k: int = 20,
        dtype: str = "bfloat16",
        mem_fraction_static: float = 0.40,
        max_total_tokens: int = 65536,
        seed: int = 0,
        engine_kwargs: dict | None = None,
    ):
        sgl = _lazy_import()
        self.model_name = model_name
        self.top_k = top_k
        self.dtype = dtype
        self.max_total_tokens = int(max_total_tokens)
        kwargs = dict(
            model_path=model_name,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            max_total_tokens=max_total_tokens,
            random_seed=seed,
            disable_overlap_schedule=False,
            disable_cuda_graph=True,  # disable CUDA graphs so each forward draws fresh noise
        )
        if engine_kwargs:
            kwargs.update(engine_kwargs)
        self.engine = sgl.Engine(**kwargs)
        # sglang exposes tokenizer via the engine; fallback to HF if needed.
        try:
            self.tokenizer = self.engine.tokenizer_manager.tokenizer
        except Exception:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._base_seed = int(seed)

    @property
    def vocab_size(self) -> int:
        return getattr(self.tokenizer, "vocab_size", None) or len(self.tokenizer)

    @property
    def device(self):
        return torch.device("cuda")

    def _flush_cache(self):
        """Drop the radix cache (independence between K-repeats)."""
        fn = getattr(self.engine, "flush_cache", None)
        if fn is None:
            raise RuntimeError("this sglang version exposes no Engine.flush_cache()")
        if fn() is False:  # returns False instead of raising when it cannot flush
            raise RuntimeError("flush_cache() failed — K-repeats would share KV")

    @torch.inference_mode()
    def sample_trajectories_batched(
        self, prompt_ids: torch.Tensor, n_tokens: int, B: int, rng: torch.Generator,
        return_topk_logits: bool = False, processor_parity: bool = False,
    ):
        prompt_list = prompt_ids.tolist()
        seed_off = int(torch.randint(0, 2**31 - 1, (1,), generator=rng).item())
        sampling_params_list = [
            {
                "max_new_tokens": n_tokens,
                "top_k": self.top_k,
                "temperature": 1.0,
                "n": 1,
                "ignore_eos": True,
            }
            for _ in range(B)
        ]
        gen_kwargs = {}
        if processor_parity:
            # attach the SAME custom processor as replay scoring with no
            # force_tokens (a no-op): the sampler pipeline then matches the
            # replay pipeline exactly. Requires enable_custom_logit_processor.
            from experiments.inference_engines.force_tokens_sglang import (
                CaptureForceProcessor)
            gen_kwargs["custom_logit_processor"] = CaptureForceProcessor().to_str()
            for p in sampling_params_list:
                p["custom_params"] = {}
        # sglang Engine takes input_ids (list of token-id lists) for batched gen.
        outs = self.engine.generate(
            input_ids=[prompt_list] * B,
            sampling_params=sampling_params_list,
            return_logprob=True,
            top_logprobs_num=self.top_k,
            **gen_kwargs,
        )
        # outs is list of dict per request:
        #   meta_info.output_token_logprobs: list of (logprob, token_id, decoded) per gen step
        #   meta_info.output_top_logprobs: list of list of (logprob, token_id, decoded) at each gen step
        full = torch.zeros((B, len(prompt_list) + n_tokens), dtype=torch.int64)
        sample_lps = torch.full((B, n_tokens), float("-inf"), dtype=torch.float32)
        if return_topk_logits:
            topk_idx = torch.full((B, n_tokens, self.top_k), -1, dtype=torch.int32)
            topk_val = torch.full((B, n_tokens, self.top_k), float("-inf"), dtype=torch.float32)
        for i, out in enumerate(outs):
            meta = out["meta_info"] if isinstance(out, dict) else out.meta_info
            full[i, :len(prompt_list)] = torch.tensor(prompt_list, dtype=torch.int64)
            tok_lps = meta["output_token_logprobs"]
            for t, (lp, tok_id, _) in enumerate(tok_lps[:n_tokens]):
                full[i, len(prompt_list) + t] = int(tok_id)
                sample_lps[i, t] = float(lp)
            if return_topk_logits:
                top_lps = meta.get("output_top_logprobs", [])
                for t, step_lps in enumerate(top_lps[:n_tokens]):
                    for k, (tlp, ttid, _) in enumerate(step_lps[:self.top_k]):
                        if tlp is None:
                            continue
                        topk_idx[i, t, k] = int(ttid)
                        topk_val[i, t, k] = float(tlp)
        if return_topk_logits:
            return full, sample_lps, topk_idx, topk_val
        return full, sample_lps

    @torch.inference_mode()
    def score_kv_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        flush_each_step: bool = False,
    ):
        """PREFILL-path score (the May-2026 production semantics): for each
        step t in 0..n, submit B prefixes of length prefix_len+t and ask for
        next-token logprobs (max_new_tokens=1, return_logprob=True).

        Note: the radix cache is ON by default, but in production
        max_total_tokens (8192-65536) was far below one step's batch footprint
        (B * seq_len up to 265k tokens), so the cache thrashed and every step
        effectively re-prefilled — pi_prefill semantics. Pass
        flush_each_step=True to make that explicit / robust to larger caches.
        """
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
            seq_len = prefix_len + t
            input_ids = [sequences[i, :seq_len].tolist() for i in range(B)]
            sampling_params_list = [
                {"max_new_tokens": 1, "temperature": 1.0, "top_k": self.top_k}
                for _ in range(B)
            ]
            outs = self.engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params_list,
                return_logprob=True,
                top_logprobs_num=self.top_k,
            )
            for i, out in enumerate(outs):
                meta = out["meta_info"] if isinstance(out, dict) else out.meta_info
                # output_token_logprobs[0] is the next-token logprob (the generated 1 token,
                # which we override): we need top_logprobs at this step.
                top_lps = meta.get("output_top_logprobs", [])
                if not top_lps:
                    continue
                step_lps = top_lps[0]            # list of (lp, token_id, decoded)
                target = int(sequences[i, seq_len].item())
                lp_map = {int(tid): float(lp) for lp, tid, _ in step_lps if lp is not None}
                if target in lp_map:
                    lps[i, t] = lp_map[target]
                if return_topk_logits:
                    for k, (tlp, ttid, _) in enumerate(step_lps[:self.top_k]):
                        if tlp is None:
                            continue
                        topk_idx[i, t, k] = int(ttid)
                        topk_val[i, t, k] = float(tlp)

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_replay_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        capture_dir: str = "/tmp", k_save: int | None = None,
    ):
        """Teacher-forced replay scoring — the generation-path oracle.

        The engine truly GENERATES (decode path, same scheduler behavior as
        sampling) while CaptureForceProcessor pins every step to the given
        sequence AND records the raw pre-forcing top-k log-softmax (sglang
        computes its returned logprobs after processors run, so those are
        polluted; the capture sees the very logits the sampler used).
        Validated 2026-06-12: forcing exact, own-zero = 0, repeats
        bit-identical, capture within ~4e-5 of the sample-time readout.

        Values are renormalized over the captured top-k set (= the actual
        top-k sampling distribution). Requires the engine constructed with
        enable_custom_logit_processor=True. One K-repeat per call.
        Returns (B, n_tokens) fp32 lps on CPU.
        """
        import json, math, os, uuid
        from experiments.inference_engines.force_tokens_sglang import CaptureForceProcessor
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        width = k_save if k_save is not None else self.top_k + 8
        if return_topk_logits:
            topk_idx = torch.full((B, n, width), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, width), float("-inf"), dtype=torch.float32)

        tag = uuid.uuid4().hex[:10]
        paths = [f"{capture_dir}/sgcap_{tag}_{i}.json" for i in range(B)]
        # capture beyond top_k: the engines' top-k filters are THRESHOLD
        # semantics (all tokens tied with the k-th survive), so the true
        # sampling support can exceed top_k at bf16 ties. Extraction below
        # reconstructs the tie-inclusive support.
        k_cap = k_save if k_save is not None else self.top_k + 8
        sp = [{"max_new_tokens": n, "temperature": 1.0, "top_k": self.top_k,
               "ignore_eos": True,
               "custom_params": {"force_tokens": sequences[i, prefix_len:].tolist(),
                                 "capture_path": paths[i],
                                 "capture_topk": k_cap}}
              for i in range(B)]
        outs = self.engine.generate(
            input_ids=[sequences[i, :prefix_len].tolist() for i in range(B)],
            sampling_params=sp,
            custom_logit_processor=CaptureForceProcessor().to_str(),
            # same logprob request as sampling => identical sampler pipeline
            # (values are polluted by forcing; capture is the source of truth)
            return_logprob=True, top_logprobs_num=self.top_k,
        )
        bad = []
        for i, out in enumerate(outs):
            meta = out["meta_info"] if isinstance(out, dict) else out.meta_info
            toks = [int(t) for _, t, *_ in meta["output_token_logprobs"]][:n]
            tgt = sequences[i, prefix_len:].tolist()
            if toks != tgt:
                first = next((j for j, (a, b) in enumerate(zip(toks, tgt)) if a != b),
                             min(len(toks), len(tgt)))
                bad.append((i, len(toks), first))
        if bad:
            i0, lt, fm = bad[0]
            raise RuntimeError(
                f"teacher forcing failed for {len(bad)}/{B} requests; first: req "
                f"{i0} got {lt}/{n} tokens, first mismatch at step {fm}")
        try:
            for i, pth in enumerate(paths):
                steps = json.load(open(pth))
                if len(steps) < n:
                    raise RuntimeError(f"capture for request {i} has {len(steps)} < {n} steps")
                for t, (idx, vals) in enumerate(steps[:n]):
                    target = int(sequences[i, prefix_len + t].item())
                    if k_save is not None:
                        # sigma-study: store RAW top-k_save log-softmax (capture records
                        # raw pre-forcing log-softmax); reconstruct support offline.
                        order = sorted(range(len(vals)), key=lambda j: -vals[j])[:k_save]
                        for k, j in enumerate(order):
                            tid = int(idx[j])
                            if tid == target:
                                lps[i, t] = float(vals[j])
                            if return_topk_logits:
                                topk_idx[i, t, k] = tid
                                topk_val[i, t, k] = float(vals[j])
                        continue
                    # tie-inclusive support: everything with value >= the
                    # top_k-th value (exact fp equality at bf16 ties)
                    kth = sorted(vals, reverse=True)[self.top_k - 1]
                    sup = [(v, int(tid)) for v, tid in zip(vals, idx) if v >= kth]
                    m = max(v for v, _ in sup)
                    log_z = m + math.log(sum(math.exp(v - m) for v, _ in sup))
                    for k, (v, tid) in enumerate(sup):
                        if tid == target:
                            lps[i, t] = v - log_z
                        if return_topk_logits and k < topk_idx.shape[-1]:
                            topk_idx[i, t, k] = tid
                            topk_val[i, t, k] = v - log_z
        finally:
            for pth in paths:
                try:
                    os.remove(pth)
                except OSError:
                    pass

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_seq_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        k_save: int | None = None,
    ):
        """Whole-sequence single-call scoring via input logprobs — the
        engine's standard "score this text" API (return_logprob with
        logprob_start_len). All positions of a repeat come from ONE forward,
        so per-call noise does not accumulate along positions.

        Alignment is verified against the input token ids (sglang's
        input_token_logprobs indexing is version-dependent). Top-k entries
        are renormalized client-side over the top_k set (raw log-softmax).

        Returns (B, n_tokens) fp32 lps on CPU.
        """
        import math
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        width = k_save if k_save is not None else self.top_k
        if return_topk_logits:
            topk_idx = torch.full((B, n, width), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, width), float("-inf"), dtype=torch.float32)

        input_ids = [sequences[i].tolist() for i in range(B)]
        sampling_params_list = [
            {"max_new_tokens": 1, "temperature": 1.0, "top_k": self.top_k}
            for _ in range(B)
        ]
        outs = self.engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params_list,
            return_logprob=True,
            logprob_start_len=max(prefix_len - 1, 0),
            top_logprobs_num=width,
        )
        for i, out in enumerate(outs):
            meta = out["meta_info"] if isinstance(out, dict) else out.meta_info
            in_lps = meta.get("input_token_logprobs") or []
            in_top = meta.get("input_top_logprobs") or []
            # Align: entry j describes input token at position L - len(in_lps) + j.
            offset = L - len(in_lps)
            mismatch = 0
            for j, entry in enumerate(in_lps):
                tid = int(entry[1])
                pos = offset + j
                if 0 <= pos < L and tid != int(sequences[i, pos].item()):
                    mismatch += 1
            if mismatch:
                raise RuntimeError(
                    f"input_token_logprobs misaligned for request {i}: "
                    f"{mismatch}/{len(in_lps)} token-id mismatches (offset={offset})")
            for j, step_lps in enumerate(in_top):
                pos = offset + j
                t = pos - prefix_len
                if t < 0 or t >= n or not step_lps:
                    continue
                target = int(sequences[i, pos].item())
                if k_save is not None:
                    # sigma-study: store RAW top-k_save log-softmax (input_top_logprobs
                    # are raw log-softmax); reconstruct support offline.
                    entries = [(float(lp), int(tid)) for lp, tid, *_ in step_lps
                               if lp is not None][:k_save]
                    for k, (lp, tid) in enumerate(entries):
                        if tid == target:
                            lps[i, t] = lp
                        if return_topk_logits:
                            topk_idx[i, t, k] = tid
                            topk_val[i, t, k] = lp
                    continue
                entries = [(float(lp), int(tid)) for lp, tid, *_ in step_lps
                           if lp is not None][:self.top_k]
                if not entries:
                    continue
                m = max(lp for lp, _ in entries)
                log_z = m + math.log(sum(math.exp(lp - m) for lp, _ in entries))
                for k, (lp, tid) in enumerate(entries):
                    if tid == target:
                        lps[i, t] = lp - log_z
                    if return_topk_logits:
                        topk_idx[i, t, k] = tid
                        topk_val[i, t, k] = lp - log_z

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps

    @torch.inference_mode()
    def score_decode_batched(
        self, sequences: torch.Tensor, prefix_len: int, return_topk_logits: bool = False,
        chunk: int = 64,
    ):
        """DECODE-path score (one K-repeat per call). Same per-step loop as
        score_kv_batched, but sized so the radix cache actually holds the
        chunk: step t's prefix extends step t-1's by exactly one token
        (sglang's radix cache is token-granular), so the engine computes a
        1-token extend instead of re-prefilling — the closest public-API
        approximation to the decode path generation uses. The cache is
        flushed at call start and between chunks, so separate calls are
        independent K-repeats.

        Cache footprint per chunk is chunk*(L + n) tokens: the trunk
        chunk*L PLUS chunk*n dead one-token generation leaves (each
        max_new_tokens=1 call parks its throwaway token in the radix tree;
        LRU should evict leaves before the constantly re-touched trunk, but
        we require headroom rather than rely on it). Guarded below; with
        chunk=64 and 530-token sequences that is ~66k tokens → construct the
        oracle with max_total_tokens >= 131072. Validation check: the
        own-zero rate on self-sampled trajectories must drop to ~0.
        """
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        B, L = sequences.shape
        n = L - prefix_len
        chunk = min(chunk, B)
        footprint = chunk * (L + n)
        if footprint > 0.8 * self.max_total_tokens:
            raise RuntimeError(
                f"decode-path chunk footprint {footprint} tokens (trunk + dead "
                f"generation leaves) exceeds 80% of max_total_tokens="
                f"{self.max_total_tokens}; lower chunk or raise max_total_tokens")
        lps = torch.full((B, n), float("-inf"), dtype=torch.float32)
        if return_topk_logits:
            topk_idx = torch.full((B, n, self.top_k), -1, dtype=torch.int32)
            topk_val = torch.full((B, n, self.top_k), float("-inf"), dtype=torch.float32)

        for c0 in range(0, B, chunk):
            self._flush_cache()
            rows = list(range(c0, min(c0 + chunk, B)))
            for t in range(n):
                seq_len = prefix_len + t
                input_ids = [sequences[i, :seq_len].tolist() for i in rows]
                sampling_params_list = [
                    {"max_new_tokens": 1, "temperature": 1.0, "top_k": self.top_k}
                    for _ in rows
                ]
                outs = self.engine.generate(
                    input_ids=input_ids,
                    sampling_params=sampling_params_list,
                    return_logprob=True,
                    top_logprobs_num=self.top_k,
                )
                for i, out in zip(rows, outs):
                    meta = out["meta_info"] if isinstance(out, dict) else out.meta_info
                    top_lps = meta.get("output_top_logprobs", [])
                    if not top_lps:
                        continue
                    step_lps = top_lps[0]
                    target = int(sequences[i, seq_len].item())
                    lp_map = {int(tid): float(lp) for lp, tid, _ in step_lps if lp is not None}
                    if target in lp_map:
                        lps[i, t] = lp_map[target]
                    if return_topk_logits:
                        for k, (tlp, ttid, _) in enumerate(step_lps[:self.top_k]):
                            if tlp is None:
                                continue
                            topk_idx[i, t, k] = int(ttid)
                            topk_val[i, t, k] = float(tlp)

        if return_topk_logits:
            return None, topk_idx, topk_val
        return lps
