"""Teacher-forcing logit processor for sglang (>=0.5, custom_logit_processor API).

Forces the generation to follow a given token list while the engine runs its
real decode path — the returned per-step logprobs then constitute a replay
of the generation-path distribution under teacher forcing.

Usage:
    engine = sgl.Engine(..., enable_custom_logit_processor=True)
    engine.generate(
        input_ids=[prompt_ids],
        sampling_params=[{"max_new_tokens": n, "temperature": 1.0, "top_k": 20,
                          "custom_params": {"force_tokens": target_token_list}}],
        custom_logit_processor=ForceTokensProcessor().to_str(),
        return_logprob=True, top_logprobs_num=20,
    )
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


class ForceTokensProcessor(CustomLogitProcessor):
    """Mask logits so the next sampled token is force_tokens[step]."""

    def __call__(
        self,
        logits,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ):
        if not custom_param_list:
            return logits
        for i, pd in enumerate(custom_param_list):
            if pd is None:
                continue
            targets = pd.get("force_tokens")
            if not targets:
                continue
            # Requires disable_overlap_schedule=True: with overlap on,
            # req.output_ids lags and invocation counters can desync at high
            # concurrency. output_ids-based indexing is also robust to
            # request retraction/recompute.
            req = pd.get("__req__")
            step = len(req.output_ids)
            if step < len(targets):
                logits[i, :] = -float("inf")
                logits[i, int(targets[step])] = 0.0
        return logits


class CaptureForceProcessor(CustomLogitProcessor):
    """Teacher-force AND capture the RAW pre-forcing top-k logprobs.

    The engine computes returned logprobs after logits processors run, so the
    standard logprob fields are polluted by the forcing mask. This processor
    therefore records log_softmax(raw logits) top-k itself, accumulating per
    request and dumping to `custom_params["capture_path"]` (a node-local file,
    e.g. /tmp/...) as JSON [[idx...], [lp...]] per step when the last forced
    step is reached. Runs inside the scheduler process — the caller on the
    same node reads the files after generate() returns.
    """

    def __call__(
        self,
        logits,
        custom_param_list: Optional[List[Dict[str, Any]]] = None,
    ):
        if not custom_param_list:
            return logits
        import json
        import torch
        # batched raw top-k over the active rows (one kernel instead of B)
        active = []
        for i, pd in enumerate(custom_param_list):
            if pd is None or not (pd.get("force_tokens") or pd.get("capture_path")):
                continue
            req = pd.get("__req__")
            step = len(req.output_ids)  # requires disable_overlap_schedule=True
            n_steps = pd.get("capture_steps") or len(pd.get("force_tokens") or [])
            if step < n_steps:
                active.append((i, pd, step, n_steps))
        if not active:
            return logits
        k = int(active[0][1].get("capture_topk", 20))
        rows = torch.tensor([i for i, *_ in active], device=logits.device)
        sub = logits[rows].float()
        lse = torch.logsumexp(sub, dim=-1, keepdim=True)
        vals, idx = (sub - lse).topk(k, dim=-1)
        idx_c, vals_c = idx.cpu(), vals.cpu()
        for j, (i, pd, step, n_steps) in enumerate(active):
            store = pd.setdefault("_captured", {})
            store[step] = [idx_c[j].tolist(), vals_c[j].tolist()]  # idempotent on recompute
            targets = pd.get("force_tokens")
            if targets:
                logits[i, :] = -float("inf")
                logits[i, int(targets[step])] = 0.0
            if step == n_steps - 1 and pd.get("capture_path"):
                with open(pd["capture_path"], "w") as f:
                    json.dump([store[k] for k in sorted(store)], f)
        return logits
