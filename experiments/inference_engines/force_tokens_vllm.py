"""Teacher-forcing logits processor for vllm V1 (>=0.10 custom LogitsProcessor API).

Forces generation to follow per-request `extra_args["force_tokens"]` while the
engine runs its real decode path. Register at engine construction:

    LLM(..., logits_processors=[
        "experiments.inference_engines.force_tokens_vllm.ForceTokensLogitsProcessor"])

and per request:

    SamplingParams(max_tokens=n, ..., extra_args={"force_tokens": target_list})

The engine-core process imports this module by path — PYTHONPATH must include
the project root in the environment that constructs the LLM.
"""
from __future__ import annotations

import torch

try:
    from vllm.v1.sample.logits_processor import (
        BatchUpdate, LogitsProcessor, MoveDirectionality)
except ImportError:  # older layout
    from vllm.v1.sample.logits_processor.interface import (  # type: ignore
        BatchUpdate, LogitsProcessor, MoveDirectionality)


class ForceTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, vllm_config, device: torch.device, is_pin_memory: bool):
        # batch index -> (targets, live output_tok_ids reference)
        self.reqs: dict[int, tuple[list[int], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: "BatchUpdate | None") -> None:
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.reqs.pop(index, None)
        for index, params, _prompt_ids, out_ids in batch_update.added:
            extra = getattr(params, "extra_args", None) or {}
            targets = extra.get("force_tokens")
            if targets:
                self.reqs[index] = (targets, out_ids)
            else:
                self.reqs.pop(index, None)
        for i1, i2, direc in batch_update.moved:
            if direc == MoveDirectionality.SWAP:
                a, b = self.reqs.pop(i1, None), self.reqs.pop(i2, None)
                if b is not None:
                    self.reqs[i1] = b
                if a is not None:
                    self.reqs[i2] = a
            else:  # UNIDIRECTIONAL i1 -> i2
                a = self.reqs.pop(i1, None)
                self.reqs.pop(i2, None)
                if a is not None:
                    self.reqs[i2] = a

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for idx, (targets, out_ids) in self.reqs.items():
            step = len(out_ids)
            if step < len(targets) and idx < logits.shape[0]:
                t = int(targets[step])
                logits[idx, :] = float("-inf")
                logits[idx, t] = 0.0
        return logits
