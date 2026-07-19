"""
Core TV distance estimation library.

Implements:
  - batched_log_likelihood: per-sequence log P(generated | prompt) via full-forward
  - incremental_log_likelihood: same, via KV-cache token-by-token (matches generation path)
  - generate_sequences: batch sampling from a model (optionally records per-token logprobs)
  - estimate_tv_v2: White-Box-EstV2 (density ratio, samples only from pi)
  - estimate_tv_v1: White-Box-EstV1 (indicator, samples from both pi and mu)
  - load_model: load HF model with optional bitsandbytes quantization
"""

import math
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


# ── Model loading ─────────────────────────────────────────────────────────────

QUANT_TYPES = ("bf16", "fp16", "int8", "nf4", "fp4")
ATTN_IMPLS = ("sdpa", "eager", "flash_attention_2", "flash_attention_3")


def load_model(
    model_name_or_path: str,
    quant_type: str = "bf16",
    attn_impl: str = "sdpa",
) -> AutoModelForCausalLM:
    """
    Load a causal LM with optional bitsandbytes quantization.

    quant_type: 'bf16' | 'fp16' | 'int8' | 'nf4' | 'fp4'
      - 'bf16' / 'fp16' are plain dtypes, no quantization
      - 'int8' / 'nf4' / 'fp4' use bitsandbytes
    attn_impl: 'sdpa' | 'eager' | 'flash_attention_2' | 'flash_attention_3'
    """
    if quant_type not in QUANT_TYPES:
        raise ValueError(f"quant_type must be one of {QUANT_TYPES}")
    if attn_impl not in ATTN_IMPLS:
        raise ValueError(f"attn_impl must be one of {ATTN_IMPLS}")

    if quant_type in ("bf16", "fp16"):
        dtype = torch.bfloat16 if quant_type == "bf16" else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    elif quant_type == "int8":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    elif quant_type == "nf4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation=attn_impl,
        )
    elif quant_type == "fp4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="fp4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=bnb_cfg,
            device_map="auto",
            attn_implementation=attn_impl,
        )

    model.eval()
    return model


def load_tokenizer(model_name_or_path: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_name_or_path)
    # The sampler below never pads, but setting pad_token_id suppresses HF warnings
    # and keeps the tokenizer usable with other tooling.
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return tok


def _model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """
    Return the device where the model expects its input.

    next(model.parameters()).device is fragile for sharded / offloaded models
    (the first parameter may not be the embedding).  Use the embedding weight
    instead, which is guaranteed to be the input entry point.
    """
    return model.get_input_embeddings().weight.device


# ── Core computation ──────────────────────────────────────────────────────────

@torch.inference_mode()
def batched_log_likelihood(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,   # (B, seq_len)  no padding, same prompt for all
    prompt_len: int,
) -> torch.Tensor:             # (B,)
    """
    Compute sum_t log P(x_t | x_{<t}) for the generated portion only.

    Assumes all sequences share the same prompt prefix (no padding).
    prompt_len: number of prompt tokens to mask out of the loss.
    """
    device = _model_input_device(model)
    input_ids = input_ids.to(device)

    logits = model(input_ids).logits.float()  # (B, seq_len, V) in float32

    # logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :].contiguous()    # (B, seq_len-1, V)
    shift_labels = input_ids[:, 1:].clone()           # (B, seq_len-1)

    # Mask the prompt: positions 0..prompt_len-2 in shift_labels predict
    # tokens 1..prompt_len-1, which are still inside the prompt.
    shift_labels[:, : prompt_len - 1] = -100

    B, L = shift_labels.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * L, -1),
        shift_labels.reshape(B * L),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, L)  # (B, seq_len-1)

    return -loss.sum(dim=1)  # (B,)  sum log prob of generated tokens


@torch.inference_mode()
def batched_per_token_log_likelihood(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,   # (B, prompt_len + n_tokens)
    prompt_len: int,
) -> torch.Tensor:             # (B, n_tokens)
    """
    Per-token log-probabilities for the generated portion.
    Entry [b, t] = log P(x_{prompt_len+t} | x_{<prompt_len+t}).
    Summing along dim=1 recovers batched_log_likelihood.
    """
    device = _model_input_device(model)
    input_ids = input_ids.to(device)
    logits = model(input_ids).logits.float()  # float32 for numerical precision
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].clone()
    shift_labels[:, : prompt_len - 1] = -100
    B, L = shift_labels.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * L, -1),
        shift_labels.reshape(B * L),
        reduction="none",
        ignore_index=-100,
    ).reshape(B, L)
    return -loss[:, prompt_len - 1:]  # (B, n_tokens)


@torch.inference_mode()
def incremental_log_likelihood(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,   # (N, prompt_len + n_tokens)
    prompt_len: int,
    batch_size: int = 64,
) -> torch.Tensor:             # (N,)
    """
    Compute sum log P(x_t | x_{<t}) for the generated portion using KV-cache
    token-by-token forward, numerically matching the autoregressive generation path.

    This avoids the bf16 shape-dependent GEMM discrepancy that causes
    full-sequence forward (batched_log_likelihood) to give different logits
    than KV-cache decode at the same causal position.

    Delegates to incremental_per_token_log_likelihood().sum(dim=1) so that the
    float32 reduction order matches what the caller would get from summing the
    generation-time per-token logprobs (which is the pi-side L_pi source). This
    guarantees bitwise same_model = 0.
    """
    return incremental_per_token_log_likelihood(
        model, input_ids, prompt_len, batch_size
    ).sum(dim=1)


@torch.inference_mode()
def incremental_per_token_log_likelihood(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,   # (N, prompt_len + n_tokens)
    prompt_len: int,
    batch_size: int = 64,
) -> torch.Tensor:             # (N, n_tokens)
    """
    Per-token log-probabilities using KV-cache token-by-token forward.
    Summing along dim=1 recovers incremental_log_likelihood.
    """
    device = _model_input_device(model)
    N, seq_len = input_ids.shape
    n_tokens = seq_len - prompt_len
    all_lps = []
    n_batches = math.ceil(N / batch_size)

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        batch = input_ids[start : start + batch_size].to(device)
        bs = batch.shape[0]
        prompt = batch[:, :prompt_len].contiguous()

        out = model(prompt, use_cache=True)
        kv = out.past_key_values
        step_lps = []

        for t in range(n_tokens):
            logits = out.logits[:, -1, :].float()
            log_probs = F.log_softmax(logits, dim=-1)
            target = batch[:, prompt_len + t]
            step_lps.append(
                log_probs.gather(1, target.unsqueeze(1)).squeeze(1).cpu()
            )

            if t < n_tokens - 1:
                if (t + 1) % 50 == 0:
                    print(f"\r      eval batch {batch_idx+1}/{n_batches}: "
                          f"token {t+1}/{n_tokens}", end="", flush=True)

                out = model(
                    input_ids=target.unsqueeze(1),
                    past_key_values=kv,
                    use_cache=True,
                )
                kv = out.past_key_values

        if n_tokens >= 50:
            print(f"\r      eval batch {batch_idx+1}/{n_batches}: "
                  f"token {n_tokens}/{n_tokens}    ", flush=True)

        all_lps.append(torch.stack(step_lps, dim=1))

    return torch.cat(all_lps, dim=0)


@torch.inference_mode()
def generate_sequences(
    model: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,  # (1, prompt_len)
    N: int,
    n_new_tokens: int,
    batch_size: int = 64,
    return_logprobs: bool = False,
):
    """
    Sample N sequences of exactly n_new_tokens from the model given a fixed prompt.

    Returns (N, prompt_len + n_new_tokens) tensor of token ids.
    If return_logprobs=True, also returns (N, n_new_tokens) per-token log-probs
    from the generation (KV-cache) path — these are L_pi_sample at zero extra cost.

    We use a manual autoregressive loop rather than model.generate(..., min_new_tokens=...)
    because HF's min-length logic suppresses EOS before the minimum length is reached.
    That changes the sampling distribution. Here EOS is treated like any other token:
    we always sample exactly n_new_tokens steps with no early stopping and no padding.
    """
    if n_new_tokens < 1:
        raise ValueError("n_new_tokens must be >= 1")

    device = _model_input_device(model)
    prompt_ids = prompt_ids.to(device)
    all_seqs = []
    all_logprobs = [] if return_logprobs else None
    n_batches = math.ceil(N / batch_size)

    for batch_idx, start in enumerate(range(0, N, batch_size)):
        bs = min(batch_size, N - start)
        batch_input = prompt_ids.expand(bs, -1).contiguous()  # (bs, prompt_len)

        outputs = model(
            input_ids=batch_input,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        generated_tokens = []
        step_logprobs = [] if return_logprobs else None

        for step in range(n_new_tokens):
            next_token_logits = outputs.logits[:, -1, :].float()
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(log_probs.exp(), num_samples=1)

            if return_logprobs:
                step_logprobs.append(
                    log_probs.gather(1, next_token).cpu()
                )

            generated_tokens.append(next_token.cpu())

            if step == n_new_tokens - 1:
                break

            if (step + 1) % 50 == 0:
                print(f"\r      gen batch {batch_idx+1}/{n_batches}: "
                      f"token {step+1}/{n_new_tokens}", end="", flush=True)

            outputs = model(
                input_ids=next_token.to(device),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

        if n_new_tokens >= 50:
            print(f"\r      gen batch {batch_idx+1}/{n_batches}: "
                  f"token {n_new_tokens}/{n_new_tokens}    ", flush=True)

        out = torch.cat([batch_input.cpu(), torch.cat(generated_tokens, dim=1)], dim=1)
        all_seqs.append(out)
        if return_logprobs:
            all_logprobs.append(torch.cat(step_logprobs, dim=1))

    seqs = torch.cat(all_seqs, dim=0)  # (N, prompt_len + n_new_tokens)
    if return_logprobs:
        return seqs, torch.cat(all_logprobs, dim=0)  # + (N, n_new_tokens)
    return seqs


@torch.inference_mode()
def _paired_log_likelihoods(
    model_pi,
    model_mu,
    sequences: torch.Tensor,
    prompt_len: int,
    batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute paired log-likelihoods under pi and mu for a fixed set of sequences."""
    all_l_pi = []
    all_l_mu = []

    for start in range(0, sequences.shape[0], batch_size):
        batch = sequences[start : start + batch_size]
        l_pi = batched_log_likelihood(model_pi, batch, prompt_len)
        l_mu = batched_log_likelihood(model_mu, batch, prompt_len)
        all_l_pi.append(l_pi.cpu())
        all_l_mu.append(l_mu.cpu())

    return torch.cat(all_l_pi), torch.cat(all_l_mu)


def estimate_tv_v2_from_sequences(
    model_pi,
    model_mu,
    sequences: torch.Tensor,
    prompt_len: int,
    batch_size: int = 64,
    l_pi=None,
) -> dict:
    """Estimate V2 on a pre-sampled batch of sequences from pi.

    If l_pi is provided (sum log-probs recorded during generation via
    return_logprobs=True), it is used directly and L_mu is computed via
    incremental (KV-cache) eval to avoid bf16 shape-dependent GEMM bias.
    Otherwise falls back to full-forward for both (legacy behavior).
    """
    if l_pi is not None:
        if l_pi.dim() != 1 or l_pi.shape[0] != sequences.shape[0]:
            raise ValueError(
                f"l_pi must be 1D with shape ({sequences.shape[0]},), "
                f"got {tuple(l_pi.shape)}. Did you forget to sum along dim=1?"
            )
        l_mu = incremental_log_likelihood(model_mu, sequences, prompt_len, batch_size)
    else:
        l_pi, l_mu = _paired_log_likelihoods(model_pi, model_mu, sequences, prompt_len, batch_size)
    log_ratio = l_mu - l_pi
    cs = torch.where(
        log_ratio > 0,
        torch.zeros_like(log_ratio),
        1.0 - torch.exp(log_ratio),
    )
    n_samples = sequences.shape[0]
    mean = cs.mean().item()
    stderr = cs.std().item() / math.sqrt(n_samples)
    return {
        "tv": mean,
        "stderr": stderr,
        "N": n_samples,
        "estimator": "v2",
        "samples": cs.tolist(),
    }


def estimate_tv_v1_from_sequences(
    model_pi,
    model_mu,
    seqs_pi: torch.Tensor,
    seqs_mu: torch.Tensor,
    prompt_len: int,
    batch_size: int = 64,
) -> dict:
    """Estimate V1 from pre-sampled batches from pi and mu."""
    l_pi_on_pi, l_mu_on_pi = _paired_log_likelihoods(model_pi, model_mu, seqs_pi, prompt_len, batch_size)
    ind_pi = (l_mu_on_pi > l_pi_on_pi).float()
    p1_hat = ind_pi.mean().item()

    l_pi_on_mu, l_mu_on_mu = _paired_log_likelihoods(model_pi, model_mu, seqs_mu, prompt_len, batch_size)
    ind_mu = (l_pi_on_mu >= l_mu_on_mu).float()
    p2_hat = ind_mu.mean().item()

    n_samples = seqs_pi.shape[0]
    tv = 1.0 - (p1_hat + p2_hat)
    se_p1 = ind_pi.std().item() / math.sqrt(n_samples)
    se_p2 = ind_mu.std().item() / math.sqrt(seqs_mu.shape[0])
    stderr = math.sqrt(se_p1**2 + se_p2**2)

    return {
        "tv": tv,
        "stderr": stderr,
        "N": n_samples,
        "estimator": "v1",
        "p1": p1_hat,
        "p2": p2_hat,
    }


# ── Estimators ────────────────────────────────────────────────────────────────

def estimate_tv_v2(
    model_pi,
    model_mu,
    tokenizer,
    prompt_text: str,
    N: int,
    n_new_tokens: int,
    batch_size: int = 64,
) -> dict:
    """
    White-Box-EstV2: TV(pi, mu) = E_{X~pi}[(1 - mu(X)/pi(X))_+]

    Only samples from pi. Returns a dict with estimate and stderr.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    sequences = generate_sequences(model_pi, prompt_ids, N, n_new_tokens, batch_size)
    return estimate_tv_v2_from_sequences(model_pi, model_mu, sequences, prompt_len, batch_size)


def estimate_tv_v1(
    model_pi,
    model_mu,
    tokenizer,
    prompt_text: str,
    N: int,
    n_new_tokens: int,
    batch_size: int = 64,
) -> dict:
    """
    White-Box-EstV1: TV = 1 - (p1 + p2)
      p1 = P_{X~pi}[mu(X) > pi(X)]
      p2 = P_{Y~mu}[pi(Y) >= mu(Y)]

    Samples N from pi and N from mu. Returns a dict with estimate and stderr.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    seqs_pi = generate_sequences(model_pi, prompt_ids, N, n_new_tokens, batch_size)
    seqs_mu = generate_sequences(model_mu, prompt_ids, N, n_new_tokens, batch_size)
    return estimate_tv_v1_from_sequences(model_pi, model_mu, seqs_pi, seqs_mu, prompt_len, batch_size)


# ── Convergence helper ────────────────────────────────────────────────────────

def estimate_tv_v2_convergence_from_sequences(
    model_pi,
    model_mu,
    sequences: torch.Tensor,
    prompt_len: int,
    batch_size: int = 64,
) -> list[dict]:
    """Compute convergence checkpoints from a fixed set of sequences sampled from pi."""
    result_full = estimate_tv_v2_from_sequences(
        model_pi,
        model_mu,
        sequences,
        prompt_len,
        batch_size,
    )
    all_cs = torch.tensor(result_full["samples"])
    if not torch.isfinite(all_cs).all():
        n_bad = (~torch.isfinite(all_cs)).sum().item()
        raise ValueError(
            f"estimate_tv_v2 returned {n_bad} non-finite values; "
            "check for numerical issues in log-likelihood computation"
        )

    checkpoints = []
    n_samples = 100
    while n_samples <= sequences.shape[0]:
        subset = all_cs[:n_samples]
        checkpoints.append({
            "N": n_samples,
            "tv": subset.mean().item(),
            "stderr": subset.std().item() / math.sqrt(n_samples),
        })
        n_samples += 100

    return checkpoints


def estimate_tv_v2_convergence(
    model_pi,
    model_mu,
    tokenizer,
    prompt_text: str,
    N_max: int,
    n_new_tokens: int,
    batch_size: int = 64,
) -> list[dict]:
    """
    Draw N_max samples once, then compute running TV estimate at
    N = 100, 200, 400, 800, 1600, 3200, ... up to N_max.
    Returns a list of dicts, one per checkpoint N.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prompt_len = prompt_ids.shape[1]
    sequences = generate_sequences(model_pi, prompt_ids, N_max, n_new_tokens, batch_size)
    return estimate_tv_v2_convergence_from_sequences(
        model_pi,
        model_mu,
        sequences,
        prompt_len,
        batch_size,
    )
