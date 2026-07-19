"""Unified storage schema + manifest for the per-NTP noise (sigma) taxonomy study.

One ``.pt`` file per score job, keyed by ``<scorer>__<source>__<regime>__<path>``. Every job
stores top-``K_save`` raw logits (idx + value) for every (trajectory, repeat, position) cell --
NOT just the top-20 sampling support -- so all downstream analysis (sigma-vs-K, top-k union, TV)
can choose any k <= K_save offline. A top-level ``manifest.json`` (jobs keyed by job_key, so
updates are idempotent) ties the run together; analysis reads only the manifest + its files.

``K`` is per-job (warm=16, fresh=8). ``batch_B`` / ``batch_sig`` length follows the job's K.

Full-vocab calibration subset only: when ``k_save >= vocab_size`` the orchestrator stores a
dense ``logits_full`` field instead of idx/val (HF only -- engines have no full-vocab path).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def job_key(job) -> str:
    return f"{job['scorer']}__{job['source']}__{job['regime']}__{job['path']}"


def new_job(scorer, source, regime, path, N, K, n, K_save, prompt_len, tv_only=False):
    """Allocate an empty job with all tensors pre-sized to the declared shapes.

    ``tv_only``: store ONLY the target token's log-prob per (traj, repeat, pos) in a
    dense ``target_logp`` (N,K,n) field instead of the full top-``K_save`` (idx+val).
    This is the minimal representation for TV / self-TV / sigma -- 1/(3*K_save) the
    storage -- so K (repeats) can be pushed to 1e4+ for a low self-TV noise floor.
    It drops the top-k support, so topk_union / sigma-vs-K must use a full-topk job."""
    job = {
        "scorer": scorer, "source": source, "regime": regime, "path": path,
        "X": torch.zeros((N, prompt_len + n), dtype=torch.long),
        "batch_B": torch.zeros((K,), dtype=torch.int32),
        "batch_sig": [""] * K,
        "sample_logprobs": torch.zeros((N, n), dtype=torch.float16),
        "meta": {
            "scorer": scorer, "source": source, "regime": regime, "path": path,
            "N": N, "K": K, "n": n, "K_save": K_save, "prompt_len": prompt_len,
            "tv_only": bool(tv_only),
        },
    }
    if tv_only:
        # -inf = own-zero (target outside this repeat's top-k support), same convention
        # the full-topk path yields after target extraction.
        job["target_logp"] = torch.full((N, K, n), float("-inf"), dtype=torch.float16)
    else:
        # padding is unambiguous: idx=-1 / val=-inf for cols an engine didn't return
        # (avoids colliding with real token-0; analysis treats non-finite as absent).
        job["topk_idx"] = torch.full((N, K, n, K_save), -1, dtype=torch.int32)
        job["topk_val"] = torch.full((N, K, n, K_save), float("-inf"), dtype=torch.float16)
    return job


def save_job(out_dir, job):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(job, out_dir / f"{job_key(job)}.pt")


def load_job(out_dir, key):
    return torch.load(Path(out_dir) / f"{key}.pt", weights_only=False)


def update_manifest(out_dir, job, md5="", status="done"):
    """Insert/replace this job's manifest row, keyed by job_key (idempotent)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mpath = out_dir / "manifest.json"
    man = {"jobs": {}}
    if mpath.exists():
        try:
            man = json.loads(mpath.read_text())
        except json.JSONDecodeError:
            man = {"jobs": {}}
    if not isinstance(man.get("jobs"), dict):
        man = {"jobs": {}}
    key = job_key(job)
    man["jobs"][key] = {
        "scorer": job["scorer"], "source": job["source"],
        "regime": job["regime"], "path": job["path"],
        "K": job["meta"]["K"], "K_save": job["meta"]["K_save"],
        "file": f"{key}.pt", "md5": md5, "status": status,
    }
    mpath.write_text(json.dumps(man, indent=2))
    return man
