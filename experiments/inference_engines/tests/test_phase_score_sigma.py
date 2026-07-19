import json
import sys

import torch

import experiments.inference_engines.phase_score_sigma as ps


class FakeOracle:
    vocab_size = 1000

    def __init__(self):
        self.calls = []

    def _topk(self, batch, k_save, tag):
        B, L = batch.shape
        n = L - 1                                   # prompt_len=1 in tests
        self.calls.append(tag)
        ti = torch.arange(k_save).view(1, 1, k_save).expand(B, n, k_save).int().clone()
        tv = -torch.arange(k_save).float().view(1, 1, k_save).expand(B, n, k_save).clone()
        return None, ti, tv

    def score_kv_batched(self, batch, prefix_len, return_topk_logits, k_save):
        return self._topk(batch, k_save, "hf-decode")

    def score_prefill_batched(self, batch, prefix_len, k_save, return_topk_logits, chunk=64):
        return self._topk(batch, k_save, "hf-prefill")

    def score_replay_batched(self, batch, prefix_len, return_topk_logits, k_save):
        return self._topk(batch, k_save, "engine-replay")

    def score_seq_batched(self, batch, prefix_len, return_topk_logits, k_save):
        return self._topk(batch, k_save, "engine-seq")


def test_path_mapping_routes_correctly():
    """The May-bug guard: decode->generation oracle, prefill->whole-seq oracle, per engine."""
    o = FakeOracle()
    x = torch.zeros(2, 5, dtype=torch.long)
    ps.score_topk(o, "hf", "decode", x, 1, 8)
    ps.score_topk(o, "hf", "prefill", x, 1, 8)
    ps.score_topk(o, "vllm", "decode", x, 1, 8)
    ps.score_topk(o, "sglang", "prefill", x, 1, 8)
    assert o.calls == ["hf-decode", "hf-prefill", "engine-replay", "engine-seq"]


def test_engine_score_path():
    assert ps.engine_score_path("decode") == "replay"
    assert ps.engine_score_path("prefill") == "seq"


def _run(tmp_path, monkeypatch, regime):
    N, plen, n = 3, 1, 4
    X = torch.randint(0, 100, (N, plen + n))
    src = {"X": X, "meta": {"n": n, "prompt_len": plen, "prompt": "p"},
           "sample_logprobs": torch.zeros(N, n)}
    xp = tmp_path / "src.pt"
    torch.save(src, xp)
    monkeypatch.setattr(ps, "make_oracle", lambda *a, **k: FakeOracle())
    argv = ["prog", "--engine", "hf", "--sdpa_backend", "auto", "--path", "decode",
            "--regime", regime, "--X_path", str(xp), "--source", "hf-flash",
            "--K", "3", "--K_save", "8", "--out_dir", str(tmp_path), "--top_k", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    ps.main()
    key = f"hf-auto__hf-flash__{regime}__decode"
    job = torch.load(tmp_path / f"{key}.pt", weights_only=False)
    man = json.loads((tmp_path / "manifest.json").read_text())
    return job, man, key, N, n


def test_warm_iso_constant_B(tmp_path, monkeypatch):
    job, man, key, N, n = _run(tmp_path, monkeypatch, "warm_iso")
    assert tuple(job["topk_idx"].shape) == (N, 3, n, 8)
    assert set(job["batch_B"].tolist()) == {N}            # iso => constant batch
    assert man["jobs"][key]["status"] == "done"
    assert man["jobs"][key]["K_save"] == 8


def test_warm_serve_targets_preserved(tmp_path, monkeypatch):
    job, man, key, N, n = _run(tmp_path, monkeypatch, "warm_serve")
    assert tuple(job["topk_idx"].shape) == (N, 3, n, 8)
    assert all(b >= N for b in job["batch_B"].tolist())   # serve >= N (filler prepended)
    assert man["jobs"][key]["status"] == "done"
