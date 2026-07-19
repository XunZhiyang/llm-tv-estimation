import torch

from experiments.inference_engines.regime import build_batch


def test_iso_is_fixed_across_repeats():
    X = torch.arange(8 * 10).reshape(8, 10)
    b0 = build_batch(X, regime="warm_iso", k=0, filler_pool=X, seed=1)
    b1 = build_batch(X, regime="warm_iso", k=5, filler_pool=X, seed=1)
    assert b0["B"] == 8 and b1["B"] == 8
    assert torch.equal(b0["batch"], b1["batch"])          # identical batch every repeat
    assert b0["target_rows"] == list(range(8))


def test_serve_varies_B_but_preserves_targets():
    X = torch.arange(8 * 10).reshape(8, 10)
    filler = torch.arange(100 * 10).reshape(100, 10)
    sizes = set()
    for k in range(8):
        b = build_batch(X, regime="warm_serve", k=k, filler_pool=filler, seed=1)
        sizes.add(b["B"])
        tr = b["target_rows"]
        assert torch.equal(b["batch"][tr], X)             # targets recoverable, unchanged
    assert len(sizes) > 1                                  # B genuinely varies


def test_fresh_batches_like_iso():
    X = torch.arange(8 * 10).reshape(8, 10)
    b = build_batch(X, regime="fresh", k=3, filler_pool=X, seed=1)
    assert b["B"] == 8 and torch.equal(b["batch"], X)
