import json

from experiments.inference_engines import storage


def test_job_roundtrip_and_manifest(tmp_path):
    job = storage.new_job(scorer="hf-auto", source="hf-flash", regime="warm_iso",
                          path="decode", N=2, K=3, n=4, K_save=5, prompt_len=1)
    job["topk_idx"][:] = 7
    storage.save_job(tmp_path, job)
    storage.update_manifest(tmp_path, job, md5="abc", status="done")
    storage.update_manifest(tmp_path, job, md5="abc", status="done")   # idempotent
    man = json.loads((tmp_path / "manifest.json").read_text())
    assert len(man["jobs"]) == 1
    key = storage.job_key(job)
    assert key == "hf-auto__hf-flash__warm_iso__decode"
    loaded = storage.load_job(tmp_path, key)
    assert int(loaded["topk_idx"][0, 0, 0, 0]) == 7
    assert loaded["meta"]["K_save"] == 5
    assert tuple(loaded["topk_idx"].shape) == (2, 3, 4, 5)
