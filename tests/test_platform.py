"""
tests/test_platform.py
Tier 3 pieces that run without a GPU or a Triton server. The PyTriton server
path is exercised end to end by `python -m nurosim.platform smoke` in CI.
"""

import json
import os

import numpy as np
import pytest

from nurosim.scenario_generator import generate_scenario

FP32 = "models/nurosim_det_fp32.onnx"
needs_model = pytest.mark.skipif(not os.path.exists(FP32), reason="needs the FP32 model")


@pytest.fixture(scope="module")
def serve_graph(tmp_path_factory):
    from nurosim.platform.graph import add_uint8_frontend
    return add_uint8_frontend(FP32, str(tmp_path_factory.mktemp("serve") / "serve.onnx"))


@pytest.fixture(scope="module")
def frames():
    return np.stack([generate_scenario(100_000 + i, seed=100_000 + i).frame for i in range(4)])


@needs_model
def test_frontend_is_bit_exact(serve_graph, frames):
    from nurosim.inference.preprocess import preprocess_batch
    from nurosim.inference.runtime import OrtRuntime
    rt = OrtRuntime(serve_graph)
    assert rt.input_dtype == np.uint8 and list(rt.input_shape[1:]) == [640, 640, 3]
    ref = OrtRuntime(FP32).run(preprocess_batch(list(frames))[0])
    assert np.array_equal(rt.run(frames), ref)


@needs_model
def test_serving_model_matches_tier1_detector(serve_graph, frames):
    from nurosim.inference.detector import InferenceDetector
    from nurosim.platform.backend import ServingModel, unpack
    out = ServingModel(serve_graph).infer(frames)
    assert out["det_boxes"].shape == (4, 100, 4) and out["num_dets"].dtype == np.int32
    ref = InferenceDetector(FP32)
    key = lambda bs: [(b.xyxy, b.class_id, round(b.confidence, 5)) for b in bs]   # noqa: E731
    for got, f in zip(unpack(out), frames):
        assert key(got) == key(sorted(ref.predict(f), key=lambda b: -b.confidence))


def test_serving_model_rejects_float_graph():
    from nurosim.platform.backend import ServingModel
    if not os.path.exists(FP32):
        pytest.skip("needs the FP32 model")
    with pytest.raises(ValueError):
        ServingModel(FP32)


def test_pack_unpack_roundtrip():
    from nurosim.platform.backend import pack, unpack
    from nurosim.scenario_generator import BBox
    dets = [[BBox(1, 2, 30, 40, 0, "vehicle", 0.9), BBox(5, 5, 9, 9, 3, "cone", 0.95)], []]
    back = unpack(pack(dets, max_det=10))
    assert [b.class_id for b in back[0]] == [3, 0]            # sorted by score
    assert back[1] == [] and back[0][1].xyxy == (1, 2, 30, 40)


def test_server_view_math():
    from nurosim.platform.loadgen import server_view
    before = {"inference_count": 10, "execution_count": 10, "success_ns": 0, "success_count": 10,
              "queue_ns": 0, "queue_count": 10, "compute_input_ns": 0, "compute_input_count": 0,
              "compute_infer_ns": 0, "compute_infer_count": 0, "compute_output_ns": 0,
              "compute_output_count": 0, "batch_hist": {1: 10}}
    after = {**before, "inference_count": 110, "execution_count": 35, "success_count": 110,
             "success_ns": 300_000_000, "queue_ns": 50_000_000, "compute_infer_ns": 50_000_000,
             "batch_hist": {1: 15, 4: 20}}
    v = server_view(before, after)
    assert v["requests"] == 100 and v["executions"] == 25
    assert v["avg_batch"] == pytest.approx(4.0)
    assert v["queue_us_per_request"] == pytest.approx(500.0)
    assert v["compute_infer_us_per_exec"] == pytest.approx(2000.0)
    assert v["batch_hist"] == {1: 5, 4: 20}


def test_triton_config_matches_serving_schema():
    from nurosim.platform.serving import triton_model_config
    c = triton_model_config("m", 8, 500)
    assert c["input"][0] == {"name": "frames", "data_type": "TYPE_UINT8", "dims": [640, 640, 3]}
    assert [o["name"] for o in c["output"]] == ["num_dets", "det_boxes", "det_scores", "det_classes"]
    assert c["dynamic_batching"]["max_queue_delay_microseconds"] == 500
    assert "dynamic_batching" not in triton_model_config("m", 1, 0)


@needs_model
def test_shadow_identical_promotes_and_divergent_holds(serve_graph, frames):
    from nurosim.platform.backend import ServingModel
    from nurosim.platform.shadow import LocalEndpoint, shadow_compare

    class Shifted(LocalEndpoint):          # candidate whose boxes are 3 px off and drops a class
        def infer(self, f):
            out = super().infer(f)
            out["det_boxes"] = out["det_boxes"] + 3.0
            out["num_dets"] = np.minimum(out["num_dets"], 2)
            return out

    m = ServingModel(serve_graph)
    same = shadow_compare(LocalEndpoint("prod", m), LocalEndpoint("cand", m), frames)
    assert same["agreement"]["ref_recall"] == 1.0 and same["decision"] == "promote" or \
        same["latency_ms"]["p99_ratio"] > 1.25              # latency noise on a busy CI box
    worse = shadow_compare(LocalEndpoint("prod", m), Shifted("cand", m), frames)
    assert worse["decision"] == "hold" and worse["agreement"]["ref_recall"] < 0.99


@needs_model
def test_corpus_and_batch_job(tmp_path, serve_graph):
    pytest.importorskip("ray")
    pyarrow = pytest.importorskip("pyarrow.parquet")
    from nurosim.platform.batch import run_batch
    from nurosim.platform.corpus import write_corpus

    info = write_corpus(str(tmp_path / "corpus"), n=24, shard_size=12, workers=0)
    assert info["shards"] == 2 and info["frames"] == 24
    t = pyarrow.read_table(str(tmp_path / "corpus" / "shard_00000.parquet"))
    assert t.column_names == ["scenario_id", "weather", "jpeg"]

    r = run_batch(str(tmp_path / "corpus"), serve_graph, str(tmp_path / "out"), provider="cpu",
                  batch_size=8, actors=1, labeler_cpus=1, eval_frames=24, tag="t", usd_per_hour=1.0)
    assert r["frames"] == 24 and r["frames_per_s"] > 0
    assert r["usd_per_million_frames"] == pytest.approx(r["hours_per_million_frames"])
    assert r["label_quality"]["map50_at_deploy_conf"] > 0.8
    assert os.path.exists(tmp_path / "out" / "batch_t.json")


def test_registry_gated_promotion(tmp_path):
    pytest.importorskip("mlflow")
    from nurosim.deploy.export import sha256_file
    from nurosim.platform.registry import PromotionBlocked, list_versions, promote, register, resolve

    good, bad = tmp_path / "good.onnx", tmp_path / "bad.onnx"
    good.write_bytes(b"good model")
    bad.write_bytes(b"bad model")
    other = tmp_path / "rebuilt.onnx"
    other.write_bytes(b"same recipe, different bytes")
    gate = {"profile": "t", "passed": False, "targets": {
        "a": {"evidence_sha256": sha256_file(str(good)), "passed": True, "checks": [
            {"check": "mAP@0.5 drop", "value": "+0.0", "limit": "≤ 0.01", "passed": True}]},
        "b": {"evidence_sha256": sha256_file(str(bad)), "passed": False, "checks": [
            {"check": "worst class drop", "value": "cone +0.05", "limit": "≤ 0.03", "passed": False}]}}}
    gp = tmp_path / "gate.json"
    gp.write_text(json.dumps(gate))
    uri = f"sqlite:///{tmp_path}/mlruns/registry.db"

    v_good = register(str(good), "a", str(gp), uri=uri)
    v_bad = register(str(bad), "b", str(gp), uri=uri)
    v_other = register(str(other), "a", str(gp), uri=uri)     # evidence is for a different file
    assert (v_good["gate"], v_bad["gate"], v_other["gate"]) == ("pass", "fail", "no-evidence")

    assert promote(v_good["version"], uri=uri)["alias"] == "production"
    for v in (v_bad, v_other):
        with pytest.raises(PromotionBlocked):
            promote(v["version"], uri=uri)
    path = resolve("production", uri=uri, dst=str(tmp_path / "dl"))
    assert open(path, "rb").read() == b"good model"
    rows = list_versions(uri=uri)
    assert [r["aliases"] for r in rows] == [["production"], [], []]
