"""
python -m detect2deploy.platform <command>

  export         serving graphs (uint8 frame input) for ORT and for Triton/TensorRT
  serve          PyTriton server with dynamic batching
  serving-bench  batching-config × concurrency sweep → serving.json / .md / .png
  corpus         JPEG frame corpus as Parquet shards
  batch          Ray Data offboard labelling job
  registry       register | promote | resolve | list   (MLflow, gated promotion)
  shadow         production vs candidate on the same frames → promote / hold
  report         SUMMARY.md for a run directory
  smoke          short end-to-end check on CPU (CI)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback

logger = logging.getLogger("detect2deploy.platform")
BUILD = "build/platform"


def cmd_export(args):
    """serve_<v>.onnx: frontend + model (ORT / PyTriton; NMS on the host).
    serve_<v>_nms.onnx: frontend + model + EfficientNMS (TensorRT plan for Triton)."""
    from detect2deploy.deploy.config import load_config
    from detect2deploy.deploy.trt_prep import add_efficient_nms
    from detect2deploy.platform.graph import add_uint8_frontend

    cfg = load_config(args.config)
    os.makedirs(args.out, exist_ok=True)
    sources = {v: cfg["models"][v] for v in args.variants}
    if args.from_registry:
        from detect2deploy.platform.registry import resolve
        alias = args.from_registry
        sources[f"registry_{alias}"] = resolve(alias, uri=args.uri,
                                               dst=os.path.join(args.out, f"registry_{alias}"))
    for name, src in sources.items():
        if not os.path.exists(src):
            logger.warning("skip %s: %s missing (run `python -m detect2deploy.deploy quantize`)", name, src)
            continue
        print(add_uint8_frontend(src, os.path.join(args.out, f"serve_{name}.onnx")))
        tmp = os.path.join(args.out, f"_{name}_nms.onnx")
        add_efficient_nms(src, tmp, cfg["deploy"]["conf_threshold"], cfg["deploy"]["iou_threshold"], max_det=100)
        print(add_uint8_frontend(tmp, os.path.join(args.out, f"serve_{name}_nms.onnx")))
        os.remove(tmp)


def cmd_serve(args):
    from detect2deploy.platform.graph import add_uint8_frontend
    from detect2deploy.platform.registry import resolve
    from detect2deploy.platform.server import serve
    models = dict(m.split("=", 1) for m in args.model)
    for name, src in models.items():
        if src.startswith("registry:"):          # e.g. production=registry:production
            alias = src.split(":", 1)[1]
            path = resolve(alias, uri=args.uri, dst=os.path.join(BUILD, f"registry_{alias}"))
            models[name] = add_uint8_frontend(path, os.path.join(BUILD, f"serve_registry_{alias}.onnx"))
            logger.info("%s ← registry alias '%s' (%s)", name, alias, os.path.basename(path))
    serve(models, provider=args.provider, max_batch=args.max_batch, queue_delay_us=args.queue_delay_us,
          preferred_batch=args.preferred_batch)


def cmd_serving_bench(args):
    from detect2deploy.platform.serving import plot_serving, run_sweep, serving_markdown
    rep = run_sweep(args.model, args.out, provider=args.provider, backend=args.backend,
                    duration_s=args.duration, concurrency=args.concurrency, url=args.url)
    plot_serving(rep, os.path.join(args.out, "serving.png"))
    md = serving_markdown(rep)
    with open(os.path.join(args.out, "serving.md"), "w") as f:
        f.write(md)
    print(md)


def cmd_corpus(args):
    from detect2deploy.platform.corpus import write_corpus
    info = write_corpus(args.out, n=args.n, shard_size=args.shard_size, quality=args.quality)
    if args.info:
        with open(args.info, "w") as f:
            json.dump(info, f, indent=2)
    print(info)


def cmd_batch(args):
    from detect2deploy.platform.batch import batch_markdown, run_batch
    rep = run_batch(args.corpus, args.model, args.out, provider=args.provider, batch_size=args.batch_size,
                    actors=args.actors, labeler_cpus=args.labeler_cpus, usd_per_hour=args.usd_per_hour,
                    eval_frames=args.eval_frames, tag=args.tag)
    print(batch_markdown([rep]))


def cmd_registry(args):
    from detect2deploy.platform import registry as reg
    if args.action == "register":
        print(json.dumps(reg.register(args.model, args.target, args.gate, args.parity, uri=args.uri), indent=2))
    elif args.action == "promote":
        print(json.dumps(reg.promote(args.version, alias=args.alias, uri=args.uri), indent=2))
    elif args.action == "resolve":
        print(reg.resolve(args.alias, uri=args.uri))
    else:
        rows = reg.list_versions(uri=args.uri)
        if args.json:
            with open(args.json, "w") as f:
                json.dump({"versions": rows, "events": args.event or []}, f, indent=2)
        for r in rows:
            print(r)


def cmd_shadow(args):
    import numpy as np
    from detect2deploy.deploy.seeds import EVAL_SEED
    from detect2deploy.platform.shadow import TritonEndpoint, shadow_compare, shadow_markdown, write_shadow
    from detect2deploy.scenario_generator import generate_scenario
    frames = np.stack([generate_scenario(EVAL_SEED + i, seed=EVAL_SEED + i).frame for i in range(args.frames)])
    r = shadow_compare(TritonEndpoint(args.production, args.url), TritonEndpoint(args.candidate, args.url), frames)
    write_shadow(r, args.out)
    print(shadow_markdown(r))


def cmd_report(args):
    from detect2deploy.platform.report import write_summary
    print(open(write_summary(args.dir)).read())


def cmd_smoke(args):
    """CI: serve two models on CPU through PyTriton, drive load, run a shadow comparison."""
    import asyncio
    from detect2deploy.platform.graph import add_uint8_frontend
    from detect2deploy.platform.loadgen import closed_loop, wait_ready
    from detect2deploy.platform.serving import PyTritonProcess, bench_frames
    from detect2deploy.platform.shadow import TritonEndpoint, shadow_compare, shadow_markdown

    os.makedirs(args.out, exist_ok=True)
    a = add_uint8_frontend("models/d2d_det_fp32.onnx", os.path.join(args.out, "serve_a.onnx"))
    server = PyTritonProcess({"prod": a, "cand": a}, "cpu", 4, 1000, os.path.join(args.out, "server.log"))
    try:
        if not wait_ready("cand", timeout_s=300):
            raise RuntimeError("PyTriton did not become ready — see " + os.path.join(args.out, "server.log"))
        frames = bench_frames(8)
        r = asyncio.run(closed_loop("prod", frames, concurrency=4, duration_s=3, warmup_s=1))
        print(json.dumps(r, indent=2))
        assert r["latency_ms"]["n"] > 0 and r["server"]["requests"] > 0

        # one request through each client flavour, parsed — before the shadow run
        import tritonclient.grpc as g
        import tritonclient.grpc.aio as ga
        from detect2deploy.platform.backend import OUTPUTS
        from detect2deploy.platform.shadow import decode_response

        async def aio_once():
            c = ga.InferenceServerClient("localhost:8001")
            i = ga.InferInput("frames", [1, 640, 640, 3], "UINT8")
            i.set_data_from_numpy(frames[:1])
            res = await c.infer("prod", [i], outputs=[ga.InferRequestedOutput(n) for n in OUTPUTS])
            await c.close()
            return res
        for label, res in (("aio", asyncio.run(aio_once())),
                           ("sync", g.InferenceServerClient("localhost:8001").infer(
                               "prod", [_inp(g, frames[:1])], outputs=[g.InferRequestedOutput(n) for n in OUTPUTS]))):
            resp = res.get_response()
            print(label, [(o.name, list(o.shape), o.datatype) for o in resp.outputs],
                  [len(b) for b in resp.raw_output_contents])
            print(label, {k: v.shape for k, v in decode_response(res).items()})
        s = shadow_compare(TritonEndpoint("prod"), TritonEndpoint("cand"), frames)
        print(shadow_markdown(s))
        assert s["agreement"]["ref_recall"] == 1.0, "identical models must agree exactly"
    finally:
        server.stop()
    print("smoke OK")


def _inp(g, frames):
    i = g.InferInput("frames", list(frames.shape), "UINT8")
    i.set_data_from_numpy(frames)
    return i


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m detect2deploy.platform", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/deploy.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("export")
    p.add_argument("--out", default=BUILD)
    p.add_argument("--variants", nargs="+", default=["fp32", "fp16_mixed", "int8_trt"])
    p.add_argument("--from-registry", default=None, metavar="ALIAS",
                   help="also export the model a registry alias points to")
    p.add_argument("--uri", default="sqlite:///mlruns/registry.db")

    p = sub.add_parser("serve")
    p.add_argument("--model", action="append", required=True, help="name=path (repeatable)")
    p.add_argument("--provider", default="tensorrt")
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-delay-us", type=int, default=500)
    p.add_argument("--preferred-batch", type=int, nargs="*", default=None)
    p.add_argument("--uri", default="sqlite:///mlruns/registry.db", help="registry for registry:<alias> models")

    p = sub.add_parser("serving-bench")
    p.add_argument("--model", help="serving graph (pytriton backend)")
    p.add_argument("--provider", default="tensorrt")
    p.add_argument("--backend", choices=["pytriton", "triton"], default="pytriton")
    p.add_argument("--out", required=True)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    p.add_argument("--url", default="localhost:8001")

    p = sub.add_parser("corpus")
    p.add_argument("--out", default="data/corpus")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--shard-size", type=int, default=500)
    p.add_argument("--quality", type=int, default=95)
    p.add_argument("--info", default=None, help="write corpus stats JSON here")

    p = sub.add_parser("batch")
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--provider", default="tensorrt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--actors", type=int, default=1)
    p.add_argument("--labeler-cpus", type=float, default=1.0)
    p.add_argument("--usd-per-hour", type=float, default=None)
    p.add_argument("--eval-frames", type=int, default=2000)
    p.add_argument("--tag", default="")

    p = sub.add_parser("registry")
    p.add_argument("action", choices=["register", "promote", "resolve", "list"])
    p.add_argument("--uri", default="sqlite:///mlruns/registry.db")
    p.add_argument("--model")
    p.add_argument("--target")
    p.add_argument("--gate")
    p.add_argument("--parity")
    p.add_argument("--version")
    p.add_argument("--alias", default="production")
    p.add_argument("--json", default=None, help="list: also write the table here")
    p.add_argument("--event", action="append", help="list: notes to record alongside")

    p = sub.add_parser("shadow")
    p.add_argument("--production", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--out", required=True)
    p.add_argument("--url", default="localhost:8001")

    p = sub.add_parser("report")
    p.add_argument("--dir", required=True)

    p = sub.add_parser("smoke")
    p.add_argument("--out", default="build/platform_smoke")

    args = ap.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    globals()[f"cmd_{args.cmd.replace('-', '_')}"](args)


if __name__ == "__main__":
    code = 0
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except KeyboardInterrupt:
        code = 0                    # `serve` is stopped with SIGINT
    except Exception:
        traceback.print_exc()
        code = 1
    # Same reason as detect2deploy.deploy: skip ORT's occasionally crashing native teardown
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
