"""
python -m nurosim.deploy <command>

  dataset   write synthetic scenarios as a YOLO dataset
  train     train YOLOv8n on it (MPS / CUDA / CPU)
  export    checkpoint → FP32 ONNX
  quantize  FP32 ONNX → FP16 / INT8, each full and with the head tail in FP32
  parity    accuracy + output parity of every target vs the reference
  bench     latency benchmark of every target
  gate      check reports against budgets; non-zero exit on failure
  all       quantize → parity → bench → gate
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from nurosim.deploy.config import DEFAULT_CONFIG, load_config, load_profile

logger = logging.getLogger("nurosim.deploy")


def _profile(args, cfg):
    only = args.targets.split(",") if getattr(args, "targets", None) else None
    return load_profile(cfg, args.profile, only)


def cmd_dataset(args, cfg):
    from nurosim.deploy.dataset import build_yolo_dataset
    print(build_yolo_dataset(args.out, args.n_train, args.n_val))


def cmd_train(args, cfg):
    from nurosim.deploy.train import train_detector
    train_detector(args.data, out_path=args.out, epochs=args.epochs, batch=args.batch,
                   device=args.device)


def cmd_export(args, cfg):
    from nurosim.deploy.export import export_onnx
    export_onnx(args.weights, cfg["models"]["fp32"], imgsz=cfg["imgsz"])


def cmd_quantize(args, cfg):
    from nurosim.deploy.quantize import to_fp16, to_int8
    m, q = cfg["models"], cfg["quantization"]
    fp32 = m["fp32"]
    if not os.path.exists(fp32):
        sys.exit(f"{fp32} not found — run `python -m nurosim.deploy export` first")
    common = dict(n_calib=q["calib_frames"], method=q["method"],
                  per_channel=q["per_channel"],
                  symmetric_activations=q["symmetric_activations"], imgsz=cfg["imgsz"])
    to_fp16(fp32, m["fp16"])
    to_fp16(fp32, m["fp16_mixed"], keep_head_tail_fp32=True)
    to_int8(fp32, m["int8"], keep_head_tail_fp32=True, **common)
    to_int8(fp32, m["int8_full"], keep_head_tail_fp32=False, **common)


def cmd_parity(args, cfg):
    from nurosim.deploy.parity import run_parity, write_parity, parity_markdown
    report = run_parity(_profile(args, cfg), cfg)
    path = write_parity(report, args.reports)
    print(parity_markdown(report))
    logger.info("Parity report → %s", path)


def cmd_bench(args, cfg):
    from nurosim.deploy.benchmark import run_benchmark, write_benchmark, benchmark_markdown
    report = run_benchmark(_profile(args, cfg), cfg)
    path = write_benchmark(report, args.reports)
    print(benchmark_markdown(report))
    logger.info("Benchmark report → %s", path)


def cmd_gate(args, cfg):
    from nurosim.deploy.gate import gate_markdown, load_reports, run_gate
    profile = _profile(args, cfg)
    parity, bench = load_reports(profile.name, args.reports)
    checks = run_gate(cfg, profile, parity, bench)
    md = gate_markdown(profile, checks)
    with open(os.path.join(args.reports, f"gate_{profile.name}.md"), "w") as f:
        f.write(md)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(md + "\n")
    print(md)
    failed = [c for c in checks if not c.passed]
    if failed:
        logger.error("Gate FAILED: %d of %d checks", len(failed), len(checks))
        sys.exit(1)
    logger.info("Gate passed: %d checks", len(checks))


def cmd_all(args, cfg):
    cmd_quantize(args, cfg)
    cmd_parity(args, cfg)
    cmd_bench(args, cfg)
    cmd_gate(args, cfg)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nurosim.deploy", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--reports", default="reports")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dataset")
    p.add_argument("--out", default="data/nurosim_yolo")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-val", type=int, default=300)

    p = sub.add_parser("train")
    p.add_argument("--data", default="data/nurosim_yolo/data.yaml")
    p.add_argument("--out", default="models/nurosim_det.pt")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)

    p = sub.add_parser("export")
    p.add_argument("--weights", default="models/nurosim_det.pt")

    sub.add_parser("quantize")

    for name in ("parity", "bench", "gate", "all"):
        p = sub.add_parser(name)
        p.add_argument("--profile", required=True)
        p.add_argument("--targets", default=None,
                       help="comma-separated subset (reference is always included)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                        datefmt="%H:%M:%S")
    os.makedirs(args.reports, exist_ok=True)
    cfg = load_config(args.config)
    globals()[f"cmd_{args.cmd}"](args, cfg)


if __name__ == "__main__":
    main()
