"""
corpus.py
An offboard "drive log" to label: N scenario frames stored as JPEG in
Parquet shards (scenario_id, weather, jpeg) — the shape of data a batch
inference job reads from object storage. JPEG rather than raw (1.2 MB/frame)
or PNG (the rain noise defeats PNG compression); decode is then real CPU work
in the pipeline, as it is in production.
"""

from __future__ import annotations

import logging
import os
from multiprocessing import Pool
from typing import Tuple

import cv2

from detect2deploy.deploy.seeds import CORPUS_SEED
from detect2deploy.scenario_generator import generate_scenario

logger = logging.getLogger(__name__)


def _write_shard(args: Tuple[str, int, int, int]) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path, start, end, quality = args
    ids, weathers, blobs = [], [], []
    for sid in range(start, end):
        sc = generate_scenario(sid, seed=sid)
        ok, buf = cv2.imencode(".jpg", sc.frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert ok
        ids.append(sid)
        weathers.append(sc.weather)
        blobs.append(buf.tobytes())
    pq.write_table(pa.table({"scenario_id": pa.array(ids, pa.int64()),
                             "weather": pa.array(weathers, pa.string()),
                             "jpeg": pa.array(blobs, pa.binary())}), path)
    return sum(len(b) for b in blobs)


def write_corpus(out_dir: str, n: int = 20_000, shard_size: int = 500, quality: int = 95,
                 workers: int | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(os.path.join(out_dir, f"shard_{i:05d}.parquet"), CORPUS_SEED + s,
             CORPUS_SEED + min(s + shard_size, n), quality)
            for i, s in enumerate(range(0, n, shard_size))]
    if workers == 0:                     # in-process (tests, notebooks)
        sizes = [_write_shard(j) for j in jobs]
    else:
        with Pool(workers or os.cpu_count()) as pool:
            sizes = pool.map(_write_shard, jobs)
    info = {"frames": n, "shards": len(jobs), "bytes": int(sum(sizes)),
            "avg_kb_per_frame": sum(sizes) / n / 1024, "jpeg_quality": quality}
    logger.info("corpus: %d frames in %d shards, %.0f KB/frame avg", n, len(jobs), info["avg_kb_per_frame"])
    return info
