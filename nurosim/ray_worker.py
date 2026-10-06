"""
ray_worker.py
Distributed evaluation pipeline using Ray for parallelism.

Architecture mirrors how AV companies scale sim eval runs across a compute cluster:
  - BatchConfig       : unit of work (a contiguous range of scenario IDs)
  - EvalWorker        : long-lived Ray actor that loads the detector ONCE and
                        then processes many batches (model load is the
                        expensive part for real models — never do it per task)
  - ParallelEvaluator : shards scenarios into more batches than workers,
                        dispatches them through an ActorPool (dynamic load
                        balancing), merges the partial results

Falls back to multiprocessing.Pool (one detector per process via the pool
initializer) when Ray is not installed.
"""

from __future__ import annotations

import time
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np

from nurosim.scenario_generator import generate_scenario
from nurosim.perception_model   import TimedDetector, build_detector
from nurosim.metrics            import Evaluator, EvalResult

logger = logging.getLogger(__name__)

BATCHES_PER_WORKER = 4      # oversubscribe so slow batches don't stall the run

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class BatchConfig:
    batch_id:    int
    start_id:    int
    end_id:      int          # exclusive
    base_seed:   int = 42
    iou_thr:     float = 0.5


@dataclass
class BatchResult:
    batch_id:    int
    records:     list         # raw _records from Evaluator (reused for merging)
    latencies:   List[float]
    wall_time_s: float
    n_scenarios: int


# ── Worker-side logic (shared by Ray actors and the process pool) ────────────

class _DetectorCache:
    """Holds one detector per worker. Mock detectors are re-seeded per batch so
    results don't depend on which worker happened to pick up which batch."""

    def __init__(self, spec: Optional[Dict[str, Any]]):
        self.spec = spec or {"kind": "mock"}
        self._det = None

    def get(self, cfg: BatchConfig):
        if self.spec.get("kind", "mock") == "mock":
            return build_detector(self.spec, seed=cfg.batch_id)
        if self._det is None:
            self._det = build_detector(self.spec)
        return self._det


def _run_batch(cache: _DetectorCache, cfg: BatchConfig) -> BatchResult:
    """Generates scenarios on-the-fly — no serialisation of large frame arrays."""
    detector = TimedDetector(cache.get(cfg))
    ev       = Evaluator(iou_threshold=cfg.iou_thr)

    t0 = time.perf_counter()
    for sid in range(cfg.start_id, cfg.end_id):
        scenario = generate_scenario(sid, seed=cfg.base_seed + sid)
        preds    = detector.predict(scenario.frame,
                                    ground_truth=scenario.ground_truth)
        ev.add(scenario, preds)

    return BatchResult(
        batch_id=cfg.batch_id,
        records=ev._records,          # lightweight dicts — no frame arrays
        latencies=detector.latencies,
        wall_time_s=time.perf_counter() - t0,
        n_scenarios=cfg.end_id - cfg.start_id,
    )


def _run_batch_local(cfg: BatchConfig,
                     detector_spec: Optional[Dict[str, Any]] = None) -> BatchResult:
    """Process one batch with a freshly built detector (single-shot helper)."""
    return _run_batch(_DetectorCache(detector_spec), cfg)


# multiprocessing.Pool: one cache per process, built by the pool initializer
_PROCESS_CACHE: Optional[_DetectorCache] = None


def _pool_init(detector_spec: Optional[Dict[str, Any]]) -> None:
    global _PROCESS_CACHE
    _PROCESS_CACHE = _DetectorCache(detector_spec)


def _pool_task(cfg: BatchConfig) -> BatchResult:
    assert _PROCESS_CACHE is not None
    return _run_batch(_PROCESS_CACHE, cfg)


# ── Parallel Evaluator ────────────────────────────────────────────────────────

class ParallelEvaluator:
    """
    Runs evaluation across N scenarios using either:
      - Ray actors  (if installed and use_ray=True)
      - multiprocessing.Pool  (fallback)

    Example
    -------
        pe = ParallelEvaluator(n_scenarios=500, n_workers=8)
        result, stats = pe.run()

        # real model, one session per worker
        spec = {"kind": "onnx", "model_path": "models/nurosim_det_int8.onnx",
                "provider": "cpu", "intra_op_threads": 2}
        pe = ParallelEvaluator(n_scenarios=500, n_workers=4, detector_spec=spec)
    """

    def __init__(self,
                 n_scenarios: int   = 500,
                 n_workers:   int   = 4,
                 iou_thr:     float = 0.5,
                 base_seed:   int   = 42,
                 use_ray:     bool  = True,
                 detector_spec: Optional[Dict[str, Any]] = None,
                 gpus_per_worker: float = 0.0):
        self.n_scenarios     = n_scenarios
        self.n_workers       = n_workers
        self.iou_thr         = iou_thr
        self.base_seed       = base_seed
        self.use_ray         = use_ray
        self.detector_spec   = detector_spec
        self.gpus_per_worker = gpus_per_worker

    def _build_batch_configs(self) -> List[BatchConfig]:
        n_batches  = max(1, self.n_workers * BATCHES_PER_WORKER)
        batch_size = math.ceil(self.n_scenarios / n_batches)
        configs    = []
        for i in range(n_batches):
            start = i * batch_size
            end   = min(start + batch_size, self.n_scenarios)
            if start >= end:
                break
            configs.append(BatchConfig(
                batch_id=i,
                start_id=start,
                end_id=end,
                base_seed=self.base_seed,
                iou_thr=self.iou_thr,
            ))
        return configs

    def _merge_results(self, batch_results: List[BatchResult]) -> EvalResult:
        """Merge partial Evaluator records from all workers into one result."""
        master_ev = Evaluator(iou_threshold=self.iou_thr)
        for br in sorted(batch_results, key=lambda b: b.batch_id):
            master_ev._records.extend(br.records)
        return master_ev.compute()

    def _run_ray(self, configs: List[BatchConfig]) -> List[BatchResult]:
        import ray
        from ray.util import ActorPool

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True,
                     num_cpus=self.n_workers,
                     log_to_driver=False)

        @ray.remote
        class EvalWorker:
            def __init__(self, spec):
                self.cache = _DetectorCache(spec)       # model loaded once here

            def run(self, cfg: BatchConfig) -> BatchResult:
                return _run_batch(self.cache, cfg)

        n_actors = min(self.n_workers, len(configs))
        actors = [EvalWorker.options(num_cpus=1, num_gpus=self.gpus_per_worker)
                  .remote(self.detector_spec) for _ in range(n_actors)]
        pool = ActorPool(actors)
        return list(pool.map_unordered(lambda a, c: a.run.remote(c), configs))

    def run(self) -> tuple[EvalResult, Dict[str, Any]]:
        """
        Returns (EvalResult, stats_dict).
        stats_dict contains wall time, throughput, latency percentiles.
        """
        configs     = self._build_batch_configs()
        t_start     = time.perf_counter()
        batch_results: List[BatchResult] = []

        ray_used = False
        if self.use_ray:
            try:
                batch_results = self._run_ray(configs)
                ray_used      = True
                logger.info("Ray backend used with %d actors.", self.n_workers)
            except ImportError:
                logger.warning("Ray not installed — falling back to multiprocessing.")
            except Exception as e:
                logger.warning("Ray failed (%s) — falling back to multiprocessing.", e)

        if not ray_used:
            from multiprocessing import Pool
            with Pool(processes=self.n_workers, initializer=_pool_init,
                      initargs=(self.detector_spec,)) as pool:
                batch_results = pool.map(_pool_task, configs)
            logger.info("multiprocessing.Pool backend used with %d workers.",
                        self.n_workers)

        total_wall = time.perf_counter() - t_start

        # Aggregate latency stats across all workers
        all_latencies = []
        for br in batch_results:
            all_latencies.extend(br.latencies)
        lat_arr = np.array(all_latencies) if all_latencies else np.array([0.])

        stats = {
            "backend":          "ray" if ray_used else "multiprocessing",
            "n_workers":        min(self.n_workers, len(configs)),
            "n_batches":        len(configs),
            "n_scenarios":      self.n_scenarios,
            "total_wall_s":     round(total_wall, 3),
            "throughput_fps":   round(self.n_scenarios / total_wall, 1),
            "lat_mean_ms":      round(float(np.mean(lat_arr)), 3),
            "lat_p50_ms":       round(float(np.percentile(lat_arr, 50)), 3),
            "lat_p95_ms":       round(float(np.percentile(lat_arr, 95)), 3),
            "lat_p99_ms":       round(float(np.percentile(lat_arr, 99)), 3),
        }

        result = self._merge_results(batch_results)
        return result, stats
