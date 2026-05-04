"""
ray_worker.py
Distributed evaluation pipeline using Ray for parallelism.

Architecture mirrors how AV companies scale sim eval runs across a compute cluster:
  - ScenarioBatch  : unit of work (a list of scenario IDs + configs)
  - EvalWorker     : Ray remote actor that owns a Detector and Evaluator
  - ParallelEvaluator : orchestrates workers, collects partial results, merges

Falls back gracefully to multiprocessing.Pool when Ray is not installed.
"""

from __future__ import annotations

import os
import time
import math
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np

from nurosim.scenario_generator import generate_scenario, Scenario
from nurosim.perception_model   import MockDetector, TimedDetector
from nurosim.metrics            import Evaluator, EvalResult

logger = logging.getLogger(__name__)

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


# ── Ray Actor ─────────────────────────────────────────────────────────────────

def _run_batch_local(cfg: BatchConfig) -> BatchResult:
    """
    Process one batch locally (used both by Ray actor and fallback pool).
    Generates scenarios on-the-fly — no serialisation of large frame arrays.
    """
    detector = TimedDetector(MockDetector(seed=cfg.batch_id))
    ev       = Evaluator(iou_threshold=cfg.iou_thr)

    t0 = time.perf_counter()
    for sid in range(cfg.start_id, cfg.end_id):
        scenario = generate_scenario(sid, seed=cfg.base_seed + sid)
        preds    = detector.predict(scenario.frame,
                                    ground_truth=scenario.ground_truth)
        ev.add(scenario, preds)

    wall = time.perf_counter() - t0

    return BatchResult(
        batch_id=cfg.batch_id,
        records=ev._records,          # lightweight dicts — no frame arrays
        latencies=detector.latencies,
        wall_time_s=wall,
        n_scenarios=cfg.end_id - cfg.start_id,
    )


# ── Parallel Evaluator ────────────────────────────────────────────────────────

class ParallelEvaluator:
    """
    Runs evaluation across N scenarios using either:
      - Ray  (if installed and use_ray=True)
      - multiprocessing.Pool  (fallback)

    Example
    -------
        pe = ParallelEvaluator(n_scenarios=500, n_workers=8)
        result, stats = pe.run()
        print(result.summary())
    """

    def __init__(self,
                 n_scenarios: int   = 500,
                 n_workers:   int   = 4,
                 iou_thr:     float = 0.5,
                 base_seed:   int   = 42,
                 use_ray:     bool  = True):
        self.n_scenarios = n_scenarios
        self.n_workers   = n_workers
        self.iou_thr     = iou_thr
        self.base_seed   = base_seed
        self.use_ray     = use_ray

    def _build_batch_configs(self) -> List[BatchConfig]:
        batch_size = math.ceil(self.n_scenarios / self.n_workers)
        configs    = []
        for i in range(self.n_workers):
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
        for br in batch_results:
            master_ev._records.extend(br.records)
        return master_ev.compute()

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
                import ray

                if not ray.is_initialized():
                    ray.init(ignore_reinit_error=True,
                             num_cpus=self.n_workers,
                             log_to_driver=False)

                @ray.remote
                def _ray_task(cfg: BatchConfig) -> BatchResult:
                    return _run_batch_local(cfg)

                futures      = [_ray_task.remote(c) for c in configs]
                batch_results = ray.get(futures)
                ray_used      = True
                logger.info("Ray backend used with %d workers.", self.n_workers)

            except ImportError:
                logger.warning("Ray not installed — falling back to multiprocessing.")
            except Exception as e:
                logger.warning("Ray failed (%s) — falling back to multiprocessing.", e)

        if not ray_used:
            # Fallback: multiprocessing.Pool
            from multiprocessing import Pool
            with Pool(processes=self.n_workers) as pool:
                batch_results = pool.map(_run_batch_local, configs)
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
            "n_workers":        len(configs),
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
