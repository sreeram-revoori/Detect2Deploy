"""
seeds.py
Disjoint seed ranges for every data split. Train, val, calibration and eval
frames never overlap, so parity / accuracy numbers are on held-out scenarios
and INT8 calibration never sees the frames it is later scored on.
"""

TRAIN_SEED = 0            # scenarios      0 … n_train-1
VAL_SEED   = 50_000       # scenarios 50_000 … (ultralytics val split)
EVAL_SEED  = 100_000      # held-out parity / accuracy set
CALIB_SEED = 200_000      # INT8 calibration frames
BENCH_SEED = 300_000      # latency benchmark frames
CORPUS_SEED = 400_000     # offboard batch-inference corpus
