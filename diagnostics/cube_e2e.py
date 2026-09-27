"""Run the upstream cube optimization path with a fixed seed and convergence check."""
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "samples" / "torch"))
import cube

for discontinuous in (False, True):
    np.random.seed(20260927)
    torch.manual_seed(20260927)
    log = io.StringIO()
    start = time.perf_counter()
    with contextlib.redirect_stdout(log):
        cube.fit_cube(max_iter=1000, resolution=32, discontinuous=discontinuous, log_interval=100)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(log.getvalue(), end="", flush=True)
    errors = [float(line.split("err=")[1]) for line in log.getvalue().splitlines()
              if line.startswith("iter=")]
    assert all(math.isfinite(error) for error in errors)
    assert errors[-1] < errors[0] * 0.05
    print(json.dumps(dict(case="discontinuous" if discontinuous else "continuous",
        seed=20260927, iterations=1001, resolution=32,
        initial_geometric_error=errors[0], final_interval_geometric_error=errors[-1],
        elapsed_s=elapsed, result="PASS")), flush=True)
