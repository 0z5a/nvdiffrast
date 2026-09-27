# Threaded rasterizer and optimization follow-up

Source: NVlabs/nvdiffrast `253ac4fcea7de5f396371124af597e6cc957bfae`.
Two RTX 5090 GPUs, driver 580.82.07, PyTorch 2.13.0+cu130,
CUDA toolkit 13.0.88, Python 3.12.13. The extension was built in place;
no packages or environments were changed.

## Issue #226 conditions

`threaded_context_probe.py` creates a persistent worker thread and CUDA
rasterizer context for each device. Dispatch uses threading events.
DataParallel uses two devices, batch size four, and the worker contexts
render/interpolate a 64×64 triangle. Every iteration checks all images
against the independent 968-pixel coverage result using unit attributes.
Both normal logging (1) and info logging (0) were exercised.

The harness adds CUDA events for producer/worker/consumer dependencies and
a normal worker shutdown sentinel. It uses an already compiled extension.
Consequently it tests the reported threading arrangement on current main;
it does not reproduce a historical JIT compilation or identical original
application. There is no forced garbage collection inside the workload.

| Mode | Log level | Work | Fresh processes | Result |
|---|---:|---|---:|---|
| Single worker context idle | 0 | 120 seconds | 1 | Completed; no deadlock |
| Single worker render/interpolate | 0 and 1 | 1,000 iterations each | 2 | All outputs passed |
| Two-device DataParallel | 0 | 1,000 iterations | 3 | All outputs passed |
| Two-device DataParallel | 1 | 1,000 iterations | 1 | All outputs passed |
| Two-device DataParallel | 0 | 5,000 iterations | 1 | All outputs passed |

## Resource measurements

RSS/USS are Linux process measurements. GPU counters below cover the
PyTorch allocator only. USS can change as sharing with other processes
changes; this shared host had unrelated workloads on other GPUs.

| Mode and window | RSS change | USS change | Threads / FDs |
|---|---:|---:|---|
| Worker idle, ready → 120 s | +36 KiB | +19,948 KiB | Unchanged |
| Single worker, log 0, 250 → 1,000 | +232 KiB | +168 KiB | Unchanged |
| Single worker, log 1, 250 → 1,000 | +32 KiB | −46,848 KiB | Unchanged |
| DataParallel, log 0, 250 → 1,000, median of 3 | +388 KiB | +292 KiB | Unchanged |
| DataParallel, log 1, 250 → 1,000 | +328 KiB | +172 KiB | Unchanged |
| DataParallel, log 0, 1,250 → 5,000 | +236 KiB | −14,452 KiB | Unchanged |

In the 5,000-iteration run, the GC-tracked object count stayed at 253,244
from iteration 1,250 onward. PyTorch reserved memory stayed at 2 MiB/device;
live allocated memory stayed at 1,024 bytes on device 0 and zero on device 1.
The worker threads exited normally, restoring the pre-worker thread count.
Native info logs showed each rasterizer initially growing its buffers to
10 MB. The measurements show a plateau in these windows, not proof against
every host or device leak. The original issue's precise source/environment
version remains unknown.

## Official cube optimization E2E

`cube_e2e.py` calls the unmodified upstream `samples/torch/cube.py`
optimization loop. Each case executes iterations 0 through 1,000 at 32×32,
including rasterization, interpolation, antialiasing, backward, Adam, and
the learning-rate scheduler. NumPy/PyTorch seed: 20260927.

| Case | Initial geometric error | Final logged interval error | Final / initial | Status |
|---|---:|---:|---:|---|
| Continuous colors | 0.489277 | 0.003337 | 0.00682 | Passed |
| Discontinuous colors | 0.489277 | 0.000614 | 0.00125 | Passed |

The executable check requires finite logged errors and a final error below
5% of the initial error. The final log averages the last logging interval,
rather than representing a single-step pixel loss. An earlier same-seed run
also converged (0.000029 and 0.000880 respectively); fixed seeds did not make
the full GPU optimization trajectory identical. The validated runs took
8.20 s and 5.21 s in one process, with cold/warm state differing. They are
completion timings, not a speed comparison.

## Reproduction

From the built repository root, using the existing Python environment:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PWD python diagnostics/threaded_context_probe.py --mode parallel --log-level 0 --iterations 5000
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python diagnostics/threaded_context_probe.py --mode idle --log-level 0 --seconds 120
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python diagnostics/cube_e2e.py
```

`threaded_results.json` contains the measured samples, cube results, and
source/native hashes. This remains a fork-only diagnostic contribution.
No renderer implementation changed and no speedup is claimed.
