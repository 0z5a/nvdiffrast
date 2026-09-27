# CUDA rasterizer diagnostic evidence

Pinned source: NVlabs/nvdiffrast `253ac4fcea7de5f396371124af597e6cc957bfae`.
Built in place with PyTorch 2.13.0+cu130, CUDA toolkit 13.0, Python 3.12.13,
driver 580.82.07, and one RTX 5090 (SM120). The loaded native extension came
from this source tree. Current `RasterizeGLContext` inherits
`RasterizeCudaContext` and is a deprecated compatibility entry, so it was
not treated as a distinct OpenGL backend.

Run from the repository root after building the extension in place:

```bash
PYTHONPATH=$PWD python diagnostics/smoke.py
PYTHONPATH=$PWD python diagnostics/numeric_reference.py
PYTHONPATH=$PWD python diagnostics/context_memory_probe.py C2 1 30
PYTHONPATH=$PWD python diagnostics/context_memory_probe.py C3 1 1000
PYTHONPATH=$PWD python diagnostics/context_memory_probe.py C4 1 1000
PYTHONPATH=$PWD python diagnostics/context_memory_probe.py C5 1 1000
```

## Host memory on current main

Each case ran in a fresh process. C0 is sampler-only idle; C1 initializes
PyTorch CUDA then idles; C2 creates one context then idles; C3 repeatedly
creates and releases contexts; C4 reuses one context for fixed 64×64
rasterization; C5 adds interpolation and backward. Samples synchronize the
GPU only at checkpoints. RSS and USS are from Linux `/proc`; GPU allocated
and reserved are PyTorch allocator counters and do not include every native
allocation.

| Case and window | Fresh runs | Median RSS change | Median USS change | Interpretation |
|---|---:|---:|---:|---|
| C0, 30 s idle | 1 | 0 KiB | 0 KiB | Sampler stayed flat |
| C1, 30 s idle | 1 | +480 KiB | +356 KiB | CUDA idle control |
| C2, 30 s context idle repeat | 1 | +592 KiB | +360 KiB | No sustained RSS growth in this window |
| C3, iterations 250→1000 | 3 | +220 KiB | +24 KiB | Creation/release stayed near plateau |
| C4, iterations 250→1000 | 3 | +92 KiB | +84 KiB | Fixed-shape rasterization stayed near plateau |
| C5, iterations 250→1000 | 3 | +152 KiB | +196 KiB | Interpolation/backward grew slightly in short window |

In one earlier C2 30-second process, USS rose about 136 MiB while RSS rose
about 2 MiB; a fresh C2 repeat had only +360 KiB USS. USS can change when
shared pages become private as other processes exit, so this single USS
transition does not establish an allocation leak. A separate C5 run through
5000 iterations had RSS 778372→778556 KiB and USS 538224→533692 KiB from
iteration 1250 to 5000; PyTorch reserved memory remained 2 MiB. These
results narrow #226 on current main and these conditions. They do not
reproduce its threaded/DataParallel setup or higher log level, and do not
settle historical-version behavior.

The current native wrapper's destructor deletes its rasterizer
(`csrc/torch/torch_rasterize.cpp`); `Buffer` destructors call
`cudaFree`/`cudaFreeHost`. This is an ownership map, not proof that every
runtime reference was released in the reported training setup.

## Function and gradient checks

A single triangle at 64×64 covered 968 pixels. The sum of vertex-attribute
gradients was 967.99994, versus 968 expected from barycentric weights.
At pixel (32,32), affine interpolation was 0.511160731; the independent
pixel-center barycentric calculation gives 0.511160714. The position
gradient was -0.182557; centered finite differences were -0.182509,
-0.182559, and -0.182566 for steps 0.001, 0.003, and 0.01.
This validates the listed CUDA rasterize/interpolate path, not every
renderer operation.

Inside an active `DepthPeeler`, ordinary `rasterize()` returned a
`RuntimeError` object instead of raising it. The context became usable
again after leaving the `with` block. This is a separate API error-path
finding; no link to #226's memory report is established.

There is no speedup comparison because this is a diagnostic contribution,
not an execution-path optimization. The repository currently states that
outside code pull requests are not accepted; this evidence is staged in a
fork for review before any issue follow-up.
