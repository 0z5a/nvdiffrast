import json
import os
import sys
import time
from pathlib import Path

case = sys.argv[1]
run = int(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else (3 if case in ("C0", "C1", "C2") else 200)
start = time.perf_counter()

def proc_memory():
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
            key, value = line.split(":", 1)
            fields[key] = int(value.split()[0])
    private_kb = 0
    for line in Path("/proc/self/smaps_rollup").read_text().splitlines():
        if line.startswith(("Private_Clean:", "Private_Dirty:")):
            private_kb += int(line.split()[1])
    fields["USS_kB"] = private_kb
    fields["FDs"] = len(list(Path("/proc/self/fd").iterdir()))
    return fields

def sample(stage, iteration, torch_module=None):
    if torch_module is not None:
        torch_module.cuda.synchronize()
    record = {
        "case": case, "run": run, "stage": stage, "iteration": iteration,
        "elapsed_s": round(time.perf_counter() - start, 4),
        **proc_memory(),
        "torch_allocated_bytes": None if torch_module is None else torch_module.cuda.memory_allocated(),
        "torch_reserved_bytes": None if torch_module is None else torch_module.cuda.memory_reserved(),
    }
    print(json.dumps(record), flush=True)

sample("start", 0)
if case == "C0":
    for i in range(1, limit + 1):
        time.sleep(1)
        sample("idle", i)
    raise SystemExit(0)

import torch
import nvdiffrast.torch as dr

torch.cuda.init()
sample("cuda_ready", 0, torch)
if case == "C1":
    for i in range(1, limit + 1):
        time.sleep(1)
        sample("idle", i, torch)
    raise SystemExit(0)

ctx = dr.RasterizeCudaContext(device="cuda:0")
sample("context_ready", 0, torch)
if case == "C2":
    for i in range(1, limit + 1):
        time.sleep(1)
        sample("idle", i, torch)
    raise SystemExit(0)

if case == "C3":
    del ctx
    for i in range(1, limit + 1):
        ctx = dr.RasterizeCudaContext(device="cuda:0")
        del ctx
        if i in (limit // 4, limit // 2, limit):
            sample("steady", i, torch)
    raise SystemExit(0)

pos = torch.tensor([[[-0.7, -0.7, 0.0, 1.0],
                     [0.7, -0.7, 0.0, 1.0],
                     [0.0, 0.7, 0.0, 1.0]]],
                   device="cuda:0", dtype=torch.float32, requires_grad=case == "C5")
tri = torch.tensor([[0, 1, 2]], device="cuda:0", dtype=torch.int32)
attr = torch.tensor([[[0.0], [1.0], [0.5]]],
                    device="cuda:0", dtype=torch.float32, requires_grad=case == "C5")
for i in range(1, limit + 1):
    rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
    if case == "C5":
        interp, _ = dr.interpolate(attr, rast, tri)
        loss = interp[..., 0].square().mean()
        loss.backward()
        pos.grad.zero_()
        attr.grad.zero_()
    if i in (limit // 4, limit // 2, limit):
        sample("steady", i, torch)
