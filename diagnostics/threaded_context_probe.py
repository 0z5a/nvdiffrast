import argparse
import gc
import json
import threading
import time
from pathlib import Path

import torch
import nvdiffrast.torch as dr

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("idle", "render", "parallel"), required=True)
parser.add_argument("--log-level", type=int, default=1)
parser.add_argument("--iterations", type=int, default=1000)
parser.add_argument("--seconds", type=int, default=120)
args = parser.parse_args()
dr.set_log_level(args.log_level)
torch.set_num_threads(1)
start = time.perf_counter()
devices = [0, 1] if args.mode == "parallel" else [0]


def sample(stage, iteration):
    for device in devices:
        torch.cuda.synchronize(device)
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    uss = sum(int(line.split()[1]) for line in Path("/proc/self/smaps_rollup").read_text().splitlines()
              if line.startswith(("Private_Clean:", "Private_Dirty:")))
    print(json.dumps(dict(stage=stage, iteration=iteration, elapsed_s=time.perf_counter()-start,
        rss_kib=int(status["VmRSS"].split()[0]), uss_kib=uss,
        threads=int(status["Threads"]), fds=len(list(Path("/proc/self/fd").iterdir())),
        gc_objects=len(gc.get_objects()),
        allocated=[torch.cuda.memory_allocated(d) for d in devices],
        reserved=[torch.cuda.memory_reserved(d) for d in devices])), flush=True)


class Dispatcher:
    def __init__(self):
        self.events = {d: threading.Event() for d in devices}
        self.done = {d: threading.Event() for d in devices}
        self.funcs = {}
        self.values = {}
        self.threads = [threading.Thread(target=self.worker, args=(d,)) for d in devices]
        for thread in self.threads:
            thread.start()
        for event in self.done.values():
            event.wait()
            event.clear()

    def worker(self, device):
        torch.cuda.set_device(device)
        print(json.dumps(dict(stage="context_enter", device=device)), flush=True)
        ctx = dr.RasterizeCudaContext(device=device)
        print(json.dumps(dict(stage="context_ready", device=device)), flush=True)
        self.done[device].set()
        while True:
            self.events[device].wait()
            self.events[device].clear()
            func = self.funcs.pop(device)
            if func is None:
                break
            self.values[device] = func(ctx)
            del func
            self.done[device].set()

    def __call__(self, device, func):
        self.funcs[device] = func
        self.events[device].set()
        self.done[device].wait()
        self.done[device].clear()
        return self.values.pop(device)

    def close(self):
        for device in devices:
            self.funcs[device] = None
            self.events[device].set()
        for thread in self.threads:
            thread.join()


class Rasterizer(torch.nn.Module):
    def __init__(self, dispatcher):
        super().__init__()
        self.dispatcher = dispatcher

    def forward(self, pos, attr):
        device = pos.device.index
        ready = torch.cuda.Event()
        ready.record()
        def render(ctx):
            torch.cuda.current_stream(device).wait_event(ready)
            tri = torch.tensor([[0, 1, 2]], device=pos.device, dtype=torch.int32)
            rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
            color, _ = dr.interpolate(attr, rast, tri)
            finished = torch.cuda.Event()
            finished.record()
            return color, finished
        color, finished = self.dispatcher(device, render)
        torch.cuda.current_stream(device).wait_event(finished)
        return color


sample("cuda_ready", 0)
dispatcher = Dispatcher()
sample("threads_ready", 0)
if args.mode == "idle":
    for i in range(1, args.seconds + 1):
        time.sleep(1)
        if i % 30 == 0 or i == args.seconds:
            sample("idle", i)
else:
    batch = 4 if args.mode == "parallel" else 1
    pos = torch.tensor([[[-0.7, -0.7, 0., 1.], [0.7, -0.7, 0., 1.], [0., 0.7, 0., 1.]]],
                       device="cuda:0").repeat(batch, 1, 1)
    attr = torch.ones(batch, 3, 1, device="cuda:0")
    model = Rasterizer(dispatcher)
    if args.mode == "parallel":
        model = torch.nn.DataParallel(model, device_ids=devices)
    for i in range(1, args.iterations + 1):
        color = model(pos, attr)
        # Unit attributes integrate to the independently established coverage.
        torch.testing.assert_close(color.sum((1, 2, 3)),
                                   torch.full((batch,), 968., device="cuda:0"), atol=1e-3, rtol=0)
        del color
        if i in (10, args.iterations // 4, args.iterations // 2, args.iterations):
            sample("steady", i)
dispatcher.close()
sample("closed", args.iterations if args.mode != "idle" else args.seconds)
print(json.dumps(dict(result="PASS", mode=args.mode, log_level=args.log_level)), flush=True)
