import json
import time
from pathlib import Path

import torch
import nvdiffrast.torch as dr
import _nvdiffrast_c

device = torch.device("cuda:0")
pos = torch.tensor([[[-0.7, -0.7, 0.0, 1.0],
                     [0.7, -0.7, 0.0, 1.0],
                     [0.0, 0.7, 0.0, 1.0]]],
                   device=device, dtype=torch.float32, requires_grad=True)
tri = torch.tensor([[0, 1, 2]], device=device, dtype=torch.int32)
attr = torch.tensor([[[0.0], [1.0], [0.5]]],
                    device=device, dtype=torch.float32, requires_grad=True)
start = time.perf_counter()
ctx = dr.RasterizeCudaContext(device=device)
rast, _ = dr.rasterize(ctx, pos, tri, resolution=[64, 64])
covered = rast[..., 3] > 0
assert bool(covered.any()) and bool((~covered).any())
assert bool(torch.isfinite(rast).all())
interp, _ = dr.interpolate(attr, rast, tri)
loss = interp[..., 0][covered].sum()
loss.backward()
torch.cuda.synchronize(device)
assert attr.grad is not None and bool(torch.isfinite(attr.grad).all())
assert pos.grad is not None and bool(torch.isfinite(pos.grad).all())
count = int(covered.sum().item())
grad_sum = float(attr.grad.sum().item())
assert abs(grad_sum - count) < 1e-2, (grad_sum, count)
with dr.DepthPeeler(ctx, pos.detach(), tri, resolution=[64, 64]):
    blocked = dr.rasterize(ctx, pos.detach(), tri, resolution=[64, 64])
assert ctx.active_depth_peeler is None
result = {
    "repo_sha": "253ac4fcea7de5f396371124af597e6cc957bfae",
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(device),
    "module": str(Path(dr.__file__).resolve()),
    "native_module": str(Path(_nvdiffrast_c.__file__).resolve()),
    "covered_pixels": count,
    "attribute_gradient_sum": grad_sum,
    "position_gradient_finite": True,
    "depth_peeler_blocked_return_type": type(blocked).__name__,
    "elapsed_seconds": round(time.perf_counter() - start, 4),
}
print(json.dumps(result, indent=2))
