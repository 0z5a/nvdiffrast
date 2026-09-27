import json

import torch
import nvdiffrast.torch as dr

ctx = dr.RasterizeCudaContext(device="cuda:0")
base = torch.tensor([[[-0.7, -0.7, 0.0, 1.0],
                      [0.7, -0.7, 0.0, 1.0],
                      [0.0, 0.7, 0.0, 1.0]]],
                    device="cuda:0", dtype=torch.float32)
tri = torch.tensor([[0, 1, 2]], device="cuda:0", dtype=torch.int32)
attr = torch.tensor([[[0.0], [1.0], [0.5]]],
                    device="cuda:0", dtype=torch.float32)
row = col = 32

def render(positions):
    rast, _ = dr.rasterize(ctx, positions, tri, [64, 64])
    interp, _ = dr.interpolate(attr, rast, tri)
    return rast, interp[0, row, col, 0]

pos = base.clone().requires_grad_()
rast, value = render(pos)
assert int(rast[0, row, col, 3].item()) == 1
pixel_x = 2 * (col + 0.5) / 64 - 1
pixel_y = 2 * (row + 0.5) / 64 - 1
w2 = (pixel_y + 0.7) / 1.4
w1 = (pixel_x + 0.7 - 0.7 * w2) / 1.4
expected = w1 + 0.5 * w2
gradient = torch.autograd.grad(value, pos)[0]
analytic_dx = float(gradient[0, 1, 0].item())
differences = {}
for step in (0.001, 0.003, 0.01):
    plus = base.clone()
    minus = base.clone()
    plus[0, 1, 0] += step
    minus[0, 1, 0] -= step
    _, value_plus = render(plus)
    _, value_minus = render(minus)
    differences[str(step)] = float(((value_plus - value_minus) / (2 * step)).item())
torch.cuda.synchronize()
print(json.dumps({
    "pixel": [row, col],
    "interpolated": float(value.item()),
    "analytic_reference": expected,
    "autograd_position_derivative": analytic_dx,
    "position_finite_differences": differences,
}, indent=2))
assert abs(float(value.item()) - expected) < 1e-4
assert max(abs(x - analytic_dx) for x in differences.values()) < 1e-2
