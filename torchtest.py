import torch

print(torch.__version__)

a = torch.ones(3)

print(a)

points = torch.tensor([[4, 1, 4, 5], [5, 3, 5, 5], [2, 2, 3, 4]])

print(points.shape)


if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(x)
else:
    print("MPS is not found")

import imageio

