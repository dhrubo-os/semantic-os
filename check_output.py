import torch

model = torch.load("ivalue_jit.pt")

print(model.forward(torch.Tensor([1, 2, 3])))