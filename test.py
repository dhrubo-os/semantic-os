import torch
from torch import nn


class IValueProcessing(nn.Module):
    def __init__(self):
        super(IValueProcessing, self).__init__()

    def forward(self, testInput: torch.Tensor):
        print("Testing if the model executed or not first line")
        sumValue: torch.Tensor = torch.sum(testInput)
        print("Testing if the model executed or not second line")
        print(sumValue)
        return sumValue


jit_processing = torch.jit.script(IValueProcessing())

print(jit_processing.forward(torch.Tensor([1, 2, 3])))

print(jit_processing.code)
torch.jit.save(jit_processing, "ivalue_jit.pt")