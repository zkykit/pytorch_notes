import torch
from torch import nn


class ky(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input + 1
        return output

Ky= ky()
x = torch.tensor(1.0)
output = Ky(x)
print(output)
