import torch
from torch import nn

from onnx2pytorch.operations.base import Operator


class Sub(Operator):
    def __init__(self, y=None):
        super().__init__()
        self.y = torch.nn.Parameter(torch.Tensor(y))

    def forward(self, x, y=None):
        if y is None:
            y = self.y
        return torch.sub(x, y)
