import torch.nn.functional as F

from onnx2pytorch.operations.base import Operator


class Pad(Operator):
    def __init__(self, mode="constant", padding=None, constant=None):
        self.mode = mode
        self.padding = padding
        self.constant_value = constant
        super().__init__()

    def forward(self, input, pads=None, value=0):
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        
        if self.constant_value is not None:
            value = self.constant_value
        
        return F.pad(input, list(pads), mode=self.mode, value=value)