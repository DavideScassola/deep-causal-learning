import math

import torch
import torch.nn.functional as F
from torch import Tensor

from .preprocessor import TensorPreprocessor


def float2digits(x: Tensor, base: int = 2, base10digits: int = 4) -> Tensor:
    assert x.dtype == torch.float32, "x must be float32"
    n = 10**base10digits
    digits = math.ceil(base10digits * math.log(10, base))
    integers = torch.round(x * n).type(torch.int32)

    powers = base ** torch.arange(digits, 0, -1)
    digits_matrix = (
        base * torch.remainder(integers.unsqueeze(-1), powers.reshape(1, -1)) // powers
    )
    return digits_matrix.type(torch.int64)


def digits2float(x: Tensor, base: int = 2, base10digits: int = 4) -> Tensor:
    digits = math.ceil(base10digits * math.log(10, base))
    n = 10**base10digits
    powers = base ** torch.arange(digits - 1, -1, -1, device=x.device)
    return x.float() @ (powers.float()) / n


class Quantizer(TensorPreprocessor):
    def __init__(
        self,
        base: int = 2,
        base10digits: int = 4,
        one_hot: bool = False,
        flatten: bool = True,
    ) -> None:
        super().__init__()
        self.base = base
        self.base10digits = base10digits
        self.one_hot = one_hot
        self.flatten = flatten

    def fit(self, x: Tensor) -> None:
        if self.flatten:
            self.parameters["shape"] = list(x[0].shape)

    def transform(self, x: Tensor) -> Tensor:
        digits = float2digits(x, self.base, self.base10digits)

        if not self.one_hot:
            return digits.flatten(1) if self.flatten else digits

        out = (
            F.one_hot(digits, num_classes=self.base)
            if self.base > 2
            else digits.unsqueeze(-1)
        )

        return out

    def reverse_transform(self, x: Tensor) -> Tensor:
        x = x.reshape([x.shape[0]] + self.parameters["shape"] + [-1])
        digits = torch.argmax(x, dim=-1) if self.base > 2 and self.one_hot else x
        return digits2float(digits, self.base, self.base10digits)

    def serialize(self, p: dict):
        return super().serialize(p)
