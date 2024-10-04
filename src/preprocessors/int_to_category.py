import torch
from torch import Tensor

from .preprocessor import TensorPreprocessor


def values_to_index(x: Tensor, *, values: Tensor) -> Tensor:
    d = torch.zeros(int(torch.max(values).item()) + 1).long()
    d[values] = torch.arange(len(values)).long()
    return d[x]


class IntToCategory(TensorPreprocessor):
    def fit(self, x: Tensor) -> None:
        self.parameters["shape"] = list(x[0].shape)
        self.parameters["possible_values"] = torch.unique(x.long()).tolist()

    def transform(self, x: Tensor) -> Tensor:
        return values_to_index(
            x.long().flatten(1), values=torch.tensor(self.parameters["possible_values"])
        )

    def reverse_transform(self, x: Tensor) -> Tensor:
        return torch.tensor(self.parameters["possible_values"], device=x.device)[
            x
        ].reshape([-1] + self.parameters["shape"])
