import math
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.nn.module_config import ModuleConfig


@dataclass
class MLPconfig(ModuleConfig):
    hidden_channels: List[int]
    activation_layer: Callable = torch.nn.SiLU
    batch_norm: bool = False

    def get_module_class(self):
        return TimeResidualMLP


class FlexibleResidualBlock(nn.Module):
    """
    This block is a simple layer of an MLP if the input and output dimensions are different,
    otherwise it's a residual block, where the input is summed to the output
    """

    def __init__(
        self, *, input_size: int, output_size: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

        self.f = (
            self.residual_layer if input_size == output_size else self.standard_layer
        )

    def residual_layer(self, x: Tensor) -> Tensor:
        return x + self.activation(self.linear(x))

    def standard_layer(self, x: Tensor) -> Tensor:
        return self.activation(self.linear(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def sin_cos_embedding(t: torch.Tensor) -> torch.Tensor:
    x = t.reshape(-1, 1) * 2 * torch.pi
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def classic_embedding(t, embedding_dim=20, max_positions=10000) -> torch.Tensor:
    timesteps = t.flatten() * 1000
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class TimeResidualMLP(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        hidden_channels: List[int],
        activation_layer: nn.Module,
        batch_norm: bool = False,
        time_embedding: Callable | None = None
    ):
        super().__init__()
        # self.time_embedding = sin_cos_embedding
        # self.time_embedding = lambda x: x
        self.time_embedding = time_embedding
        self.batch_norm = batch_norm
        t_emb_dim = (
            self.time_embedding(torch.tensor([[1]])).shape[-1]
            if self.time_embedding
            else 0
        )

        concat_size = in_channels + t_emb_dim
        layers = [
            FlexibleResidualBlock(
                input_size=concat_size,
                output_size=hidden_channels[0],
                activation=activation_layer,
            )
        ]
        if self.batch_norm:
            layers = [torch.nn.BatchNorm1d(num_features=concat_size)] + layers

        for i in range(len(hidden_channels) - 1):
            layers.append(
                FlexibleResidualBlock(
                    input_size=hidden_channels[i],
                    output_size=hidden_channels[i + 1],
                    activation=activation_layer,
                )
            )
        layers.append(
            torch.nn.Linear(
                in_features=hidden_channels[-1], out_features=out_channels, bias=False
            )
        )
        # TODO: bias = False since the expected value of the score is 0, but not sure
        self.sequential = torch.nn.Sequential(*layers)

    def forward(self, X: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        if self.time_embedding:
            t_emb = self.time_embedding(t)
            X = torch.cat((X, t_emb), dim=-1)
        return self.sequential(X.flatten(1))
