from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor

from src.constants import DEFAULT_MASK_ID, WEIGHTS_FILE
from src.nn.gpt import GPT, GPTConfig
from src.nn.mlp import TimeResidualMLP
from src.util import batch_function, cross_entropy, gradient, logits_sample

from .module_config import ModuleConfig

DEFAULT_LOGIT_MINIMUM = -torch.inf


def expand_with_ar_masks(X: torch.Tensor, mask_id: int) -> torch.Tensor:
    """
    Expand the input tensor with autoregressive masks at the second dimension
    """
    mask_matrix = ~torch.tril(
        torch.ones((X.shape[1], X.shape[1]), dtype=torch.bool), diagonal=-1
    ).unsqueeze(0)

    return torch.where(
        mask_matrix,
        mask_id,
        X.unsqueeze(1),
    )


class DemaskingPredictor(torch.nn.Module):
    def __init__(
        self,
        input_shape,
        device,
        classes_dim,
        mask_id=DEFAULT_MASK_ID,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.mask_id = mask_id
        self.classes_dim = classes_dim
        self.actual_num_classes = None
        self.marginal_logits = None

    def fit_marginals(self, X: Tensor):
        prop = (
            torch.nn.functional.one_hot(X, num_classes=infer_num_classes(X))
            .float()
            .mean(0)
        )
        self.valid_int_mask = prop > 0
        self.marginal_logits = torch.where(
            prop > 0,
            torch.log(prop) + torch.log(torch.sum(prop > 0, dim=-1, keepdim=True)),
            DEFAULT_LOGIT_MINIMUM,
        ).to(X.device)

    def get_marginals(self):
        return self.marginal_logits

    def get_actual_num_classes(self):
        # TODO: probably works only for categories, this info should get out of the preprocessor
        if not hasattr(self, "actual_num_classes") or self.actual_num_classes is None:
            self.actual_num_classes = (
                self.classes_dim
                - torch.sum(self.marginal_logits == DEFAULT_LOGIT_MINIMUM, dim=1)
            ).tolist()
        return self.actual_num_classes

    def get_input_shape(self) -> tuple[int]:
        return self.input_shape

    def get_mask(self, masked_x: torch.Tensor) -> Tensor:
        return masked_x != self.mask_id

    def to(self, device):
        if self.marginal_logits is not None:
            self.marginal_logits = self.marginal_logits.to(device)
        return super().to(device)

    def get_device(self):
        return list(self.parameters())[0].device

    @abstractmethod
    def _forward(self, masked_x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, masked_x: Tensor) -> Tensor:
        out = torch.where(
            is_all_masked(masked_x, self.mask_id).unsqueeze(1).unsqueeze(1),
            0.0,
            self._forward(masked_x),
        ) + self.get_marginals().unsqueeze(0)

        return out

    def cross_entropy(self, *, x: Tensor, mask: Tensor) -> Tensor:
        logits = self.forward(torch.where(mask, x, self.mask_id))
        targets = torch.where(mask, self.mask_id, x)
        return F.cross_entropy(
            logits.permute(0, 2, 1),
            targets,
            ignore_index=self.mask_id,
            reduction="none",
        ).sum(1)

    def store(self, path: str):
        # TODO: using safe tensors would be better
        self.to("cpu")
        torch.save(self.state_dict(), f"{path}/{WEIGHTS_FILE}.pth")

    def load_(self, path: str):
        # TODO: using safe tensors would be better
        self.load_state_dict(
            torch.load(f"{path}/{WEIGHTS_FILE}.pth", map_location=self.device)
        )

    def prediction_sample(self, x: Tensor) -> Tensor:
        return torch.where(x == self.mask_id, logits_sample(self(x)), x)

    def ar_log_likelihood(
        self, X: Tensor, *, no_grad: bool = True, verbose=False
    ) -> Tensor:
        self.eval()
        with torch.set_grad_enabled(not no_grad):
            x = torch.full_like(X, self.mask_id, device=X.device)
            ll = torch.zeros(X.shape[0], device=X.device)
            it = (
                range(X.shape[1])
                if not verbose
                else tqdm.tqdm(range(X.shape[1]), desc="log likelihood computation")
            )
            for i in it:
                ll += -cross_entropy(
                    logits=self(x)[:, i, :],
                    targets=X[:, i],
                )
                x[:, i] = X[:, i]
        return ll

    def vectorized_ar_log_likelihood(
        self,
        X: torch.Tensor,
        *,
        max_batch_size: int = 500,
        no_grad: bool = True,
    ) -> torch.Tensor:
        # TODO: check correctness
        with torch.set_grad_enabled(not no_grad):
            self.eval()

            X_all_masks = expand_with_ar_masks(X, self.mask_id)

            def nn_logits(X: Tensor) -> Tensor:
                return self(X.reshape(-1, X.shape[-1])).reshape(list(X.shape) + [-1])

            X_all_masks_logits = batch_function(
                nn_logits,
                X=X_all_masks,
                max_batch_size=max_batch_size,
                verbose_lim=100,
                verbose_text="ll computation",
            )

            i = torch.arange(X_all_masks.shape[1])

            ll = -cross_entropy(
                logits=X_all_masks_logits[:, i, i],
                targets=X,
            ).sum(1)

        return ll


def ll_grad(self, X: Tensor) -> Tensor:
    raise NotImplementedError


def masked_one_hot_embedding(x: Tensor, classes_dim: int, mask_id: int = -1) -> Tensor:
    assert "Long" in x.type(), "x must be LongTensor"
    # mask = x == mask_id
    # TODO: this is done in order to output a single tensor, but it will make it sparse
    one_hot = F.one_hot(x + 1, num_classes=classes_dim + 1)
    return one_hot[:, :, 1:]


def is_all_masked(x: Tensor, mask_id: int = -1):
    return torch.all(x == mask_id, dim=1)


class DemaskingMLP(DemaskingPredictor):
    def __init__(
        self,
        input_shape,
        device,
        classes_dim: list | int,
        mask_id=DEFAULT_MASK_ID,
        **mlp_args,
    ) -> None:
        super().__init__(input_shape, device, mask_id)
        self.classes_dim = classes_dim
        numel = int(np.prod(np.array(input_shape))) * classes_dim
        self.mlp = TimeResidualMLP(
            in_channels=numel, out_channels=numel, time_embedding=None, **mlp_args
        )

    def _forward(self, masked_x: Tensor) -> Tensor:
        X = masked_one_hot_embedding(
            masked_x.flatten(1), self.classes_dim, self.mask_id
        ).float()
        return self.mlp(X=X.flatten(1)).reshape(X.shape)

    def ll_from_one_hot(self, x: Tensor) -> Tensor:
        B, F, C = x.shape  # x: (Batch, Features, Categories)
        mask_matrix = ~torch.tril(
            torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool), diagonal=-1
        )
        expanded_x = torch.where(
            mask_matrix.unsqueeze(0).unsqueeze(-1), 0, x.unsqueeze(1)
        )  # (B, F, F, C)
        expanded_x_reshaped = expanded_x.reshape(B * F, F * C)  # (BxF, FxC)
        logits = torch.where(
            torch.all(expanded_x_reshaped == 0, dim=-1).reshape(B * F, 1, 1),
            0.0,
            self.mlp(expanded_x_reshaped).reshape(B * F, F, C),
        ) + self.get_marginals().unsqueeze(0)
        useful_logits = logits.reshape(B, F, F, C)[
            :, torch.arange(F), torch.arange(F)
        ]  # (B, F, F, C)
        useful_logits[useful_logits == -torch.inf] = -1000
        return -torch.nn.functional.cross_entropy(
            useful_logits.permute(0, -1, 1),
            x.permute(0, -1, 1),
            reduction="none",
        ).sum(1)

    def ll_grad(self, X_discrete: Tensor) -> Tensor:
        X_continuous = masked_one_hot_embedding(
            X_discrete.flatten(1), self.classes_dim, self.mask_id
        ).float()
        self.eval()

        f = self.ll_from_one_hot

        return gradient(f=f, X=X_continuous)


class DemaskingNN(DemaskingPredictor):
    def __init__(
        self,
        input_shape,
        device,
        config: ModuleConfig,
        classes_dim: list | int,
        mask_id=DEFAULT_MASK_ID,
    ) -> None:
        super().__init__(input_shape, device, mask_id)
        self.input_shape = input_shape
        self.output_shape = list(input_shape) + [classes_dim]
        self.classes_dim = classes_dim
        self.module = config.build(
            input_shape=input_shape, output_shape=self.output_shape, device=device
        )


class DemaskingPredictorConfig(ModuleConfig):
    mask_id: int = DEFAULT_MASK_ID


def infer_num_classes(data_samples: Tensor) -> int:
    return int(data_samples.max().item()) + 1


@dataclass
class DemaskingMLPConfig(DemaskingPredictorConfig):
    hidden_channels: tuple
    activation_layer: torch.nn.Module
    batch_norm: bool = False

    def build_fit(self, data_samples: Tensor) -> DemaskingMLP:
        return DemaskingMLP(
            input_shape=data_samples[0].shape,
            device=data_samples.device,
            classes_dim=infer_num_classes(data_samples),
            **vars(self),
        )


@dataclass
class DemaskingNNConfig(DemaskingPredictorConfig):
    config: ModuleConfig

    def build_fit(self, data_samples: Tensor) -> DemaskingNN:
        return DemaskingNN(
            input_shape=data_samples[0].shape,
            device=data_samples.device,
            config=self.config,
            classes_dim=infer_num_classes(data_samples),
        )


class NanoGPT(DemaskingPredictor):
    def __init__(
        self,
        input_shape,
        device,
        classes_dim: int,
        config: GPTConfig,
        mask_id=DEFAULT_MASK_ID,
    ) -> None:
        super().__init__(input_shape, device, classes_dim, mask_id)
        self.gpt = GPT(config=config)

    def _forward(self, masked_x: Tensor) -> Tensor:
        return self.gpt(masked_x, mask=self.get_mask(masked_x))

    def forward(self, masked_x: Tensor) -> Tensor:
        out = torch.where(
            is_all_masked(masked_x, self.mask_id).unsqueeze(1).unsqueeze(1),
            0.0,
            self._forward(masked_x),
        ) + self.get_marginals().unsqueeze(0)
        # out[:, self.get_marginals() == -torch.inf] = -torch.inf
        return out


@dataclass
class NanoGPTConfig(DemaskingPredictorConfig):
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    def build_fit(self, data_samples: Tensor) -> NanoGPT:
        return NanoGPT(
            input_shape=data_samples[0].shape,
            device=data_samples.device,
            classes_dim=infer_num_classes(data_samples),
            config=GPTConfig(
                block_size=data_samples.shape[1],
                vocab_size=infer_num_classes(
                    data_samples
                ),  # TODO: +1 is due to mask_id, even if it not used
                **vars(self),
            ),
        )
