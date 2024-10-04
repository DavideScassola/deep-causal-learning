from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from src.constants import DEFAULT_INPUT_ONLY_MASK_ID, DEFAULT_MASK_ID
from src.nn.demasking_predictor import DemaskingPredictor
from src.util import bernoulli


class DiffusionProcess(ABC):
    @abstractmethod
    def sample(self, *, x0: Tensor, t: Tensor | None = None) -> Tensor:
        pass


class AbsorbingDiffusionProcess(ABC):
    def loss_mask(self, x: Tensor) -> Tensor:
        return x == self.loss_mask_id()

    def loss_mask_id(self) -> int:
        return DEFAULT_MASK_ID

    def input_mask_id(self) -> int:
        return DEFAULT_MASK_ID

    def to_nn_input(self, x: Tensor) -> Tensor:
        return torch.where(x < 0, DEFAULT_MASK_ID, x)

    @abstractmethod
    def sample(self, *, x0: Tensor, t: Tensor | None = None) -> Tensor:
        pass

    def sample_with_loss_mask(self, *, x0: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.sample(x0=x0)
        return self.to_nn_input(x), self.loss_mask(x)

    @abstractmethod
    def sample_transition(self, *, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """
        sample from p(x{t-1} | x0, xt)
        """
        pass

    def denoising_loss(self, X: Tensor, nn: DemaskingPredictor) -> Tensor:
        xt, loss_mask = self.sample_with_loss_mask(x0=X)
        return F.cross_entropy(
            input=nn(xt)[loss_mask],
            target=X[loss_mask],
            reduction="mean",
        )


class GibbsMask(AbsorbingDiffusionProcess):
    def __init__(self, mask_id: int = DEFAULT_MASK_ID) -> None:
        self.mask_id = mask_id

    def sample(self, *, x0: Tensor, t: Tensor | None = None) -> Tensor:
        masked_x0 = x0.clone()
        n, p = x0.shape
        indices = torch.randint(p, size=(n,))
        masked_x0[torch.arange(n), indices] = self.mask_id
        return masked_x0

    def sample_transition(self, *, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        pass


class AutoregressiveMask(AbsorbingDiffusionProcess):
    def __init__(self, mask_id: int = DEFAULT_MASK_ID) -> None:
        self.mask_id = mask_id
        self.possible_masks = None

    def get_possible_ar_masks(self, x0: Tensor) -> Tensor:
        if self.possible_masks is None:
            self.possible_masks = torch.tril(
                torch.ones(
                    (x0.shape[1], x0.shape[1]), dtype=torch.bool, device=x0.device
                ),
                diagonal=-1,
            )
        return self.possible_masks  # TODO: maybe exclude the first row (all masked)

    def sample(self, *, x0: Tensor, t: Tensor | None = None) -> Tensor:
        n, p = x0.shape
        if t is None:
            t = torch.randint(p, size=(n,), device=x0.device)
        if torch.any(t > p):
            raise ValueError(f"t={t} is greater than the number of features in x0={p}")
        return torch.where(self.get_possible_ar_masks(x0)[t], x0, self.mask_id)

    def sample_transition(self, *, xt: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        x_prev = xt.clone()
        i = xt.shape[1] - t
        x_prev[:, i] = x0[:, i]
        return x_prev


class AutoregressiveGibbsMask(AutoregressiveMask):
    def __init__(
        self,
        only_input_mask_id: int = DEFAULT_INPUT_ONLY_MASK_ID,
        input_output_mask_id=DEFAULT_MASK_ID,
        gibbs_mask_prob: float = 0.5,
    ) -> None:
        self.only_input_mask_id = only_input_mask_id
        self.input_output_mask_id = input_output_mask_id
        self.gibbs_mask_prob = gibbs_mask_prob
        self.possible_masks = None

    def sample(self, *, x0: Tensor) -> Tensor:
        n, p = x0.shape

        xt = x0.clone()
        t_ar = torch.randint(low=2, high=p, size=(n,), device=x0.device)
        t_gibbs = torch.randint(p, size=(n,), device=x0.device) % (t_ar - 1)
        ar_mask = self.get_possible_ar_masks(x0)[t_ar]
        xt[~ar_mask] = self.only_input_mask_id
        xt[torch.arange(n), t_gibbs] = self.input_output_mask_id
        return xt
