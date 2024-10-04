from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

from src.data import Dataset
from src.models.diffusion.absorbing_diffusion_process import \
    AbsorbingDiffusionProcess
from src.models.diffusion.absorbing_diffusion_training import \
    absorbing_diffusion_training
from src.models.model import Model
from src.models.tabular_model import TabularModel
from src.nn.optimization import Optimization
from src.util import load_json, store_json


class AbsorbingDiscreteDiffusion(TabularModel):
    def __init__(
        self,
        *,
        demasking_predictor_class,
        demasking_predictor_params: dict,
        absorbing_diffusion_process: AbsorbingDiffusionProcess,
        optimization: Optimization,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.demasking_predictor_class = demasking_predictor_class
        self.demasking_predictor_params = demasking_predictor_params
        self.optimization = optimization
        self.aborbing_diffusion_process = absorbing_diffusion_process

    def _train(self, X: torch.Tensor) -> None:
        self.shape = X.data[0].shape
        self.classes_dim = X.max().item() + 1

        self.model = self.demasking_predictor_class(
            input_shape=self.shape,
            classes_dim=self.classes_dim,
            **self.demasking_predictor_params,
        )

        self.train_losses = absorbing_diffusion_training(
            train_set=X,
            masked_denoiser=self.model,
            optimization=self.optimization,
            absorbing_diffusion_process=self.aborbing_diffusion_process,
        )

    def _generate(self, n_samples: int, **kwargs) -> torch.Tensor:
        x = self.aborbing_diffusion_process.sample(
            n_samples=n_samples, demasking_predictor=self.model, **kwargs
        )
        return x

    def _store(self, model_path: str) -> None:
        store_json(
            {"shape": self.shape, "classes_dim": self.classes_dim},
            file=self.params_file(model_path),
        )
        self.model.store(model_path)

    def _load_(self, model_path: str) -> None:
        params = load_json(self.params_file(model_path))
        self.shape = params["shape"]
        self.classes_dim = params["classes_dim"]
        self.model = self.demasking_predictor_class(
            shape=self.shape,
            **self.demasking_predictor_params,
        )
        self.model.load_(model_path)
