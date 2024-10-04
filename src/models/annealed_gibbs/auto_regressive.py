import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import tqdm
from matplotlib import pyplot as plt
from torch import Tensor

from src.data import Dataset

from src.models.diffusion.diffusion_process import (AutoregressiveGibbsMask,
                                                    AutoregressiveMask,
                                                    GibbsMask)
from src.models.diffusion.training import demasking_loss
from src.models.tabular_model import TabularModel
from src.nn.demasking_predictor import (DemaskingPredictor,
                                        DemaskingPredictorConfig)
from src.nn.optimization import Optimization
from src.nn.training import nn_training
from src.preprocessors.preprocessor import (composed_inverse_transform,
                                            composed_transform)
from src.report import (REPORT_FOLDER_NAME, histograms_comparison,
                        statistics_comparison, store_samples, summary_report)
from src.util import (CategoricalBiasModel, edit_json,
                      gradient, gradient_descent,
                      load_json, logits_sample, pickle_load, pickle_store, store_json)


class AutoRegressive(TabularModel):
    def __init__(
        self,
        *,
        architecture: DemaskingPredictorConfig,
        optimization: Optimization,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.architecture = architecture
        self.optimization = optimization
        self.diffusion_process = AutoregressiveMask()

    def initialize(self, X: torch.Tensor):
        self.demasking_predictor: DemaskingPredictor = self.architecture.build_fit(X)

    def _train(self, X: torch.Tensor) -> None:
        self.initialize(X)
        self.demasking_predictor.fit_marginals(X)

        ar_diff = AutoregressiveMask()

        def loss_function(X: Tensor) -> Tensor:
            loss_ar = ar_diff.denoising_loss(X, self.demasking_predictor)
            return loss_ar

        self.train_losses = nn_training(
            train_set=X,
            optimization=self.optimization,
            loss_function=loss_function,
            nn=self.demasking_predictor,
        )

    def _generate(
        self,
        n_samples: int,
        device: str | None = None,
        **kwargs,
    ) -> torch.Tensor:

        self.demasking_predictor.to(device)
        self.demasking_predictor.eval()

        input_dimension = self.demasking_predictor.input_shape[0]
        mask_id = self.demasking_predictor.mask_id

        x = torch.full(
            (n_samples, input_dimension),
            mask_id,
            device=self.demasking_predictor.get_device(),
        )

        for i in range(input_dimension):
            x0_logits = self.demasking_predictor(x)  # model prediction
            x0 = logits_sample(
                x0_logits
            )  # Technically not correct, since the model is not trained to predict logits of alredy fixed coponents
            x = self.diffusion_process.sample_transition(
                xt=x, x0=x0, t=input_dimension - i
            )  # p(x_t-1 | x_t, x_0)

        self.no_masks_check(x)
        return x

    def demasking_predictor_file(self, model_path: str):
        return f"{model_path}/demasking_predictor.pkl"

    def _store(self, model_path: str) -> None:
        self.demasking_predictor.to("cpu")
        pickle_store(
            self.demasking_predictor, file=self.demasking_predictor_file(model_path)
        )

    def _load_(self, model_path: str) -> None:
        self.demasking_predictor = pickle_load(
            self.demasking_predictor_file(model_path)
        )

    def no_masks_check(self, x: Tensor):
        if torch.any(x < 0):
            raise ValueError("There are still masked values in the generated sample")

    def log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        return self.demasking_predictor.vectorized_ar_log_likelihood(
            X, max_batch_size=50
        )

    def log_likelihood_center(self) -> float:
        # TODO: currently works only because all components have the same number of classes
        return (
            -math.log(self.demasking_predictor.classes_dim)
            * self.demasking_predictor.input_shape.numel()
        )


    def specific_report_plots(self, path: Path) -> None:
        if hasattr(self, "dataset"):
            dataset = {
                s: self.get_preprocessed_data(
                    train=s == "train",
                    device=self.demasking_predictor.get_device(),
                    fit=False,
                )
                for s in ("train", "test")
            }

            ll = {
                f"{s}_mean_log_likelihood": self.log_likelihood(dataset[s])
                .mean()
                .item()
                for s in ("train", "test")
            }

            for s in ("train", "test"):
                ll[f"{s}_centered_mean_ll"] = (
                    ll[f"{s}_mean_log_likelihood"] - self.log_likelihood_center()
                )

            m = CategoricalBiasModel(dataset["train"])

            for s in ("train", "test"):
                ll[f"{s}_bias_improvement"] = (
                    ll[f"{s}_mean_log_likelihood"]
                    - m.log_likelihood(dataset[s]).mean(0).sum().item()
                )

            store_json(
                ll,
                file=path / Path("log_likelihood.json"),
            )
            print(ll)

        if (
            self.demasking_predictor.input_shape[0] == 2
            and self.demasking_predictor.classes_dim == 2
        ):
            self.plot_2d_likelihood(path)


    def plot_2d_likelihood(self, path: Path) -> None:
        if hasattr(self.demasking_predictor, "ll_from_one_hot"):
            x = np.linspace(0, 1, 100)
            y = np.linspace(0, 1, 100)
            X, Y = np.meshgrid(x, y)

            X_torch = torch.from_numpy(X).float()
            Y_torch = torch.from_numpy(Y).float()
            XY = torch.stack([X_torch, Y_torch], dim=-1).reshape(-1, 2)
            XY = torch.stack([1 - XY, XY], dim=-1)
            Z = (
                self.demasking_predictor.ll_from_one_hot(XY)
                .detach()
                .cpu()
                .numpy()
                .reshape(100, 100)
            )

            plt.imshow(Z, origin="lower", interpolation=None, cmap="viridis")
            plt.colorbar()
            plt.savefig(f"{path}/2d_likelihood.png")
            plt.cla()


    def generate_report(
        self,
        *,
        path: str | Path,
        generation_options: dict,
    ):
        report_folder = path / Path(REPORT_FOLDER_NAME)
        os.makedirs(report_folder, exist_ok=False)

        constrained_samples_tensor = self._generate(**generation_options)

        samples_df: pd.DataFrame = composed_inverse_transform(
            constrained_samples_tensor, preprocessors=self.preprocessors
        )

        store_samples(
            df_generated=samples_df, path=report_folder, name="samples.csv"
        )

        if not self.dataset:
            raise ValueError("dataset should be defined")

        statistics_comparison(
            df={
                "generation": samples_df,
                "test_set": self.dataset.get(train=False),
            },
            file=report_folder / Path("stats.json"),
        )
        self.specific_report_plots(report_folder)
        summary_report(path=report_folder)

        with edit_json(report_folder / "stats_summary.json") as summary:
            ll_stats = load_json(report_folder / "log_likelihood.json")
            summary["ll_train_bias_improvement"] = ll_stats[
                "train_bias_improvement"
            ]
            summary["ll_test_bias_improvement"] = ll_stats["test_bias_improvement"]

        histograms_comparison(
            df_generated=samples_df,
            name_generated=f"guidance",
            df_train=self.dataset.get(train=False),
            name_original=f"test_set",
            path=report_folder
        )
