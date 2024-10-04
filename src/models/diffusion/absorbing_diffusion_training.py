from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.diffusion.absorbing_diffusion_process import \
    AbsorbingDiffusionProcess
from src.models.diffusion.old_demasking_predictor import OldMaskedPredictor
from src.models.score_based.nn.parameters_ema import ParametersEMA
from src.models.score_based.score_matching import log_statistic, loss_plot
from src.nn.optimization import Optimization
from src.util import reciprocal_distribution_sampler, uniform_sampler


def absorbing_diffusion_loss(
    X: Tensor,
    masked_denoiser: OldMaskedPredictor,
    absorbing_diffusion_process: AbsorbingDiffusionProcess,
    train: bool = True,
    loss_type="elbo",
) -> Tensor:
    masked_denoiser.train(train)
    return absorbing_diffusion_process.train_loss(
        X, masked_denoiser, loss_type=loss_type
    )[0]


def absorbing_diffusion_training(
    *,
    train_set: torch.Tensor,
    masked_denoiser: OldMaskedPredictor,
    absorbing_diffusion_process: AbsorbingDiffusionProcess,
    optimization: Optimization,
) -> list:
    # TODO: copied code
    device = get_available_device()
    tensor_train_set = TensorDataset(train_set.to(device))
    masked_denoiser.to(device)

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=optimization.batch_size,
        shuffle=True,
        drop_last=True,
    )
    masked_denoiser.train(True)
    optimizer = optimization.build_optimizer(masked_denoiser.parameters())

    epochs_losses = []
    n_batches = len(train_loader)

    parameters_ema = (
        ParametersEMA(
            parameters=masked_denoiser.parameters(),
            decay=optimization.parameters_momentum,
        )
        if optimization.parameters_momentum
        else None
    )

    for epoch in range(optimization.epochs):
        log_statistic(name="epoch", value=epoch + 1)
        losses = np.zeros(n_batches)

        pbar = tqdm(total=n_batches)
        for i, X_batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = absorbing_diffusion_loss(
                X=X_batch[0],
                masked_denoiser=masked_denoiser,
                train=True,
                absorbing_diffusion_process=absorbing_diffusion_process,
            )
            loss.backward()

            if optimization.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(masked_denoiser.parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

            optimizer.step()
            losses[i] = loss.detach().item()
            pbar.update(1)
        pbar.close()

        epoch_loss = np.mean(losses)
        epochs_losses.append(epoch_loss)
        log_statistic(name="train loss", value=epoch_loss)
        print()

        loss_plot(
            losses=epochs_losses,
            epoch=epoch,
            freq=1 + optimization.epochs // 10,
        )
        if parameters_ema:
            parameters_ema.update(masked_denoiser.parameters())

    if parameters_ema:
        parameters_ema.copy_to(masked_denoiser.parameters())
    loss_plot(losses=epochs_losses)

    return epochs_losses
