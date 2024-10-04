from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.util import NamedFunction, input_listener

from .optimization import Optimization


def log_statistic(*, name: str, value) -> None:
    print(f"{name}: {value}")


def loss_plot(
    *,
    losses,
    epoch: int = 0,
    freq: int = 1,
    loss_plot_name="losses.png",
    path: str | None = None,
) -> None:
    file = f"{path}/{loss_plot_name}" if path else loss_plot_name
    if epoch % freq == 0:
        plt.plot(losses, alpha=0.8)
        plt.savefig(file)
        plt.close()


def nn_training(
    *,
    train_set: Tensor,
    optimization: Optimization,
    loss_function: Callable,
    nn: torch.nn.Module,
    device: str | None = None,
) -> list:
    # TODO: copied code
    tensor_train_set = TensorDataset(train_set.to(device))

    nn.to(train_set.device)

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=optimization.batch_size,
        shuffle=True,
        drop_last=True,
    )
    nn.train(True)

    optimizer = optimization.build_optimizer(nn.parameters())

    epochs_losses = []
    n_batches = len(train_loader)

    input_happened = input_listener() if optimization.interactive else False

    for epoch in range(optimization.epochs):
        if input_happened:
            break
        log_statistic(name="epoch", value=epoch + 1)
        losses = np.zeros(n_batches)

        pbar = tqdm.tqdm(total=n_batches)
        for i, X_batch in enumerate(train_loader):
            if input_happened:
                break
            optimizer.zero_grad()
            loss = loss_function(X=X_batch[0])
            loss.backward()

            if optimization.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(nn.parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

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

    loss_plot(losses=epochs_losses)
    nn.eval()
    return epochs_losses


def get_warmup_lr(
    *, epoch: int, warmup_lr: float = 1e-8, final_lr: float, warmup_epochs: int = 3
) -> float:
    if epoch > warmup_epochs:
        return final_lr
    return warmup_lr + epoch * (final_lr - warmup_lr) / warmup_epochs


def multiple_loss_nn_training(
    *,
    train_set: Tensor,
    loss_function: Callable,
    nn_opt: list[dict],
    device: str | None = None,
    logs: list[NamedFunction] = [],
) -> dict:
    # TODO: copied code
    tensor_train_set = TensorDataset(train_set.to(device))
    batch_size = nn_opt[0]["optimization"].batch_size
    epochs = nn_opt[0]["optimization"].epochs

    for o in nn_opt:
        o["nn"].to(train_set.device)
        o["nn"].train(True)
        o["optimizer"] = o["optimization"].build_optimizer(o["nn"].parameters())

    train_loader = DataLoader(
        dataset=tensor_train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    epochs_losses = {o["name"]: [] for o in nn_opt}
    n_batches = len(train_loader)

    input_happened = (
        input_listener() if nn_opt[0]["optimization"].interactive else False
    )

    for o in nn_opt:
        o["target_rl"] = o["optimizer"].param_groups[0]["lr"]

    for epoch in range(epochs):
        for o in nn_opt:
            if o["optimization"].warmup_epochs + 1 > epoch:
                o["optimizer"].param_groups[0]["lr"] = get_warmup_lr(
                    epoch=epoch,
                    final_lr=o["target_rl"],
                    warmup_epochs=o["optimization"].warmup_epochs,
                )
        if input_happened:
            break
        log_statistic(name="epoch", value=epoch + 1)
        losses = {o["name"]: np.zeros(n_batches) for o in nn_opt}

        pbar = tqdm.tqdm(total=n_batches)
        for i, X_batch in enumerate(train_loader):
            if input_happened:
                break
            loss = loss_function(X=X_batch[0])

            for loss_index, o in enumerate(nn_opt):
                o["optimizer"].zero_grad()
                loss[loss_index].backward(retain_graph=loss_index != len(nn_opt) - 1)

                if o["optimization"].gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(o["nn"].parameters(), max_norm=optimization.gradient_clipping)  # type: ignore

                o["optimizer"].step()
                losses[o["name"]][i] = loss[loss_index].detach().item()
            pbar.update(1)

        for l in logs:
            log_statistic(name=l.name, value=l())
        pbar.close()

        for o in nn_opt:
            epoch_loss = np.mean(losses[o["name"]])
            epochs_losses[o["name"]].append(epoch_loss)

            log_statistic(name=f"train loss {o['name']}", value=epoch_loss)

            loss_plot(
                losses=epochs_losses[o["name"]],
                epoch=epoch,
                freq=1 + epochs // 10,
                loss_plot_name=f"losses_{o['name']}.png",
            )
        print()

    for o in nn_opt:
        loss_plot(
            losses=epochs_losses[o["name"]],
            loss_plot_name=f"losses_{o['name']}.png",
        )

        o["nn"].eval()
    return epochs_losses
