"""
Code from https://github.com/samb-t/unleashing-transformers
          https://github.com/samb-t/unleashing-transformers/blob/master/models/absorbing_diffusion.py
"""

import math

import numpy as np
import torch
import torch.distributions as dists
import torch.nn.functional as F
from tqdm import tqdm

from src.models.diffusion.old_demasking_predictor import OldMaskedPredictor


class AbsorbingDiffusionProcess:
    def __init__(self, mask_schedule, num_timesteps, mask_id=-1):
        self.num_timesteps = num_timesteps
        self.mask_id = mask_id
        self.mask_schedule = mask_schedule
        assert self.mask_schedule in ["random", "fixed"]

    def sample_time(self, b, device, method="uniform"):
        if method == "uniform":
            t = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt

        else:
            raise ValueError

    def q_sample(self, x_0, t):
        # samples q(x_t | x_0)
        # randomly set token to mask with probability t/T
        original_shape = x_0.shape
        x_0 = x_0.flatten(1)
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.rand_like(x_t.float()) < (
            t.float().unsqueeze(-1) / self.num_timesteps
        )
        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return (
            x_t.reshape(original_shape),
            x_0_ignore.reshape(original_shape),
            mask.reshape(original_shape),
        )

    def q_sample_mlm(self, x_0, t):
        # samples q(x_t | x_0)
        # fixed noise schedule, masks exactly int(t/T * latent_size) tokens
        original_shape = x_0.shape
        x_0 = x_0.flatten(1)
        x_t, x_0_ignore = x_0.clone(), x_0.clone()

        mask = torch.zeros_like(x_t).to(torch.bool)

        # TODO: offset so each n_masked_tokens is picked with equal probability
        n_masked_tokens = (t.float() / self.num_timesteps) * x_t.size(1)
        n_masked_tokens = torch.round(n_masked_tokens).to(torch.int64)
        n_masked_tokens[n_masked_tokens == 0] = 1
        ones = torch.ones_like(mask[0]).to(torch.bool).to(x_0.device)

        for idx, n_tokens_to_mask in enumerate(n_masked_tokens):
            index = torch.randperm(x_0.size(1))[:n_tokens_to_mask].to(x_0.device)
            mask[idx].scatter_(dim=0, index=index, src=ones)

        x_t[mask] = self.mask_id
        x_0_ignore[torch.bitwise_not(mask)] = -1
        return (
            x_t.reshape(original_shape),
            x_0_ignore.reshape(original_shape),
            mask.reshape(original_shape),
        )

    def train_loss(
        self, x_0, demasking_predictor: OldMaskedPredictor, loss_type: str = "elbo"
    ):
        b, device = x_0.size(0), x_0.device

        # choose what time steps to compute loss at
        t, pt = self.sample_time(b, device, "uniform")

        # make x noisy and denoise

        if self.mask_schedule == "random":
            x_t, x_0_ignore, mask = self.q_sample(x_0=x_0, t=t)
        elif self.mask_schedule == "fixed":
            x_t, x_0_ignore, mask = self.q_sample_mlm(x_0=x_0, t=t)

        # sample p(x_0 | x_t)
        x_0_hat_logits = demasking_predictor(x_t, t=t).permute(0, 2, 1)

        # Always compute ELBO for comparison purposes
        cross_entropy_loss = F.cross_entropy(
            x_0_hat_logits, x_0_ignore.flatten(1), ignore_index=-1, reduction="none"
        ).sum(1)
        vb_loss = cross_entropy_loss / t
        vb_loss = vb_loss / pt
        vb_loss = vb_loss / (math.log(2) * x_0.shape[1:].numel())
        if loss_type == "elbo":
            loss = vb_loss
        elif loss_type == "mlm":
            denom = mask.float().sum(1)
            denom[denom == 0] = 1  # prevent divide by 0 errors.
            loss = cross_entropy_loss / denom
        elif loss_type == "reweighted_elbo":
            weight = 1 - (t / self.num_timesteps)
            loss = weight * cross_entropy_loss
            loss = loss / (math.log(2) * x_0.shape[1:].numel())
        else:
            raise ValueError

        return loss.mean(), vb_loss.mean()

    def sample(
        self,
        n_samples: int,
        demasking_predictor: OldMaskedPredictor,
        sample_steps: int,
        temp=1.0,
    ):
        b, device = n_samples, demasking_predictor.device
        x_t = (
            torch.ones(
                (b, np.prod(demasking_predictor.get_input_shape())), device=device
            ).long()
            * self.mask_id
        )
        unmasked = torch.zeros_like(x_t, device=device).bool()

        for t in range(sample_steps, 0, -1):
            print(f"Sample timestep {t:4d}", end="\r")
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes = torch.rand(x_t.shape, device=device) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))
            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            x_0_logits = demasking_predictor(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_t[changes] = x_0_hat[changes]

        return x_t.reshape([x_t.shape[0]] + list(demasking_predictor.get_input_shape()))

    def sample_mlm(
        self,
        n_samples: int,
        demasking_predictor: OldMaskedPredictor,
        num_sample_steps: int,
        temp=1.0,
    ):
        b, device = n_samples, "cuda"
        x_0 = (
            torch.ones(
                (b, np.prod(demasking_predictor.get_input_shape())), device=device
            ).long()
            * self.mask_id
        )
        sample_steps = np.linspace(1, self.num_timesteps, num=num_sample_steps).astype(
            np.int64
        )

        for t in reversed(sample_steps):
            print(f"Sample timestep {t:4d}", end="\r")
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, _, _ = self.q_sample(x_0, t)
            x_0_logits = demasking_predictor(x_t, t=t)
            # scale by temperature
            x_0_logits = x_0_logits / temp
            x_0_dist = dists.Categorical(logits=x_0_logits)
            x_0_hat = x_0_dist.sample().long()
            x_0[x_t == self.mask_id] = x_0_hat[x_t == self.mask_id]

        return x_0

    @torch.no_grad()
    def elbo(self, x_0, demasking_predictor: OldMaskedPredictor):
        b, device = x_0.size(0), x_0.device
        elbo = 0.0
        for t in reversed(list(range(1, self.num_timesteps + 1))):
            print(f"Sample timestep {t:4d}", end="\r")
            t = torch.full((b,), t, device=device, dtype=torch.long)
            x_t, x_0_ignore, _ = self.q_sample(x_0=x_0, t=t)
            x_0_hat_logits = demasking_predictor(x_t, t=t).permute(0, 2, 1)
            cross_entropy_loss = F.cross_entropy(
                x_0_hat_logits, x_0_ignore, ignore_index=-1, reduction="none"
            ).sum(1)
            elbo += cross_entropy_loss / t
        return elbo
