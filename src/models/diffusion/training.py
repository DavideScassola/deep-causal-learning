import torch.nn.functional as F
from torch import Tensor

from src.models.diffusion.diffusion_process import AbsorbingDiffusionProcess
from src.nn.demasking_predictor import DemaskingPredictor


def demasking_loss(
    x0: Tensor,
    demasking_predictor: DemaskingPredictor,
    diffusion_process: AbsorbingDiffusionProcess,
) -> Tensor:
    xt, loss_mask = diffusion_process.sample_with_loss_mask(x0=x0)
    cross_entropy_loss = F.cross_entropy(
        input=demasking_predictor(xt)[loss_mask],
        target=x0[loss_mask],
        reduction="mean",
    )
    return cross_entropy_loss
