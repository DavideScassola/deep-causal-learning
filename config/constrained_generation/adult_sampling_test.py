import os

import torch

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

model_path = find(str(MODELS_FOLDER), pattern="*adult*")

NAME = "adult_sampling_test"
SEED = None

generation_options = dict(
    n_samples=10_000, gibbs_steps=4, final_gibbs_steps=10, resample_only_prediction=True
)

constraint = Constraint(f=lambda x: torch.ones(len(x)), explicit=False)

CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
