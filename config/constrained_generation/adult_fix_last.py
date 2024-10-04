import os

import torch

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Discrete, OneHotlogic
from src.constraints.real_logic import Product as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

model_path = find(str(MODELS_FOLDER), pattern="*adult*")

i = names_index_map("data/adult.csv")
col = i["income"]

NAME = "adult_fix_last"
SEED = None

is_rich = lambda x: Discrete.equal(x[:, -1], 2)

constraint = Constraint(f=is_rich, explicit=False)

generation_options = dict(
    n_samples=1000,
    resample_only_prediction=True,
    gibbs_steps=2,
    final_gibbs_steps=2,
    one_hot_soft_constraint=lambda x: 0,  # OneHotlogic.equal(x[:, -1], 2),
    dmala_params=dict(step_size=0.5, temp=2.0, max_steps=5, patience=None, mh=True),
)


CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
