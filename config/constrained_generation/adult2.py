import os

import pandas as pd
import torch

from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import Discrete as Logic
from src.util import find, names_index_map

model_path = find(str(MODELS_FOLDER), pattern="*adult*")

i = names_index_map("data/adult.csv")
col = i["income"]

NAME = "adult"
SEED = None

predicate = lambda x: Logic.or_(
    Logic.and_(Logic.equal(x["income"], ">50K"), Logic.equal(x["sex"], "Female")),
    Logic.and_(
        Logic.greater(x["fnlwgt"], 250000), Logic.greater(x["education-num"], 12)
    ),
)

constraint = Constraint(f=predicate, explicit=True)

generation_options = dict(
    n_samples=1000,
    resample_only_prediction=True,
    gibbs_steps=5,
    final_gibbs_steps=5,
    # one_hot_soft_constraint = lambda x : Logic.greater(x[:, -1, 2], 1.)
)


CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
