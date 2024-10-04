from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.gradient_mixer import (linear_interpolation,
                                            noise_weighting, snr)
from src.constraints.real_logic import OneHotlogic as Logic
from src.models.score_based.sdes.sampler import EulerMethod
from src.util import find, names_index_map

NAME = "equal_pair2"
SEED = None
model_path = find(str(MODELS_FOLDER), pattern=f"*{NAME}*")

predicate = lambda x: (x[:, -1] != 0)

constraint = Constraint(f=predicate, explicit=False)

generation_options = dict(
    n_samples=5000,
    resample_only_prediction=True,
    gibbs_steps=2,
    final_gibbs_steps=10,
    one_hot_soft_constraint=lambda x: Logic.different(x[:, -1], 0),
)

CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
)
