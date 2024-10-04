from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.real_logic import Discrete, OneHotlogic
from src.util import find, get_available_device

model_path = find(str(MODELS_FOLDER), pattern="*repeated_numbers*")


SEED = None
NAME = "repeated_numbers"

soft_constraint = lambda x: OneHotlogic.different(x[:, -1], 0)
# soft_constraint = lambda x: OneHotlogic.equal(x[:, -1], 2)


predicate = lambda x: (x[:, -1] != 0)
# predicate = lambda x: (x[:, -1] == 2)

hard_constraint = Constraint(f=predicate, explicit=False)

generation_options = dict(
    n_samples=1000,
    resample_only_prediction=False,
    gibbs_steps=2,
    final_gibbs_steps=10,
    # one_hot_soft_constraint=soft_constraint,
    dmala_params=None,  # dict(step_size=2.0, temp=5.0, max_steps=None, patience=5),
)


CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=hard_constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
    device=get_available_device(),
)
