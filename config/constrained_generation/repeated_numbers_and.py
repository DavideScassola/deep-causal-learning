from src.constants import MODELS_FOLDER
from src.constrained_generation_config import ConstrainedGenerationConfig
from src.constraints.constraint import Constraint
from src.constraints.real_logic import Discrete, OneHotlogic
from src.util import find, get_available_device

model_path = find(str(MODELS_FOLDER), pattern="*repeated_numbers*")


SEED = None
NAME = "repeated_numbers_and"

soft_constraint = lambda x: OneHotlogic.and_(
    OneHotlogic.different(x[:, -1], 0), OneHotlogic.different(x[:, -3], 1)
)

# predicate = lambda x: (x[:, -1] != 0) & (x[:, -3] != 1)

predicate = lambda x: Discrete.and_(
    Discrete.unequal(x[:, -1], 0), Discrete.unequal(x[:, -3], 1)
)

hard_constraint = Constraint(f=predicate, explicit=False)

generation_options = dict(
    n_samples=1000,
    resample_only_prediction=True,
    gibbs_steps=0,
    final_gibbs_steps=2,
    one_hot_soft_constraint=soft_constraint,
    dmala_params=dict(step_size=2.0, temp=5.0, max_steps=None, patience=50),
)


CONFIG = ConstrainedGenerationConfig(
    name=NAME,
    constraint=hard_constraint,
    model_path=model_path,
    generation_options=generation_options,
    seed=SEED,
    device=get_available_device(),
)
