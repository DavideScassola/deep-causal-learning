import torch

from src.models.auto_regressive.auto_regressive import AutoRegressive
from src.data import Dataset
from src.nn.demasking_predictor import DemaskingMLPConfig
from src.nn.optimization import Optimization
from src.preprocessors.everything_to_int import EverythingToInt
from src.train_config import TrainConfig
from src.util import get_available_device

SEED = 1234
NN_SCALE = 1
NAME = "repeated_numbers"

dataset = Dataset(path="data/repeated_numbers.csv", train_proportion=0.8)


mlp = DemaskingMLPConfig(
    hidden_channels=(NN_SCALE * 100, NN_SCALE * 100, NN_SCALE * 100),
    activation_layer=torch.nn.SiLU(),
    batch_norm=False,
)

mlp_opt = Optimization(
    epochs=60,
    batch_size=200,
    optimizer_class=torch.optim.RAdam,
    optimizer_hyperparameters={"lr": 1e-3, "weight_decay": 1e-4},
)

model = AutoRegressive(
    architecture=mlp,
    optimization=mlp_opt,
)

generation_options = dict(
    n_samples=2000,
)

CONFIG = TrainConfig(
    name=NAME,
    dataset=dataset,
    model=model,
    generation_options=generation_options,
    seed=SEED,
    device=get_available_device(verbose=True),
)
