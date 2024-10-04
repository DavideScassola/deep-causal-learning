from pathlib import Path

from src.constants import CONFIG_FILE_NAME
from src.models.model import Model
from src.train_config import getConfig


def load_model(experiment_path: str) -> Model:
    train_config = getConfig(Path(experiment_path) / CONFIG_FILE_NAME)
    train_config.model.load_(Path(experiment_path))
    return train_config.model
