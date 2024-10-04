import os
from pathlib import Path

import numpy as np
import pandas as pd


def self_base_name() -> str:
    return Path(os.path.basename(__file__)).stem


def csv_path() -> str:
    return f"data/{self_base_name()}.csv"


def generate(size: int = 5000):
    x = np.random.randint(0, 20, size=size)
    x[x >= 10] = 0
    return pd.DataFrame(np.tile(x, (5, 1)).T)


def main():
    os.makedirs("data", exist_ok=True)
    df = generate()
    df.to_csv(csv_path(), index=False)


if __name__ == "__main__":
    main()
