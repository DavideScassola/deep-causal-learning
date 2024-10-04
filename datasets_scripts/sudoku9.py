import os

import numpy as np
import pandas as pd
import tqdm
from sudoku import Sudoku


def generate(size: int = 5000) -> np.ndarray:
    return np.array(
        [
            np.array(Sudoku(3, seed=i).difficulty(1e-9).solve().board).flatten()
            for i in tqdm.tqdm(range(size), desc="generating sudoku matrices")
        ]
    )


def store(x: np.ndarray):
    name = os.path.basename(__file__).split(".py")[0]
    os.makedirs("data", exist_ok=True)
    np.save(f"data/{name}.npy", arr=x, allow_pickle=False)


if __name__ == "__main__":
    store(generate())
