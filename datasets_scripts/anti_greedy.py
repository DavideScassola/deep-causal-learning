import os

import numpy as np
import pandas as pd
import tqdm
from sudoku import Sudoku


def generate(size: int = 5000) -> np.ndarray:
    x = np.random.randint(0, 10, size=size)
    y = np.random.randint(0, 10, size=size)
    z = np.random.randint(0, 10, size=size)

    fx1 = x % 2
    fx2 = x % 3
    fx3 = x % 4
    fx4 = fx2 + fx3
    fxy1 = (x + y) % 10
    fxy2 = (x + y) % 2
    fxy3 = y % 2 + x % 3
    fxyz1 = (x + y + z) % 2
    fxyz2 = x % 2 + y % 2 + z % 2
    fz1 = 2 * (z % 2)
    fy1 = (5 * (y % 4)) % 10
    fxyz3 = fx3 + fz1 + fy1
    fxyz4 = (fxyz2 - z) % 10

    return (
        np.stack(
            [
                x,
                y,
                z,
                fx1,
                fx2,
                fx3,
                fx4,
                fxy1,
                fxy2,
                fxy3,
                fxyz1,
                fxyz2,
                fz1,
                fy1,
                fxyz3,
                fxyz4,
            ],
            axis=1,
        )
        % 10
    )


def store(x: np.ndarray):
    name = os.path.basename(__file__).split(".py")[0]
    os.makedirs("data", exist_ok=True)
    np.save(f"data/{name}.npy", arr=x, allow_pickle=False)


if __name__ == "__main__":
    store(generate())
