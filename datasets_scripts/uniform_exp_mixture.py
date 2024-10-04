import os

import numpy as np
import pandas as pd


def random_mixture(mix_and_samplers: tuple, size: int):
    mix_coefficients = [k["p"] for k in mix_and_samplers]
    samplers = [k["sampler"] for k in mix_and_samplers]

    z = np.random.multinomial(n=size, pvals=mix_coefficients)
    x = np.concatenate([samplers[i](n) for i, n in enumerate(z)], axis=0)
    np.random.shuffle(x)
    return x


def generate(size: int = 1000) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "uniform(-120, 20)": np.random.uniform(-120, -20, size),
            "exponential(0.1)": np.random.exponential(0.1, size),
            "mixture(0.2: N(-3, 0.2), 0.8: N(500, 1))": random_mixture(
                (
                    {"p": 0.2, "sampler": lambda s: np.random.normal(-3, 0.2, s)},
                    {"p": 0.8, "sampler": lambda s: np.random.normal(500, 1, s)},
                ),
                size,
            ),
        }
    )


def store(df: pd.DataFrame):
    os.makedirs("data", exist_ok=True)
    name = os.path.basename(__file__).split(".py")[0]
    df.to_csv(f"data/{name}.csv", index=False)


if __name__ == "__main__":
    store(generate())
