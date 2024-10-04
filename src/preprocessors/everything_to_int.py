import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import Tensor

from src.util import is_numerical

from .preprocessor import Preprocessor
from .quantizer import digits2float, float2digits

INT_IS_NUMERICAL_THRESHOLD = 20
DEFAULT_FLOAT_BASE10_DIGITS = 5
DIGITS_BASE = 10


def get_max_digits(s: pd.Series):
    if s.dtype == int:
        return int(np.ceil(np.log10(np.max(s))))
    return DEFAULT_FLOAT_BASE10_DIGITS


def min_max_normalize(
    x: Tensor, *, minimum: float | int, maximum: float | int
) -> Tensor:
    if torch.any((x > maximum) | (x < minimum)):
        raise ValueError("x out of range")
    return (x - minimum) / (maximum - minimum)


def min_max_rescale(x: Tensor, *, minimum: float, maximum: float) -> Tensor:
    return x * (maximum - minimum) + minimum


class EverythingToInt(Preprocessor):
    def string_to_int(self, c: pd.Series, *, fit: bool) -> Tensor:
        le = LabelEncoder()
        if fit:
            out = le.fit(c)
        else:
            le.classes_ = np.array(self.parameters[c.name]["classes"])
        out = le.transform(c)
        if fit:
            self.parameters[c.name]["type"] = "string"
            self.parameters[c.name]["classes"] = le.classes_.tolist()
        return torch.tensor(out, dtype=torch.long)

    def quantize(self, c: pd.Series, *, fit: bool) -> Tensor:
        if fit:
            self.parameters[c.name]["type"] = (
                "float" if c.dtype == "float" else "numerical_int"
            )
            self.parameters[c.name]["max_digits"] = get_max_digits(c)

            self.parameters[c.name]["range"] = {
                "minimum": 0 if c.dtype == "int" else c.min(),
                "maximum": (
                    DIGITS_BASE ** self.parameters[c.name]["max_digits"]
                    if c.dtype == "int"
                    else c.max()
                ),
            }

        out = float2digits(
            min_max_normalize(
                torch.tensor(c.values).float(), **self.parameters[c.name]["range"]
            ),
            base=DIGITS_BASE,
            base10digits=self.parameters[c.name]["max_digits"],
        )
        return out

    def column_to_tensor(self, c: pd.Series, fit: bool):
        if c.dtype == "O":
            out = self.string_to_int(c, fit=fit).unsqueeze(1)
        elif is_numerical(c):
            assert (
                not c.isnull().any()
            ), f"There are NaNs in this numerical column: {c.name}"
            # TODO: this could be handled
            out = self.quantize(c, fit=fit)
        else:
            if fit:
                self.parameters[c.name]["type"] = "small_int"
            out = torch.tensor(c.values, dtype=torch.int64).unsqueeze(1)

        if fit:
            self.parameters[c.name]["slice"] = (
                self.slice_index,
                self.slice_index + out.shape[1],
            )
            self.slice_index = self.parameters[c.name]["slice"][1]
        return out

    def tensor_to_column(self, x: np.ndarray, *, column_name: str):
        y = x[:, slice(*self.parameters[column_name]["slice"])].squeeze()

        match self.parameters[column_name]["type"]:
            case "float" | "numerical_int":
                c = (
                    min_max_rescale(
                        digits2float(
                            torch.tensor(y),
                            base=DIGITS_BASE,
                            base10digits=self.parameters[column_name]["max_digits"],
                        ),
                        **self.parameters[column_name]["range"],
                    )
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                if self.parameters[column_name]["type"] == "numerical_int":
                    c = np.round(c).astype(int)
            case "string":
                le = LabelEncoder()
                le.classes_ = np.array(self.parameters[column_name]["classes"])
                c = le.inverse_transform(y)
            case _:
                c = y

        return pd.Series(c, name=column_name)

    def fit(self, x: pd.DataFrame):
        pass

    def transform(self, df: pd.DataFrame):
        self.slice_index = 0
        tensor_columns = [self.column_to_tensor(df[c], fit=False) for c in df.columns]
        out = torch.concat(tensor_columns, dim=1)
        assert (
            out.shape[1] == self.parameters["num_features"]
        ), f"error in number of features ({out.shape[1]} but expected {self.parameters['num_features']}), there is a bug in this preprocessor"
        return out

    def fit_transform(self, df: pd.DataFrame) -> torch.Tensor:
        self.parameters |= {c: {} for c in df.columns}
        self.parameters["names"] = list(df.columns)
        self.slice_index = 0
        tensor_columns = [self.column_to_tensor(df[c], fit=True) for c in df.columns]
        out = torch.concat(tensor_columns, dim=1)
        self.parameters["num_features"] = out.shape[1]
        print("num_features: ", self.parameters["num_features"])
        return out

    def reverse_transform(self, x: torch.Tensor) -> pd.DataFrame:
        return pd.concat(
            [
                self.tensor_to_column(x.cpu().numpy(), column_name=c)
                for c in self.parameters["names"]
            ],
            axis=1,
        )
