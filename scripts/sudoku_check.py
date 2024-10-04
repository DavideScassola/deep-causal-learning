import fnmatch
import os
import sys

import numpy as np
import sudoku
from icecream import ic


# Define your check_correctness function here
def check_correctness(x) -> bool:
    return not (
        "INVALID" in str(sudoku.Sudoku(3, 3, board=x.reshape(9, 9).tolist()).solve())
    )


def find_npy_files(folder):
    npy_files = []
    ic(folder)
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files


def main(folder):
    npy_files = find_npy_files(folder)

    if not npy_files:
        print("No .npy files found in the specified folder and its subdirectories.")
        return

    if len(npy_files) > 1:
        print(
            "too many .npy files found in the specified folder and its subdirectories."
        )

    sudoku_array = np.load(npy_files[0])

    total_valid = np.sum(np.array([check_correctness(s) for s in sudoku_array]))

    print(total_valid / len(sudoku_array))


if __name__ == "__main__":
    folder = sys.argv[-1]
    main(folder)
