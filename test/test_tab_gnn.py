# test_math_operations.py
import sys
import unittest

sys.path.append(".")
from src.train_controller import run_model_training


class TestTabGNN(unittest.TestCase):

    def test_run(self):
        # TODO
        run_model_training("test/configs/adult.py")
        pass

    def test_performance(self):
        # TODO
        pass


if __name__ == "__main__":
    unittest.main()
