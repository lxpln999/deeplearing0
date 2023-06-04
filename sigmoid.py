import numpy as np

class Sigmoid:
    def __init__(self) -> None:
        pass

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))