import numpy as np


class GaussianKernel(object):
    def __init__(self, sigma=3):
        self.linear = False
        self.sigma = sigma

    def calculate(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (self.sigma ** 2)))


class LinearKernel(object):
    def __init__(self):
        self.linear = True

    def calculate(self, x1, x2):
        return np.dot(x1, x2)
