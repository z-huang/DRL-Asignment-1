import numpy as np


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numeric stability
    return exp_x / exp_x.sum()
