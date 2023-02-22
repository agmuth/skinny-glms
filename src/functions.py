import numpy as np
from scipy.special import erf

MACHINE_EPS = np.finfo(np.float64).eps

safe_division = np.vectorize(lambda x, y: x / y if np.abs(y) > MACHINE_EPS else np.sign(y) * x / MACHINE_EPS)

def identity(x: np.ndarray):
    return x

def logit(x: np.ndarray):
    return np.log(safe_division(x, 1-x))

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

