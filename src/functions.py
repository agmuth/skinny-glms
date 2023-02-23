import numpy as np
from scipy.special import erf

MACHINE_EPS = np.finfo(np.float64).eps

def differentiate(f: callable, h: float=1e-4) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray):
        return 0.5 * h_inv * (f(x + h) - f(x - h))
    return f_prime

safe_division = np.vectorize(lambda x, y: x / y if np.abs(y) > MACHINE_EPS else np.sign(y) * x / MACHINE_EPS)

def identity(x: np.ndarray):
    return x

def logit(x: np.ndarray):
    return np.log(safe_division(x, 1-x))

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def negative_inverse(x: np.ndarray):
    return -1/x

def exponential(x: np.ndarray):
    return np.exp(x)

def log(x: np.ndarray):
    return np.log(x)