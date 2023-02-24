import numpy as np
from scipy.special import erf, erfinv

MACHINE_EPS = np.finfo(np.float64).eps

def differentiate(f: callable, h: float=1e-8) -> callable:
    h_inv = h**-1
    def f_prime(x: np.ndarray):
        return 0.5 * h_inv * (f(x + h) - f(x - h))
    return f_prime

safe_division = np.vectorize(lambda x, y: x / y if np.abs(y) > MACHINE_EPS else np.sign(y) * x / MACHINE_EPS)

def identity(x: np.ndarray):
    return x

def logit(x: np.ndarray):
    x[x >= 1.] = 1. - MACHINE_EPS
    x[x <= 0.] = MACHINE_EPS
    return np.log(safe_division(x, 1-x))

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def negative_inverse(x: np.ndarray):
    return -1/x

def exponential(x: np.ndarray):
    return np.exp(x)

def logarithm(x: np.ndarray):
    return np.log(x)

def inv_probit(x: np.ndarray):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def probit(x: np.ndarray):
    return np.sqrt(2) * erfinv(2*x - 1)

inv_cloglog = lambda x: 1 - np.exp(-np.exp(x))
inv_loglog = lambda x: np.exp(-np.exp(x))