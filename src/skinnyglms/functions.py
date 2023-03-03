import numpy as np
from scipy.special import erf, erfinv
from numba import jit 

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
    x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
    return np.log(x) - np.log(1-x)

def sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def negative_inverse(x: np.ndarray):
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, np.Inf)
    return -1/x

def exponential(x: np.ndarray):
    return np.exp(x)

def logarithm(x: np.ndarray):
    x = np.clip(x, MACHINE_EPS, np.Inf)
    return np.log(x)

def inv_probit(x: np.ndarray):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def probit(x: np.ndarray):
    x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
    return np.sqrt(2) * erfinv(2*x - 1)

def inv_cloglog(x: np.ndarray):
    return 1 - np.exp(-np.exp(x))

def cloglog(x: np.ndarray):
    x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
    return np.log(-np.log(1-x))

def inv_loglog(x: np.ndarray):
    return np.exp(-np.exp(x))

def loglog(x: np.ndarray):
    x = np.clip(x, MACHINE_EPS, np.Inf)
    return np.log(np.log(x))