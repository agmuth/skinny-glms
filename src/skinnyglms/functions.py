import numpy as np
from scipy.special import erf, erfinv

DTYPE = np.float64
MACHINE_EPS = np.finfo(DTYPE).eps
MACHINE_MIN = np.finfo(DTYPE).min
MACHINE_MAX = np.finfo(DTYPE).max
LOG_MACHINE_MAX = np.log(MACHINE_MAX)

def clip_probability(x: np.ndarray):
    return np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)

def identity(x: np.ndarray):
    return x

def logit(x: np.ndarray):
    x = clip_probability(x)
    return logarithm(x) - logarithm(1-x)

def sigmoid(x: np.ndarray):
    return 1 / (1 + exponential(-x))

def inverse(x: np.ndarray):
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return 1/x

def exponential(x: np.ndarray):
    x = np.clip(x, MACHINE_MIN, LOG_MACHINE_MAX)
    return np.exp(x)

def logarithm(x: np.ndarray):
    x = np.clip(x, MACHINE_EPS, MACHINE_MAX)
    return np.log(x)

def inv_probit(x: np.ndarray):
    return 0.5 * (1 + erf(x / np.sqrt(2)))

def probit(x: np.ndarray):
    x = clip_probability(x)
    return np.sqrt(2) * erfinv(2*x - 1)

def inv_cloglog(x: np.ndarray):
    return 1 - exponential(-exponential(x))

def cloglog(x: np.ndarray):
    x = clip_probability(x)
    return logarithm(-logarithm(1-x))

def inv_loglog(x: np.ndarray):
    return exponential(-exponential(x))

def loglog(x: np.ndarray):
    return logarithm(logarithm(x))