import numpy as np
import scipy.special as sc
from numba.pycc import CC
from numba import njit
import numba_scipy
import math


cc = CC("functions")
# cc.verbose = True

DTYPE = np.float64
MACHINE_EPS = np.finfo(DTYPE).eps
MACHINE_MIN = np.finfo(DTYPE).min
MACHINE_MAX = np.finfo(DTYPE).max
LOG_MACHINE_MAX = np.log(MACHINE_MAX)

cc_2d_array_to_2d_array = "f8[:, :](f8[:, :])"


@njit()
@cc.export("clip_probability", cc_2d_array_to_2d_array)
def clip_probability(x: np.ndarray) -> np.ndarray:
    return np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)

@njit()
@cc.export("identity", cc_2d_array_to_2d_array)
def identity(x: np.ndarray) -> np.ndarray:
    return x

@njit()
@cc.export("logit", cc_2d_array_to_2d_array)
def logit(x: np.ndarray) -> np.ndarray:   
    x = clip_probability(x)
    return logarithm(x) - logarithm(1-x)

@njit()
@cc.export("sigmoid", cc_2d_array_to_2d_array)
def sigmoid(x: np.ndarray):
    return 1 / (1 + exponential(-x))

@njit()
@cc.export("inverse", cc_2d_array_to_2d_array)
def inverse(x: np.ndarray) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return 1/x

@njit()
@cc.export("exponential", cc_2d_array_to_2d_array)
def exponential(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, MACHINE_MIN, LOG_MACHINE_MAX)
    return np.exp(x)

@njit()
@cc.export("logarithm", cc_2d_array_to_2d_array)
def logarithm(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, MACHINE_EPS, MACHINE_MAX)
    return np.log(x)

# @njit()
# @cc.export("inv_probit", cc_2d_array_to_2d_array)
# def inv_probit(x: np.ndarray) -> np.ndarray:
#     return 0.5 * (1 + erf(x / np.sqrt(2)))

# @njit()
# @cc.export("probit", cc_2d_array_to_2d_array)
# def probit(x: np.ndarray) -> np.ndarray:
#     x = clip_probability(x)
#     return np.sqrt(2) * erfinv(2*x - 1)

@njit()
@cc.export("inv_cloglog", cc_2d_array_to_2d_array)
def inv_cloglog(x: np.ndarray) -> np.ndarray:
    return 1 - exponential(-exponential(x))

@njit()
@cc.export("cloglog", cc_2d_array_to_2d_array)
def cloglog(x: np.ndarray) -> np.ndarray:
    x = clip_probability(x)
    return logarithm(-logarithm(1-x))

@njit()
@cc.export("inv_loglog", cc_2d_array_to_2d_array)
def inv_loglog(x: np.ndarray) -> np.ndarray:
    return exponential(-exponential(x))

@njit()
@cc.export("loglog", cc_2d_array_to_2d_array)
def loglog(x: np.ndarray) -> np.ndarray:
    return logarithm(logarithm(x))



if __name__ == "__main__":
    cc.compile()