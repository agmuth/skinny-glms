import numpy as np
from numba.pycc import CC
from numba import njit
import math
import scipy.special as sc
from scipy.stats import norm


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

@njit()
@cc.export("erf", cc_2d_array_to_2d_array)
def erf(x: np.ndarray) -> np.ndarray:
    n= x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = sc.erf(x[i, j])
    return res
    # coefs = [1., 1/3, 1/10, 1/42, 1/216, 1/1320]
    # erf_of_x = x
    # x_squared = np.square(x)
    # for n in range(1, len(coefs)):
    #     x *= x_squared      
    #     erf_of_x += (-1)**n * x * coefs[n]
    # erf_of_x *= 2 / math.sqrt(math.pi)
    # return erf_of_x

@njit()
@cc.export("erfinv", cc_2d_array_to_2d_array)
def erfinv(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = sc.erfinv(x[i, j])
    return res
    # coefs = [1., 1/12, 7/480, 127/40320, 4369/5806080, 34807/18247]
    # erfinv_of_x = x
    # x_squared = np.square(x)
    # for n in range(1, len(coefs)):
    #     x *= x_squared      
    #     erfinv_of_x += math.pi**n * coefs[n] * x

    # erfinv_of_x *= 0.5 * math.sqrt(math.pi)
    # return erfinv_of_x

# @njit()
# @cc.export("norm_cdf", cc_2d_array_to_2d_array)
def norm_cdf(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = norm.cdf(x[i, j])
    return res

# @njit()
# @cc.export("norm_ppf", cc_2d_array_to_2d_array)
def norm_ppf(x: np.ndarray) -> np.ndarray:
    x = clip_probability(x)
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = norm.ppf(x[i, j])
    return res

# @njit()
# @cc.export("norm_pdf", cc_2d_array_to_2d_array)
def norm_pdf(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = norm.pdf(x[i, j])
    return res

@njit()
@cc.export("inv_probit", cc_2d_array_to_2d_array)
def inv_probit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + erf(x / np.sqrt(2)))

@njit()
@cc.export("probit", cc_2d_array_to_2d_array)
def probit(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2) * erfinv(2*clip_probability(x) - 1)

@njit()
@cc.export("inv_probit_deriv", cc_2d_array_to_2d_array)
def inv_probit_deriv(x: np.ndarray) -> np.ndarray:
    return 1 / np.sqrt(2*np.pi) * exponential(-0.5*np.square(x))

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