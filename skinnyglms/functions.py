import math

import numpy as np
from numba import njit
from numba.pycc import CC

cc = CC("functions")
cc.verbose = True

DTYPE = np.float64
MACHINE_EPS = np.finfo(DTYPE).eps
MACHINE_MIN = np.finfo(DTYPE).min
MACHINE_MAX = np.finfo(DTYPE).max
LOG_MACHINE_MAX = np.log(MACHINE_MAX)

cc_2d_array_to_2d_array = "f8[:, :](f8[:, :])"


def generate_erf_coefs(n: int) -> np.ndarray:
    # ref: https://mathworld.wolfram.com/Erf.html
    coefs = (
        2
        / math.sqrt(math.pi)
        * np.array(
            [(-1) ** k / (math.gamma(k + 1) * (2 * k + 1)) for k in range(n + 1)]
        )
    )
    return coefs


def generate_erfinv_coefs(n: int) -> np.ndarray:
    # ref: https://mathworld.wolfram.com/InverseErf.html
    coefs = [1.0, 1.0]
    for k in range(2, n + 1):
        c_k = 0
        for j in range(k):
            c_k += coefs[j] * coefs[k - 1 - j] / ((j + 1) * (2 * j + 1))
        coefs.append(c_k)
    return np.array(
        [
            c / (2 * k + 1) * (0.5 * math.sqrt(math.pi)) ** (2 * k + 1)
            for k, c in enumerate(coefs)
        ]
    )


ERF_COEFS = generate_erf_coefs(10)
ERFINV_COEFS = generate_erfinv_coefs(10)


@njit()
@cc.export("erf_f8", "f8(f8)")
def erf_f8(x: np.ndarray) -> np.ndarray:
    return np.dot(
        ERF_COEFS, np.array([x ** (2 * k + 1) for k in range(len(ERF_COEFS))])
    )


@njit()
@cc.export("erf_v", cc_2d_array_to_2d_array)
def erf_v(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = erf_f8(x[i, j])
    return res


@njit()
@cc.export("erfinv_f8", "f8(f8)")
def erfinv_f8(x):
    return np.dot(
        ERF_COEFS, np.array([x ** (2 * k + 1) for k in range(len(ERFINV_COEFS))])
    )


@njit()
@cc.export("erfinv_v", cc_2d_array_to_2d_array)
def erfinv_v(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    m = x.shape[1]
    res = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = erfinv_f8(x[i, j])
    return res


@njit()
@cc.export("clip_probability", cc_2d_array_to_2d_array)
def clip_probability(x: np.ndarray) -> np.ndarray:
    return np.clip(x, MACHINE_EPS, 1 - MACHINE_EPS)


@njit()
@cc.export("identity", cc_2d_array_to_2d_array)
def identity(x: np.ndarray) -> np.ndarray:
    return x


@njit()
@cc.export("logit", cc_2d_array_to_2d_array)
def logit(x: np.ndarray) -> np.ndarray:
    x = clip_probability(x)
    return logarithm(x) - logarithm(1 - x)


@njit()
@cc.export("sigmoid", cc_2d_array_to_2d_array)
def sigmoid(x: np.ndarray):
    return 1 / (1 + exponential(-x))


@njit()
@cc.export("inverse", cc_2d_array_to_2d_array)
def inverse(x: np.ndarray) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return 1 / x


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
@cc.export("inv_probit", cc_2d_array_to_2d_array)
def inv_probit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1 + erf_v(x / np.sqrt(2)))


@njit()
@cc.export("probit", cc_2d_array_to_2d_array)
def probit(x: np.ndarray) -> np.ndarray:
    return np.sqrt(2) * erfinv_v(2 * clip_probability(x) - 1)


@njit()
@cc.export("inv_probit_deriv", cc_2d_array_to_2d_array)
def inv_probit_deriv(x: np.ndarray) -> np.ndarray:
    return 1 / np.sqrt(2 * np.pi) * exponential(-0.5 * np.square(x))


@njit()
@cc.export("inv_cloglog", cc_2d_array_to_2d_array)
def inv_cloglog(x: np.ndarray) -> np.ndarray:
    return 1 - exponential(-exponential(x))


@njit()
@cc.export("cloglog", cc_2d_array_to_2d_array)
def cloglog(x: np.ndarray) -> np.ndarray:
    x = clip_probability(x)
    return logarithm(-logarithm(1 - x))


@njit()
@cc.export("inv_loglog", cc_2d_array_to_2d_array)
def inv_loglog(x: np.ndarray) -> np.ndarray:
    return exponential(-exponential(x))


@njit()
@cc.export("loglog", cc_2d_array_to_2d_array)
def loglog(x: np.ndarray) -> np.ndarray:
    return logarithm(logarithm(x))


@njit()
@cc.export("inverse_gaussian_link", cc_2d_array_to_2d_array)
def inverse_gaussian_link(x: np.array) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return 0.5 * np.power(x, -2)


@njit()
@cc.export("inverse_gaussian_inv_link", cc_2d_array_to_2d_array)
def inverse_gaussian_inv_link(x: np.array) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return np.power(2 * x, -0.5)


@njit()
@cc.export("inverse_gaussian_link_deriv", cc_2d_array_to_2d_array)
def inverse_gaussian_link_deriv(x: np.array) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return np.power(x, -3)


@njit()
@cc.export("inverse_gaussian_inv_link_deriv", cc_2d_array_to_2d_array)
def inverse_gaussian_inv_link_deriv(x: np.array) -> np.ndarray:
    x = np.sign(x) * np.clip(np.abs(x), MACHINE_EPS, MACHINE_MAX)
    return -0.5 * np.power(2 * x, -1.5)


if __name__ == "__main__":
    cc.compile()
