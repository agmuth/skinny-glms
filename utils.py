import numpy as np
from scipy.special import erf

MACHINE_EPS = np.finfo(np.float64).eps

safe_division = lambda x, y: x / y if np.abs(y) > MACHINE_EPS else np.sign(y) * x / MACHINE_EPS

inv_probit = lambda x: 0.5 * (1 + erf(x / np.sqrt(2)))
inv_cloglog = lambda x: 1 - np.exp(-np.exp(x))
inv_loglog = lambda x: np.exp(-np.exp(x))