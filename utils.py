import numpy as np

def differentiate(f, h=1e-4):
    h_inv = 1/h
    f_dx = np.vectorize(lambda x: 0.5 * h_inv * (f(x + h) - f(x - h)))
    return f_dx

