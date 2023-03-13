import skinnyglms as skinny
from skinnyglms.mappings.sklearn import SKLEARN_MAPPING
import sklearn.linear_model as lm
import numpy as np
import timeit


SEED = 2023
DISTRIBUTIONS = ["GAUSSIAN", "BINOMIAL", "GAMMA", "POISSON"]


def speed_test(families, links, n, p):
    np.random.seed(SEED)

    skinny_family = families[0](links[0]())
    sk_model = families[1](fit_intercept=False)
    
    b = np.random.normal(scale=0.1, size=(1, p+1))
    X = np.hstack([np.ones((n, 1)), np.random.normal(scale=0.1, size=(n, p))])
 
    mu = skinny_family.link.inv_link(X @ b.T)
    theta = skinny_family.canonical_link.link(mu)
    y = skinny_family.sample(theta)
    y_flat = y.flatten()
    repeats = 10
    number = 10
    micro = 1e6
    skinny_model = skinny.glm.SkinnyGLM(family=skinny_family)
    skinny_time = min(timeit.repeat("skinny_model._irls(X, y)", repeat=repeats, number=number, globals=locals())) / number * micro
    sk_time = min(timeit.repeat("sk_model.fit(X, y_flat)", repeat=repeats, number=number, globals=locals())) / number * micro
    res = {
        "family" :  families[0],
        "link" : links[0],
        "skinnyGLM_time": int(skinny_time),
        "sklearn_time": int(sk_time)
    }
    return res

if __name__ == "__main__":

    for distn in DISTRIBUTIONS:
        families = SKLEARN_MAPPING[distn]['families']
        for links in SKLEARN_MAPPING[distn]['links']:
            print(families[0], links[0])
            print(speed_test(families, links, 100000, 100))
            print()