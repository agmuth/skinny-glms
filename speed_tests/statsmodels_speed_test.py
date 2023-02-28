import skinnyglms as skinny
from skinnyglms.mappings.statsmodels import STATSMODELS_MAPPING
import statsmodels.api as sm
import numpy as np
import timeit


SEED = 2023
DISTRIBUTIONS = ["GAUSSIAN", "BINOMIAL", "GAMMA", "POISSON"]


def speed_test(families, links, n, p):
    np.random.seed(SEED)

    skinny_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    
    b = np.random.normal(size=(1, p+1))
    X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
 
    mu = skinny_family.link.inv_link(X @ b.T)
    theta = skinny_family.canonical_link.link(mu)
    y = skinny_family.sample(theta)
    repeats = 10
    number = 10

    skinny_model = skinny.glm.SkinnyGLM(family=skinny.families.GaussianFamily(link=skinny.links.IdentityLink()))
    stats_model = sm.GLM(y, X, family=sm.families.Gaussian(sm.genmod.families.links.identity()))
    skinny_time = min(timeit.repeat("skinny_model._irls(X, y)", repeat=repeats, number=number, globals=locals())) / number
    sm_time = min(timeit.repeat("stats_model.fit()", repeat=repeats, number=number, globals=locals())) / number
    res = {
        "family" :  families[0],
        "link" : links[0],
        "skinnyGLM_time": skinny_time,
        "statmodels_time": sm_time
    }
    return res

if __name__ == "__main__":

    for distn in DISTRIBUTIONS:
        families = STATSMODELS_MAPPING[distn]['families']
        for links in STATSMODELS_MAPPING[distn]['links']:
            print(speed_test(families, links, 1000, 1))