import skinnyglms as skinny
import statsmodels.api as sm
import numpy as np
import pytest
from tests.utils import SEED, TOL, DISTRIBUTIONS, STATSMODELS_MAPPING, N_AND_P
from itertools import product

test_params = list()

for distn in DISTRIBUTIONS:
    test_params += product(
        (STATSMODELS_MAPPING[distn]['families'],),
        STATSMODELS_MAPPING[distn]['links']
    )



@pytest.mark.parametrize("n, p", N_AND_P)
@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_model_paramas_agreement(families, links, n, p):
    np.random.seed(SEED)

    skinny_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    
    b = np.random.normal(scale=0.05, size=(1, p+1))
    X = np.hstack([np.ones((n, 1)), np.random.normal(scale=0.05, size=(n, p))])
 
    mu = skinny_family.link.inv_link(X @ b.T)
    theta = skinny_family.canonical_link.link(mu)
    y = skinny_family.sample(theta)
    
    skinny_model = skinny.glm.SkinnyGLM(skinny_family)
    skinny_model._irls(X, y)

    stats_model = sm.GLM(y, X, family=sm_familiy)
    stats_model = stats_model.fit()

    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).sum() / (p+1) < TOL