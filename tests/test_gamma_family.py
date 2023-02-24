import numpy as np
from skinny_glm import SkinnyGLM
from families import GammaFamily
from links import *
import statsmodels.api as sm
import pytest

TOL = 1e-4
SEED = 1234

LINKS = [
    (LogLink(), sm.genmod.families.links.log())
]

@pytest.mark.parametrize("links", LINKS)
def test_gamma(links):
    np.random.seed(SEED)
    skinny_link = links[0]
    sm_link = links[1]

    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    rate = skinny_link.inv_link(X @ b.T)
    y = np.random.gamma(1, 1/rate, (n, 1))
    
    skinny_model = SkinnyGLM(family=GammaFamily(skinny_link))
    skinny_model._irls(X, y)

    stats_model = sm.GLM(y, X, family=sm.families.Gamma(sm_link))
    stats_model = stats_model.fit()

    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).sum() < TOL