import skinnyglms as skinny
from skinnyglms.mappings.statsmodels import STATSMODELS_MAPPING
import statsmodels.api as sm
import numpy as np
import pytest
from tests.utils import SEED, TOL

DISTRIBUTIONS = ["GAUSSIAN", "BINOMIAL", "GAMMA", "POISSON"]

test_params = list()

for distn in DISTRIBUTIONS:
    families = STATSMODELS_MAPPING[distn]['families']
    for links in STATSMODELS_MAPPING[distn]['links']:
        test_params.append((families, links))

@pytest.mark.parametrize("n, p", [(100, 1), (100, 2)])
@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement(families, links, n, p):
    np.random.seed(SEED)

    skinny_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    
    b = np.random.normal(size=(1, p+1))
    X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
 
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
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).sum() < TOL