import skinnyglms as skinny
import statsmodels.api as sm
from skinnyglms.mappings.statsmodels import STATSMODELS_MAPPING
import numpy as np
import pytest
from tests.utils import SEED, TOL

FAMILIES = STATSMODELS_MAPPING['GAMMA']['families']
LINKS = STATSMODELS_MAPPING['GAMMA']['links']


@pytest.mark.parametrize("links", LINKS)
def test_gaussian(links):
    np.random.seed(SEED)

    skinny_family = FAMILIES[0]
    sm_familiy = FAMILIES[1]

    skinny_link = links[0]
    sm_link = links[1]

    n = 1000
    p = 1

    X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    offset = np.random.randint(1, 2, (n, 1))
    lam = offset * np.exp(X @ b.T)
    y = np.random.poisson(lam, (n, 1))

    skinny_model = skinny.glm.SkinnyGLM(family=skinny_family(skinny_link))
    skinny_model._irls(X, y)

    stats_model = sm.GLM(y, X, family=sm_familiy(sm_link))
    stats_model = stats_model.fit()

    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).sum() < TOL