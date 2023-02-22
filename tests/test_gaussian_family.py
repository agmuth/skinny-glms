import numpy as np
from skinny_glm import SkinnyGLM
from families import GaussianFamily
from links import IdentityLink
import statsmodels.api as sm

TOL = 1e-4

def test_gaussian():
    n = 100
    p = 1
    sigma = 0.5

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

    skinny_model = SkinnyGLM(family=GaussianFamily(link=IdentityLink()))
    skinny_model._irls(X, y)
    stats_model = sm.GLM(y, X, family=sm.families.Gaussian())
    stats_model = stats_model.fit()

    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).sum() < TOL