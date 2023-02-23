import numpy as np
from skinny_glm import SkinnyGLM
from families import GammaFamily
from links import LogLink
import statsmodels.api as sm

TOL = 1e-4

def test_gamma_log_link():
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    
    rate = np.exp(-1 * X @ b.T)
    y = np.random.gamma(1, 1/rate, (n, 1))

    skinny_model = SkinnyGLM(family=GammaFamily(link=LogLink()))
    skinny_model._irls(X, y)

    stats_model = sm.GLM(y, X, family=sm.families.Gamma(sm.genmod.families.links.log()))
    stats_model = stats_model.fit()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
    assert ((skinny_model.b.flatten() - stats_model.params.flatten())**2).mean() < TOL