from utils import *
from skinny_lm import SkinnyLM

class SkinnyGammaRegressionLogLink(SkinnyLM):
    def __init__(self):
        self.theta_of_mu = lambda mu: -mu**-1
        self.mu_of_eta = lambda eta: np.exp(eta)  # need to figure out why signs of b estimate are reversed

class SkinnyGammaRegressionInverseLink(SkinnyLM):
    def __init__(self):
        self.theta_of_mu = lambda mu: -mu**-1
        self.mu_of_eta = lambda eta: np.exp(eta)
    



if __name__ == "__main__":
    import time
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    
    rate = np.exp(-1 * X @ b.T)
    y = np.random.gamma(1, 1/rate, (n, 1))

    skinny_model = SkinnyGammaRegressionLogLink()
    tic1 = time.time()
    skinny_model.fit(X, y)
    toc1 = time.time()

    import statsmodels.api as sm
    stats_model = sm.GLM(y, X, family=sm.families.Gamma(sm.genmod.families.links.log()))
    tic2 = time.time()
    stats_model = stats_model.fit()
    toc2 = time.time()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()} fitting seconds: {toc1-tic1}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()} fitting seconds: {toc2-tic2}")