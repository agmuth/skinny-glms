from utils import *
from skinny_lm import SkinnyLM

class SkinnyPoissonRegressionLogLink(SkinnyLM):
    def __init__(self):
        self.theta_of_mu = lambda mu: np.log(mu)
        self.mu_of_eta = lambda eta: np.exp(eta)
    
    def fit(self, X, y, offset=None):
        if offset is not None:
            self.mu_of_eta = lambda eta: offset * np.exp(eta)
        self._iteratively_reweighted_least_squares(X, y)



if __name__ == "__main__":
    import time
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    offset = np.random.randint(1, 2, (n, 1))
    lam = offset * np.exp(X @ b.T)
    y = np.random.poisson(lam, (n, 1))

    skinny_model = SkinnyPoissonRegressionLogLink()
    tic1 = time.time()
    skinny_model.fit(X, y)
    toc1 = time.time()

    import statsmodels.api as sm
    stats_model = sm.GLM(y, X, family=sm.families.Poisson(sm.genmod.families.links.log()))
    tic2 = time.time()
    stats_model = stats_model.fit()
    toc2 = time.time()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()} fitting seconds: {toc1-tic1}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()} fitting seconds: {toc2-tic2}")