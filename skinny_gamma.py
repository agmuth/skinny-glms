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
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    
    rate = np.exp(-1 * X @ b.T)
    y = np.random.gamma(1, 1/rate, (n, 1))

    skinny_model = SkinnyGammaRegressionLogLink()
    skinny_model.fit(X, y)

    import statsmodels.api as sm
    stats_model = sm.GLM(y, X, family=sm.families.Gamma(sm.genmod.families.links.log())).fit()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")