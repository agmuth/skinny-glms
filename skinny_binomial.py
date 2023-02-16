from utils import *
from skinny_lm import SkinnyLM


class SkinnyBinomialRegressionLogitLink(SkinnyLM):
    def __init__(self):
        super().__init__()
        self.theta_of_mu = lambda mu: np.log(safe_division(mu , (1 - mu)))
        self.mu_of_eta = lambda eta: 1 / (1 + np.exp(-eta))


class SkinnyBinomialRegressionProbitLink(SkinnyBinomialRegressionLogitLink):
    def __init__(self):
        super().__init__()
        self.mu_of_eta = lambda eta: inv_probit(eta)


class SkinnyBinomialRegressionComplementaryLogLogLink(SkinnyBinomialRegressionLogitLink):
    def __init__(self):
        super().__init__()
        self.mu_of_eta = lambda eta: inv_cloglog(eta)


class SkinnyBinomialRegressionLogLogLink(SkinnyBinomialRegressionLogitLink):
    def __init__(self):
        super().__init__()
        self.mu_of_eta = lambda eta: inv_loglog(eta)

if __name__ == "__main__":
    import time
    # n = 1000
    # p = 1

    # X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    # b = np.random.normal(size=(1, p+1))
    # probs = 1 / (1 + np.exp(- X @ b.T) )
    # y = np.random.binomial(1, probs, (n, 1))

    # lm = SkinnyBinomialRegressionLogitLink()
    # lm.fit(X, y)
    # print(lm.b.flatten())
    # print(b.flatten())


    n = 10000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    probs = inv_probit(X @ b.T)
    y = np.random.binomial(1, probs, (n, 1))

    skinny_model = SkinnyBinomialRegressionProbitLink()
    tic1 = time.time()
    skinny_model.fit(X, y)
    toc1 = time.time()

    import statsmodels.api as sm
    stats_model = sm.GLM(y, X, family=sm.families.Binomial(sm.genmod.families.links.probit()))
    
    tic2 = time.time()
    stats_model = stats_model.fit()
    toc2 = time.time()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()} fitting seconds: {toc1-tic1}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()} fitting seconds: {toc2-tic2}")