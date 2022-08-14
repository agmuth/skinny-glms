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
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    offset = np.random.randint(1, 10, (n, 1))
    lam = offset * np.exp(X @ b.T)
    y = np.random.poisson(lam, (n, 1))

    lm = SkinnyPoissonRegressionLogLink()
    lm.fit(X, y, offset)
    print(lm.b.flatten())
    print(b.flatten())