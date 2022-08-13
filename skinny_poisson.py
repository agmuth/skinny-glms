from utils import *
from skinny_lm import SkinnyLM

class SkinnyPoissonRegressionLogLink(SkinnyLM):
    def __init__(self):
        super().__init__()
        self.theta_of_mu = np.vectorize(lambda mu: np.log(mu))
        self.mu_of_eta = np.vectorize(lambda eta: np.exp(eta))
        


if __name__ == "__main__":
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    lam = np.exp(X @ b.T)
    y = np.random.poisson(lam, (n, 1))

    lm = SkinnyPoissonRegressionLogLink()
    lm.fit(X, y)
    print(lm.b.flatten())
    print(b.flatten())