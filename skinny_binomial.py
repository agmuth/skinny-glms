from utils import *
from skinny_lm import SkinnyLM

class SkinnyBinomialRegressionLogitLink(SkinnyLM):
    def __init__(self):
        super().__init__()
        self.theta_of_mu = np.vectorize(lambda mu: np.log(mu / (1 - mu)))
        self.mu_of_eta = np.vectorize(lambda eta: 1 / (1 + np.exp(-eta)))


if __name__ == "__main__":
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    probs = 1 / (1 + np.exp(- X @ b.T) )
    y = np.random.binomial(1, probs, (n, 1))

    lm = SkinnyBinomialRegressionLogitLink()
    lm.fit(X, y)
    print(lm.b.flatten())
    print(b.flatten())