from utils import *
from skinny_lm import SkinnyLM

class SkinnyBinomialRegressionLogitLink(SkinnyLM):
    def __init__(self):
        self.link_fn = np.vectorize(lambda mu: np.log(mu / (1 - mu)))
        self.inv_link_fn = np.vectorize(lambda eta: 1 / (1 + np.exp(-eta)))
        self.variance_fn = np.vectorize(lambda mu: mu * (1 - mu))


if __name__ == "__main__":
    n = 100
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    probs = 1 / (1 + np.exp(- X @ b.T) )
    y = np.random.binomial(1, probs, (n, 1))

    lm = SkinnyBinomialRegressionLogitLink()
    lm.fit(X, y)
    print(lm.beta.flatten(), b.flatten())