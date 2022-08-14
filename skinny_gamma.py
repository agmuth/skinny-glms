from utils import *
from skinny_lm import SkinnyLM

class SkinnyGammaRegressionLogLink(SkinnyLM):
    def __init__(self):
        self.theta_of_mu = lambda mu: mu**-1
        self.mu_of_eta = lambda eta: np.exp(eta)  # need to figure out why signs of b estimate are reversed
    



if __name__ == "__main__":
    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    
    rate = np.exp(X @ b.T)
    y = np.random.gamma(1, 1/rate, (n, 1))

    lm = SkinnyGammaRegressionLogLink()
    lm.fit(X, y)
    print(lm.b.flatten())
    print(b.flatten())