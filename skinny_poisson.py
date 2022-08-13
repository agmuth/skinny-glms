from utils import *
from skinny_lm import SkinnyLM

class SkinnyPoissonRegressionLogLink(SkinnyLM):
    def __init__(self):
        self.link_fn = np.vectorize(lambda mu: np.log(mu))
        self.inv_link_fn = np.vectorize(lambda eta: np.exp(eta))
        self.variance_fn = np.vectorize(lambda mu: mu)


if __name__ == "__main__":
    pass