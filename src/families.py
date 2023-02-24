from functions import * 
from links import BaseLink

class BaseFamily:
    def __init__(self, link:BaseLink, canonical_link: callable=identity):
        self.link = link
        self.canonical_link = canonical_link
        self.canonical_link_deriv = differentiate(self.canonical_link)

    def inv_variance(self, mu: np.ndarray) -> np.ndarray:
        return self.canonical_link_deriv(mu)


class GaussianFamily(BaseFamily):
    def __init__(self, link):
        super().__init__(link, canonical_link = identity)

class BinomialFamily(BaseFamily):
    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link = logit)

class GammaFamily(BaseFamily):
    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link = negative_inverse)