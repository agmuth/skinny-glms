from typing import Union

from skinnyglms.links import *


class BaseFamily:
    def __str__(self):
        return "BaseFamily"

    def __init__(self, link: BaseLink, canonical_link: BaseLink):
        self.link = link
        self.canonical_link = canonical_link

    def inv_variance(self, mu: np.ndarray) -> np.ndarray:
        return self.canonical_link.link_deriv(mu)

    def sample(self, *args, **kwargs):
        raise NotImplementedError


class GaussianFamily(BaseFamily):
    def __str__(self):
        return "GaussianFamily"

    def __init__(self, link):
        super().__init__(link, canonical_link=IdentityLink())

    def sample(self, theta: np.ndarray, phi: Union[int, np.ndarray] = 1, n: int = 1):
        loc = self.canonical_link.inv_link(theta)
        scale = phi
        return np.hstack(
            [
                np.random.normal(loc=loc, scale=scale, size=(theta.shape[0], 1))
                for _ in range(n)
            ]
        )


class BinomialFamily(BaseFamily):
    def __str__(self):
        return "BinomialFamily"

    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link=LogitLink())

    def sample(self, theta: np.ndarray, phi: Union[int, np.ndarray] = 1, n: int = 1):
        p = self.canonical_link.inv_link(theta)
        return np.hstack(
            [np.random.binomial(n=1, p=p, size=(theta.shape[0], 1)) for _ in range(n)]
        )


class PoissonFamily(BaseFamily):
    def __str__(self):
        return "PoissonFamily"

    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link=LogLink())

    def sample(self, theta: np.ndarray, phi: Union[int, np.ndarray] = 1, n: int = 1):
        lam = self.canonical_link.inv_link(theta)
        return np.hstack(
            [np.random.poisson(lam=lam, size=(theta.shape[0], 1)) for _ in range(n)]
        )


class GammaFamily(BaseFamily):
    def __str__(self):
        return "GammaFamily"

    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link=NegativeInverseLink())

    def sample(self, theta: np.ndarray, phi: Union[int, np.ndarray] = 1, n: int = 1):
        scale = self.canonical_link.inv_link(theta)
        shape = phi
        return np.hstack(
            [
                np.random.gamma(shape=shape, scale=scale, size=(theta.shape[0], 1))
                for _ in range(n)
            ]
        )


class InverseGaussianFamily(BaseFamily):
    def __str__(self):
        return "InverseGaussianFamily"

    def __init__(self, link: BaseLink):
        super().__init__(link, canonical_link=InverseGaussianCanonicalLink())

    def sample(self, theta: np.ndarray, phi: Union[int, np.ndarray] = 1, n: int = 1):
        mean = self.canonical_link.inv_link(theta)
        scale = phi
        return np.hstack(
            [
                np.random.wald(mean=mean, scale=scale, size=(theta.shape[0], 1))
                for _ in range(n)
            ]
        )
