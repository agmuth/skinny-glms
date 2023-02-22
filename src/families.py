from functions import * 
from links import BaseLink
import numdifftools as nd

class BaseFamily:
    def __init__(self, link:BaseLink):
        self.link = link

class GaussianFamily(BaseFamily):
    def __init__(self, link):
        super().__init__(link)
        self.canonical_link = identity
        self.canonical_link_deriv = nd.Derivative(identity)

class BinomialFamily(BaseFamily):
    def __init__(self, link: BaseLink):
        super().__init__(link)
        self.canonical_link = logit
        self.canonical_link_deriv = nd.Derivative(logit)  