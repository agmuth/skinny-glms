from functions import *
import numdifftools as nd

class BaseLink:
    pass

class IdentityLink(BaseLink):
    def __init__(self):
        self.link = identity
        self.link_deriv = differentiate(identity)
        self.inv_link = identity
        self.inv_link_deriv = differentiate(identity)

class LogitLink(BaseLink):
    def __init__(self):
        self.link = logit
        self.link_deriv = differentiate(logit)
        self.inv_link = sigmoid
        self.inv_link_deriv = differentiate(sigmoid)

class LogLink(BaseLink):
    def __init__(self):
        self.link = log
        self.link_deriv = differentiate(log)
        self.inv_link = exponential
        self.inv_link_deriv = differentiate(exponential)
        