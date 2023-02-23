from functions import *
import numdifftools as nd

class BaseLink:
    pass

class IdentityLink(BaseLink):
    def __init__(self):
        self.inv_link = identity
        self.inv_link_deriv = differentiate(identity)

class LogitLink(BaseLink):
    def __init__(self):
        self.inv_link = sigmoid
        self.inv_link_deriv = differentiate(sigmoid)

class LogLink(BaseLink):
    def __init__(self):
        self.inv_link = exponential
        self.inv_link_deriv = differentiate(exponential)
        