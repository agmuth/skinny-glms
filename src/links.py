from functions import *
import numdifftools as nd

class BaseLink:
    def __init__(self, link: callable, inv_link: callable):
        self.link = link
        self.inv_link = inv_link
        self.link_deriv = differentiate(link)
        self.inv_link_deriv = differentiate(inv_link)
    

class IdentityLink(BaseLink):
    def __init__(self):
        super().__init__(identity, identity)
       

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
        