from skinnyglms.functions import *

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
        super().__init__(logit, sigmoid)

class ProbitLink(BaseLink):
    def __init__(self):
        super().__init__(probit, inv_probit)

class CLogLogLink(BaseLink):
    def __init__(self):
        super().__init__(cloglog, inv_cloglog)

class LogLink(BaseLink):
    def __init__(self):
        super().__init__(logarithm, exponential)

class NegativeInverseLink(BaseLink):
    def __init__(self):
        super().__init__(negative_inverse, negative_inverse)
        
        