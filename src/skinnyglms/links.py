from skinnyglms.functions import *
import numpy as np


class BaseLink:
    def __str__(self):
        return "BaseLink"
    
    @staticmethod
    def link(x: np.array):
        raise NotImplementedError
    @staticmethod
    def inv_link(x: np.array):
        raise NotImplementedError
    @staticmethod
    def link_deriv(x: np.array):
        raise NotImplementedError
    @staticmethod
    def inv_link_deriv(x: np.array):
        raise NotImplementedError
    

class IdentityLink(BaseLink):
    def __str__(self):
        return "IdentityLink"
    
    @staticmethod
    def link(x: np.array):
        return x
    @staticmethod
    def inv_link(x: np.array):
        return x
    @staticmethod
    def link_deriv(x: np.array):
        return np.ones(x.shape)
    @staticmethod
    def inv_link_deriv(x: np.array):
        return np.ones(x.shape)


class LogitLink(BaseLink):
    def __str__(self):
        return "LogitLink"
    
    @staticmethod
    def link(x: np.array):
        return logit(x)
    @staticmethod
    def inv_link(x: np.array):
        return sigmoid(x)
    @staticmethod
    def link_deriv(x: np.array):
        x = clip_probability(x)
        return inverse(x*(1-x))
    @staticmethod
    def inv_link_deriv(x: np.array):
        return sigmoid(x)*(1-sigmoid(x))





class LogLink(BaseLink):
    def __str__(self):
        return "LogLink"
    
    @staticmethod
    def link(x: np.array):
        return logarithm(x)
    @staticmethod
    def inv_link(x: np.array):
        return exponential(x)
    @staticmethod
    def link_deriv(x: np.array):
        return inverse(x)
    @staticmethod
    def inv_link_deriv(x: np.array):
        return exponential(x)


class NegativeInverseLink(BaseLink):
    def __str__(self):
        return "NegativeInverseLink"
    
    @staticmethod
    def link(x: np.array):
        return -inverse(x)
    @staticmethod
    def inv_link(x: np.array):
        return -inverse(x)
    @staticmethod
    def link_deriv(x: np.array):
        return inverse(np.square(x))
    @staticmethod
    def inv_link_deriv(x: np.array):
        return inverse(np.square(x))
        

# class CDFLink(BaseLink):
#     def __init__(cls, rv: rv_continuous):
#         cls.rv = rv
    
#     def link(cls, x: np.array):
#         return cls.rv.cdf(x)

#     def inv_link(cls, x: np.array):
#         x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
#         return cls.rv.ppf(x)
    
#     def link_deriv(cls, x: np.array):
#         return cls.rv.pdf(x)
    
#     def inv_link_deriv(cls, x: np.array):
#         return cls.rv.pdf(cls.rv.cdf(x))

# class ProbitLink(BaseLink):  

#     def link(cls, x: np.array):
#         x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
#         return norm.ppf(x)
      
#     def inv_link(cls, x: np.array):
#         return norm.cdf(x)
    
#     def link_deriv(cls, x: np.array):
#         x = np.clip(x, MACHINE_EPS, 1-MACHINE_EPS)
#         return 1. / norm.pdf(norm.ppf(x))
    
#     def inv_link_deriv(cls, x: np.array):
#         return norm.pdf(x)


class ProbitLink(BaseLink):  
    def __str__(self):
        return "ProbitLink"
    
    def link(cls, x: np.array):
        return probit(x)
      
    def inv_link(cls, x: np.array):
        return inv_probit(x)
    
    def link_deriv(cls, x: np.array):
        return 1. / inv_probit_deriv(probit(x))
    
    def inv_link_deriv(cls, x: np.array):
        return inv_probit_deriv(x) 
    

class InverseGaussianCanonicalLink(BaseLink):  
    def __str__(self):
        return "InverseGaussianCanonicalLink"
    
    def link(cls, x: np.array):
        return inverse_gaussian_link(x)
      
    def inv_link(cls, x: np.array):
        return inverse_gaussian_inv_link(x)
    
    def link_deriv(cls, x: np.array):
        return inverse_gaussian_link_deriv(x)
    
    def inv_link_deriv(cls, x: np.array):
        return inverse_gaussian_inv_link_deriv(x)
    


