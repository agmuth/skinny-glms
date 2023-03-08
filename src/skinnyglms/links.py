from skinnyglms.functions import *
from scipy.stats._distn_infrastructure import rv_continuous
from scipy.stats import (
    norm
)


class BaseLink:
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


# class ProbitLink(BaseLink):
#     def __init__(cls):
#         super().__init__(probit, inv_probit)


# class CLogLogLink(BaseLink):
#     def __init__(cls):
#         super().__init__(cloglog, inv_cloglog)


class LogLink(BaseLink):
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

def cdf_link_factory(rv):
    class CDFLink(BaseLink):
        
        @staticmethod
        def link(x: np.array):
            x = clip_probability(x)
            return rv.ppf(x)
        
        @staticmethod        
        def inv_link(x: np.array):
            return rv.cdf(x)
        
        @staticmethod        
        def link_deriv(x: np.array):
            x = clip_probability(x)
            return 1. / rv.pdf(rv.ppf(x))
            

        @staticmethod        
        def inv_link_deriv(x: np.array):
            return rv.pdf(x)
        
    return CDFLink  

ProbitLink = cdf_link_factory(norm)