from skinnyglms.functions import *

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
        return inverse(x*(1-x))
    @staticmethod
    def inv_link_deriv(x: np.array):
        return sigmoid(x)*(1-sigmoid(x))


# class ProbitLink(BaseLink):
#     def __init__(self):
#         super().__init__(probit, inv_probit)


# class CLogLogLink(BaseLink):
#     def __init__(self):
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
        
        