from skinnyglms.families import *
from skinnyglms.links import *
import numpy as np

class SkinnyGLM():
    def __init__(self, family: BaseFamily) -> None:
        self.family = family

    
    def fit(self, X, y):
        pass

    
    def _irls(self, X, y, offset=None, count_weights=None, var_weights=None, max_iters=int(1e2), tol=1e-4):

        if offset is None:
            eta_scale_offsets = np.zeros(y.shape, dtype=float)
        else:
            eta_scale_offsets = self.family.link.link(offset)

        if count_weights is None:
            count_weights = np.ones(y.shape, dtype=float)

        if var_weights is None:
            var_weights = np.ones(y.shape, dtype=float)
        var_weights_inv = 1. / np.multiply(count_weights, var_weights)

        eta_i = np.empty((X.shape[0], 1))
        m_i = np.empty((y.shape))
        u_i = np.empty(y.shape)
        W_i = np.multiply(count_weights, var_weights)
        beta_i = self._wols(X, y, W_i)

        self.iter = 1
        if not (isinstance(self.family, GaussianFamily) and isinstance(self.family.link, IdentityLink)):

            while self.iter < max_iters:
                self.iter += 1
                np.matmul(X, beta_i, out=eta_i)
                np.multiply(
                    count_weights,
                    self.family.link.inv_link(eta_i + eta_scale_offsets),
                    out=m_i
                )
                np.add(
                    eta_i, 
                    np.multiply(
                        self.family.link.link_deriv(m_i),
                        np.subtract(y, m_i)
                    ),
                    out=u_i
                )
                np.multiply(
                        np.multiply(self.family.inv_variance(m_i), var_weights_inv), 
                        np.square(self.family.link.inv_link_deriv(eta_i)),
                        out=W_i
                    )

                delta_beta = self._wols(X, u_i, W_i) - beta_i
                
                if (delta_beta**2).sum() < tol: 
                    break
                else:
                    beta_i += delta_beta


        # save vars 
        self.b = beta_i
        self.W = W_i.flatten()
        self.df = y.shape[0] - X.shape[1] - 1        
        m_i = self.family.link.inv_link(X @ beta_i + eta_scale_offsets) 
        self.dispersion = np.sum(np.square(y - m_i) * self.family.inv_variance(m_i) * var_weights_inv) / self.df # empirical estimate of dispersion


    def _wols(self, X, y, W):
        # return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        A = np.multiply(X, W).T
        b = np.linalg.solve(A @ X, A @ y)
        return b

