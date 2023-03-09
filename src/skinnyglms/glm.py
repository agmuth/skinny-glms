from skinnyglms.families import *
from skinnyglms.links import *
import numpy as np

class SkinnyGLM():
    def __init__(self, family: BaseFamily) -> None:
        self.family = family

    
    def fit(self, X, y):
        pass

    
    def _irls(self, X, y, offset=None, max_iters=int(1e4), tol=1e-4):

        if offset is None:
            eta_scale_offsets = np.zeros(y.shape)
        else:
            eta_scale_offsets = self.family.link.link(offset)

        
        eta_i = np.empty((X.shape[0], 1))
        m_i = np.empty((y.shape))
        u_i = np.empty(y.shape)
        W_i = np.ones((X.shape[0], 1))
        beta_i = self._wols(X, y, W_i)

        self.iter = 1
        if not (isinstance(self.family, GaussianFamily) and isinstance(self.family.link, IdentityLink)):

            while self.iter < max_iters:
                self.iter += 1
                # eta_i = X @ beta_i
                np.matmul(X, beta_i, out=eta_i)
                m_i = self.family.link.inv_link(eta_i + eta_scale_offsets) # current estimate of mu
                # u_i = eta_i + self.family.link.link_deriv(m_i) * (y - m_i)  # working/linearized response

                np.add(
                    eta_i, 
                    np.multiply(
                        self.family.link.link_deriv(m_i),
                        np.subtract(y, m_i)
                    ),
                    out=u_i
                )

                np.multiply(
                        self.family.inv_variance(m_i), 
                        np.square(self.family.link.inv_link_deriv(eta_i)),
                        out=W_i
                    )

                # W_i =  np.multiply(
                #         self.family.inv_variance(m_i), 
                #         np.square(self.family.link.inv_link_deriv(eta_i))
                #     )
                delta_beta = self._wols(X, u_i, W_i) - beta_i
                
                if (delta_beta**2).sum() < tol: 
                    break
                else:
                    beta_i += delta_beta
                    # m_i.fill(0.)


        # save vars 
        self.b = beta_i
        self.W = W_i.flatten()
        self.df = y.shape[0] - X.shape[1] - 1        
        m_i = self.family.link.inv_link(X @ beta_i + eta_scale_offsets) 
        self.dispersion = np.sum(np.square(y - m_i) * self.family.inv_variance(m_i)) / self.df # empirical estimate of dispersion


    def _wols(self, X, y, W):
        # return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y
        A = np.multiply(X, W).T
        b = np.linalg.solve(A @ X, A @ y)
        return b

