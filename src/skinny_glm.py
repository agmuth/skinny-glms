from families import BaseFamily
import numpy as np

class SkinnyGLM():
    def __init__(self, family: BaseFamily) -> None:
        self.family = family

    
    def fit(self, X, y):
        pass

    
    def _irls(self, X, y, max_iters=int(1e4), tol=1e-4):
        # use ols values as starting values 
        W_i = np.diag(np.ones(y.shape[0]))
        beta_i = self._wols(X, y, W_i)

        for i in range(max_iters):
            eta_i = X @ beta_i
            m_i = self.family.link.inv_link(eta_i)  # current estimate of mu
            u_i = eta_i + self.family.link.link_deriv(m_i) * (y - m_i)  # working/linearized response
            W_i = np.diag((self.family.inv_variance(m_i) * self.family.link.inv_link_deriv(eta_i)**2).flatten())
            delta_beta = self._wols(X, u_i, W_i) - beta_i
            beta_i += delta_beta

            if (delta_beta**2).sum() < tol: 
                break

        self.b = beta_i
        self.W = W_i


    def _wols(self, X, y, W):
        return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

