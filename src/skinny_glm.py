from families import BaseFamily
import numpy as np

class SkinnyGLM():
    def __init__(self, family: BaseFamily) -> None:
        self.family = family

    
    def fit(self, X, y):
        pass

    
    def _irls(self, X, y, max_iters=100, tol=1e-4):
        # use ols values as starting values 
        W_i = np.diag(np.ones(y.shape[0]))
        beta_i = self._wols(X, y, W_i)

        for _ in range(max_iters):
            eta_i = X @ beta_i
            m_i = self.family.link.inv_link(eta_i)  # current estimate of mu
            u_i = eta_i + self.family.link.inv_link_deriv(m_i) * (y - m_i)  # working/linearized response
            W_i = np.diag((self.family.canonical_link_deriv(m_i) * self.family.link.inv_link_deriv(m_i)**2).flatten())
            delta_beta = self._wols(X, u_i, W_i) - beta_i
            beta_i += delta_beta

            if (delta_beta**2).sum() < tol: 
                break

        self.b = beta_i
        self.W = W_i


    def _wols(self, X, y, W):
        return np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y


if __name__ == "__main__":
    # from families import GaussianFamily
    # from links import IdentityLink

    # n = 100
    # p = 1
    # sigma = 0.5

    # X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    # b = np.random.normal(size=(1, p+1))
    # y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

    # skinny_model = SkinnyGLM(family=GaussianFamily(link=IdentityLink()))
    # skinny_model._irls(X, y)
    
    # print(f"true parameter estimates: {b.flatten()}")
    # print(f"skinny parameter estimates: {skinny_model.b.flatten()}")

    from families import BinomialFamily
    from links import LogitLink
    from functions import sigmoid
    import statsmodels.api as sm

    n = 1000
    p = 1

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    probs = sigmoid(X @ b.T)
    y = np.random.binomial(1, probs, (n, 1))
    

    skinny_model = SkinnyGLM(family=BinomialFamily(link=LogitLink()))
    skinny_model._irls(X, y)

    stats_model = sm.GLM(y, X, family=sm.families.Binomial(sm.genmod.families.links.logit()))
    stats_model = stats_model.fit()

    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")

    