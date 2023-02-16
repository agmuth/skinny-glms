import numpy as np
import numdifftools as nd

class SkinnyLM():
    def __init__(self):
        self.theta_of_mu = lambda mu: mu  # maps theta as a function of mu
        self.mu_of_eta = lambda eta: eta  # maps mu as a function of eta


    def fit(self, X, y):
        self._iteratively_reweighted_least_squares(X, y)
        # TODO: add in model specific params here - maybe

    def _iteratively_reweighted_least_squares(self, X, y, tol=1e-4, max_iters=100):
        d_theta_d_mu = nd.Derivative(self.theta_of_mu)
        d_mu_d_eta = nd.Derivative(self.mu_of_eta)

        b = np.linalg.inv(X.T @ X) @ X.T @ y
        W = None

        for i in range(max_iters):
            eta_i = X @ b
            mu_i = self.mu_of_eta(eta_i)
            u_i = eta_i + (y - mu_i) * d_mu_d_eta(eta_i)**-1
            W = np.diag((d_theta_d_mu(mu_i) * d_mu_d_eta(eta_i)**2).flatten())

            delta_b = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ u_i - b
            b += delta_b

            if (delta_b**2).sum() < tol:
                break

        self.b = b
        self.W = W






if __name__ == "__main__":
    import time
    # square = lambda x: x**2
    # square_dx = differentiate(square)
    # print(square_dx(1))

    n = 100
    p = 1
    sigma = 0.5

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

    skinny_model = SkinnyLM()
    tic1 = time.time()
    skinny_model.fit(X, y)
    toc1 = time.time()
    import statsmodels.api as sm
    stats_model = sm.GLM(y, X, family=sm.families.Gaussian())
    tic2 = time.time()
    stats_model = stats_model.fit()
    toc2 = time.time()
    
    print(f"true parameter estimates: {b.flatten()}")
    print(f"skinny parameter estimates: {skinny_model.b.flatten()} fitting seconds: {toc1-tic1}")
    print(f"statsmodels parameter estimates: {stats_model.params.flatten()} fitting seconds: {toc2-tic2}")
    