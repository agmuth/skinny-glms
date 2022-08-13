from utils import *

class SkinnyLM():
    def __init__(self):
        self.link_fn = np.vectorize(lambda mu: mu)  # maps mu to eta
        self.inv_link_fn = np.vectorize(lambda eta: eta)

        self.d_eta_d_mu_fn = differentiate(self.link_fn)
        self.variance_fn = np.vectorize(lambda mu: 1.0)


    def fit(self, X, y):
        self._iteratively_reweighted_least_squares(X, y)
        # TODO: add in model specific params here - maybe

    def _iteratively_reweighted_least_squares(self, X, y, tol=1e-4, max_iters=100):
        d_eta_d_mu_fn = differentiate(self.link_fn)

        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        W_inv = None

        for i in range(max_iters):
            eta_hat_i = X @ beta
            mu_hat_i = self.inv_link_fn(eta_hat_i)
            z_i = eta_hat_i + (y - mu_hat_i) * d_eta_d_mu_fn(mu_hat_i)
            W_inv = np.diag((d_eta_d_mu_fn(mu_hat_i)**2 * self.variance_fn(mu_hat_i)).flatten())
            delta_beta = np.linalg.inv(X.T @ W_inv @ X) @ X.T @ W_inv @ z_i - beta
            beta += delta_beta

            if (delta_beta**2).sum() < tol:
                break

        self.beta = beta
        self.W_inv = W_inv






if __name__ == "__main__":

    # square = lambda x: x**2
    # square_dx = differentiate(square)
    # print(square_dx(1))

    n = 100
    p = 1
    sigma = 0.5

    X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
    b = np.random.normal(size=(1, p+1))
    y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

    lm = SkinnyLM()
    lm.fit(X, y)
    print(lm.beta, b)