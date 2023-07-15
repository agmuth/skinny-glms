from typing import Optional

import numpy as np

from skinnyglms.families import *
from skinnyglms.links import *


class SkinnyGLM:
    """Bare Bones GLM fitter."""

    def __init__(self, family: BaseFamily):
        """init

        Parameters
        ----------
        family : BaseFamily
            Instanciated `BaseFamily` object.
        """
        self.family = family

    def fit(self, X: np.array, y: np.array, *args, **kwargs):
        return self._irls(X, y, *args, **kwargs)

    def _irls(
        self,
        X: np.array,
        y: np.array,
        offset: Optional[np.array] = None,
        count_weights: Optional[np.array] = None,
        var_weights: Optional[np.array] = None,
        max_iters: Optional[int] = int(1e2),
        tol: Optional[float] = 1e-4,
    ):
        """_summary_

        Parameters
        ----------
        X : np.array
            (n, p) covariate matrix
        y : np.array
            (n, 1) dependent variable vector
        offset : Optional[np.array], optional
            (n, 1) offset vector, by default None
        count_weights : Optional[np.array], optional
            (n, 1) repeated measure vector only one of `count_weights` and `var_weights` should be specified, by default None
        var_weights : Optional[np.array], optional
            (n, 1) heteroskedasticity vector only one of `count_weights` and `var_weights` should be specified, by default None
        max_iters : Optional[int], optional
            maximum number of irls iterations, by default int(1e2)
        tol : Optional[float], optional
            tolerance criteria for L2 change in coefficient vector, by default 1e-4
        """
        if offset is None:
            eta_scale_offsets = np.zeros(y.shape, dtype=float)
        else:
            eta_scale_offsets = self.family.link.link(offset)

        if count_weights is None:
            count_weights = np.ones(y.shape, dtype=float)

        if var_weights is None:
            var_weights = np.ones(y.shape, dtype=float)
        var_weights_inv = 1.0 / np.multiply(count_weights, var_weights)

        eta_i = np.empty((X.shape[0], 1))
        m_i = np.empty((y.shape))
        u_i = np.empty(y.shape)
        W_i = np.multiply(count_weights, var_weights)
        beta_i = self._wols(X, y, W_i)

        self.iter = 1
        if not (
            isinstance(self.family, GaussianFamily)
            and isinstance(self.family.link, IdentityLink)
        ):
            # main irls loop
            while self.iter < max_iters:
                self.iter += 1
                np.matmul(X, beta_i, out=eta_i)
                np.multiply(
                    count_weights,
                    self.family.link.inv_link(eta_i + eta_scale_offsets),
                    out=m_i,
                )
                np.add(
                    eta_i,
                    np.multiply(self.family.link.link_deriv(m_i), np.subtract(y, m_i)),
                    out=u_i,
                )
                np.multiply(
                    np.multiply(self.family.inv_variance(m_i), var_weights_inv),
                    np.square(self.family.link.inv_link_deriv(eta_i)),
                    out=W_i,
                )

                delta_beta = self._wols(X, u_i, W_i) - beta_i

                if np.linalg.norm(delta_beta) < tol:
                    break
                else:
                    beta_i += delta_beta

        # save vars
        self.b = beta_i
        self.W = W_i.flatten()
        self.df = y.shape[0] - X.shape[1] - 1
        m_i = self.family.link.inv_link(X @ beta_i + eta_scale_offsets)
        self.dispersion = (
            np.sum(np.square(y - m_i) * self.family.inv_variance(m_i) * var_weights_inv)
            / self.df
        )  # empirical estimate of dispersion

    def _wols(self, X: np.array, y: np.array, W: np.array):
        """Weighted ordinary least squares.

        Parameters
        ----------
        X : np.array
            Covariates.
        y : np.array
            Dependent variable.
        W : np.array
            Weights.

        Returns
        -------
        _type_
            _description_
        """
        # formulate as linear systems to solve for speed + precision
        A = np.multiply(X, W).T
        b = np.linalg.solve(A @ X, A @ y)
        return b
