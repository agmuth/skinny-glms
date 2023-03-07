import skinnyglms as skinny
import statsmodels.api as sm
import numpy as np
skinny_link = skinny.links.LogitLink()

np.random.seed(9876)
n = 1000
p = 100

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
probs = skinny_link.inv_link(X @ b.T)
y = np.random.binomial(1, probs, (n, 1))
max_iter = 3
skinny_model = skinny.glm.SkinnyGLM(family=skinny.families.BinomialFamily(skinny_link))
skinny_model._irls(X, y, max_iters=max_iter)
stats_irls = sm.GLM(y, X, sm.families.Binomial(sm.genmod.families.links.Logit()))
stats_model = stats_irls.fit(maxiter=max_iter)

m=2
print(f"true parameter estimates: {b.flatten()[:m]}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()[:m]}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()[:m]}")

beta_i = skinny_model.b
eta_i = X @ beta_i
m_i = skinny_model.family.link.inv_link(eta_i)
u_i = eta_i + skinny_model.family.link.link_deriv(m_i) * (y - m_i)  # working/linearized response
w_i =  np.multiply(
        skinny_model.family.inv_variance(m_i), 
        np.square(skinny_model.family.link.inv_link_deriv(eta_i))
    )

beta_j = stats_model.params
eta_j = X @ beta_j
m_j = stats_irls.family.fitted(eta_j)
u_j = eta_j + stats_irls.family.link.deriv(m_j) * (y - m_j)
w_j = stats_irls.family.weights(m_j)



print(np.square(beta_i - beta_j).mean())
