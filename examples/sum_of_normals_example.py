import numpy as np
import statsmodels.api as sm

import skinnyglms as skinny

np.random.seed(1234)

n = 100
p = 2
sigma = 0.5

counts = np.random.poisson(lam=1, size=(n, 1))
counts[counts == 0] += 1

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p + 1))
mu = X @ b.T

y = np.random.normal(counts * mu, np.sqrt(counts) * sigma)


skinny_model = skinny.glm.SkinnyGLM(
    family=skinny.families.GaussianFamily(link=skinny.links.IdentityLink())
)
skinny_model._irls(X, y, var_weights=counts / 1.0)

stats_model = sm.GLM(
    y,
    X,
    sm.genmod.families.Gaussian(sm.genmod.families.links.identity()),
    var_weights=counts.flatten(),
).fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")


print(f"true dispersion estimates: {sigma**2}")
print(f"skinny dispersion estimates: {skinny_model.dispersion}")
print(f"statsmodels dispersion estimates: {stats_model.scale}")
