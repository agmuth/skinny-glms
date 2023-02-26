import skinnyglms as skinny
import statsmodels.api as sm
import numpy as np

n = 1000
p = 2
sigma = 0.5

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

skinny_model = skinny.skinny_glm.SkinnyGLM(family=skinny.families.GaussianFamily(link=skinny.links.IdentityLink()))
skinny_model._irls(X, y)

stats_model = sm.GLM(y, X, family=sm.families.Gaussian(sm.genmod.families.links.identity()))
stats_model = stats_model.fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")


