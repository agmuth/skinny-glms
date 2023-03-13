import numpy as np
import skinnyglms as skinny
import statsmodels.api as sm

np.random.seed(1234)

n = 100
p = 2

counts = np.random.poisson(lam=1, size=(n, 1))
counts[counts == 0] += 1

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
rate = np.exp(X @ b.T)
y = np.random.gamma(counts, 1/rate, (n, 1))


skinny_model = skinny.glm.SkinnyGLM(family=skinny.families.GaussianFamily(link=skinny.links.IdentityLink()))
skinny_model._irls(X, y, count_weights=counts/1.)

stats_model = sm.GLM(y, X, 
    sm.genmod.families.Gamma(sm.genmod.families.links.log()),
    freq_weights=counts.flatten()
).fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")

