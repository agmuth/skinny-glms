import skinnyglms as skinny
import statsmodels.api as sm
import numpy as np

skinny_link = skinny.links.LogLink()
sm_link = sm.genmod.families.links.log()

n = 1000
p = 1

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
offset = np.random.randint(1, 10, (n, 1))
lam = offset * np.exp(X @ b.T)
y = np.random.poisson(lam, (n, 1))

skinny_model = skinny.glm.SkinnyGLM(family=skinny.families.PoissonFamily(skinny_link))
skinny_model._irls(X, y, offset)

stats_model = sm.GLM(y, X, family=sm.families.Poisson(sm_link), offset=skinny_link.link(offset).flatten())
stats_model = stats_model.fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")

print(f"skinny dispersion estimates: {skinny_model.dispersion}")
print(f"statsmodels dispersion estimates: {stats_model.scale}")


