import skinnyglms as skinny
import statsmodels.api as sm
import numpy as np
import timeit
n = 1000
p = 2
sigma = 0.5

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
y = X @ b.T + np.random.normal(scale=sigma, size=(n, 1))

skinny_model = skinny.glm.SkinnyGLM(family=skinny.families.GaussianFamily(link=skinny.links.IdentityLink()))
skinny_time = timeit.repeat("skinny_model._irls(X, y)", repeat=2, number=100, globals=globals())
print(skinny_time)

stats_model = sm.GLM(y, X, family=sm.families.Gaussian(sm.genmod.families.links.identity()))


sm_time = timeit.repeat("stats_model.fit()", repeat=2, number=100, globals=globals())
print(sm_time)

stats_model = stats_model.fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")

print(f"true dispersion estimates: {sigma**2}")
print(f"skinny dispersion estimates: {skinny_model.dispersion}")
print(f"statsmodels dispersion estimates: {stats_model.scale}")

theta = skinny_model.family.link.inv_link(X @ skinny_model.b)
phi = skinny_model.dispersion
samples = skinny_model.family.sample(theta, phi, 5)
print(samples[:2])
