import numpy as np
import statsmodels.api as sm

import skinnyglms as skinny

np.random.seed(1234)

n = 100
p = 2

X = np.hstack([np.ones((n, 1)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p + 1))
offset = np.random.randint(1, 5, (n, 1)) / 1.0  # offset must be float
lam = offset * np.exp(X @ b.T)
y = np.random.poisson(lam, (n, 1))

skinny_model = skinny.glm.SkinnyGLM(
    family=skinny.families.PoissonFamily(link=skinny.links.LogLink())
)
skinny_model._irls(X, y, offset)

stats_model = sm.GLM(
    y,
    X,
    sm.genmod.families.Poisson(sm.genmod.families.links.log()),
    np.log(offset).flatten(),
).fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")
