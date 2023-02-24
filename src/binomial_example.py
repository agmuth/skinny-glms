from families import *
from links import *
from functions import *
from skinny_glm import SkinnyGLM
import statsmodels.api as sm
import numpy as np



# skinny_link = LogitLink()
# sm_link = sm.genmod.families.links.logit()

skinny_link = ProbitLink()
sm_link = sm.genmod.families.links.probit()

n = 1000
p = 1

X = np.hstack([np.ones((n, p)), np.random.normal(size=(n, p))])
b = np.random.normal(size=(1, p+1))
probs = skinny_link.inv_link(X @ b.T)
y = np.random.binomial(1, probs, (n, 1))

skinny_model = SkinnyGLM(family=BinomialFamily(skinny_link))
skinny_model._irls(X, y)

stats_model = sm.GLM(y, X, family=sm.families.Binomial(sm_link))
stats_model = stats_model.fit()

print(f"true parameter estimates: {b.flatten()}")
print(f"skinny parameter estimates: {skinny_model.b.flatten()}")
print(f"statsmodels parameter estimates: {stats_model.params.flatten()}")


