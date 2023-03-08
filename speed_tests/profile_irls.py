import skinnyglms as skinny
import numpy as np


SEED = 1234
n = 1000
p = 1

np.random.seed(SEED)

skinny_family = skinny.families.BinomialFamily(
    skinny.links.LogitLink()
)

b = np.random.normal(scale=0.1, size=(1, p+1))
X = np.hstack([np.ones((n, 1)), np.random.normal(scale=0.1, size=(n, p))])

mu = skinny_family.link.inv_link(X @ b.T)
theta = skinny_family.canonical_link.link(mu)
y = skinny_family.sample(theta)

skinny_model = skinny.glm.SkinnyGLM(skinny_family)
skinny_model._irls(X, y)


# python -m cProfile -o speed_tests/profile_irls.prof speed_tests/profile_irls.py 
# snakeviz speed_tests/profile_irls.prof