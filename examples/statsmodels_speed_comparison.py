import timeit

import numpy as np
import pandas as pd
import statsmodels.api as sm

import skinnyglms as skinny
from tests.utils import DISTRIBUTIONS, N_AND_P, SEED, STATSMODELS_MAPPING


def speed_test(families, links, n, p, repeats=5, number=10):
    skinny_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())

    b = np.random.normal(scale=0.1, size=(1, p + 1))
    X = np.hstack([np.ones((n, 1)), np.random.normal(scale=0.1, size=(n, p))])

    mu = skinny_family.link.inv_link(X @ b.T)
    theta = skinny_family.canonical_link.link(mu)
    y = skinny_family.sample(theta)

    skinnyglms_glm = skinny.glm.SkinnyGLM(family=skinny_family)
    statsmodel_glm = sm.GLM(y, X, family=sm_familiy)

    try:
        skinnyglms_seconds = (
            min(
                timeit.repeat(
                    "skinnyglms_glm._irls(X, y)",
                    repeat=repeats,
                    number=number,
                    globals=locals(),
                )
            )
            / number
        )
    except Exception as e:
        skinnyglms_seconds = -1
    try:
        statsmodels_seconds = (
            min(
                timeit.repeat(
                    "statsmodel_glm.fit()",
                    repeat=repeats,
                    number=number,
                    globals=locals(),
                )
            )
            / number
        )
    except Exception as e:
        statsmodels_seconds = -1

    return (
        int(skinnyglms_seconds * 1e6) if skinnyglms_seconds > 0 else -1,
        int(statsmodels_seconds * 1e6) if statsmodels_seconds > 0 else -1,
    )


if __name__ == "__main__":
    np.random.seed(SEED)

    results = []

    for params in N_AND_P:
        n = params[0]
        p = params[1]

        for distn in DISTRIBUTIONS:
            families = STATSMODELS_MAPPING[distn]["families"]
            for links in STATSMODELS_MAPPING[distn]["links"]:
                skinny_family = families[0](links[0]())
                fitting_times = speed_test(families, links, n, p)
                results.append(
                    [str(skinny_family), str(skinny_family.link), n, p, *fitting_times]
                )

    results = pd.DataFrame(
        results,
        columns=[
            "family",
            "link",
            "n",
            "p",
            "skinnyglms_micro_secs",
            "statsmodels_micro_secs",
        ],
    )
    results.to_csv("examples/speed_comparison.csv", index=False)
