import skinnyglms as skinny
from skinnyglms.mappings.statsmodels import STATSMODELS_MAPPING
import statsmodels.api as sm
import numpy as np
import pytest
from tests.utils import SEED, TOL, DISTRIBUTIONS, ETA_BOUNDS
from itertools import product

test_params = list()

for distn in DISTRIBUTIONS:
    test_params += product(
        (STATSMODELS_MAPPING[distn]['families'],),
        STATSMODELS_MAPPING[distn]['links']
    )


@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement(families, links):
    np.random.seed(SEED)

    sk_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    
    eta = np.linspace(ETA_BOUNDS[0], ETA_BOUNDS[1], int(1e1))
    mu_sk = sk_family.link.inv_link(eta)
    mu_sm = sm_familiy.fitted(eta)

    assert np.allclose(mu_sk, mu_sm)

    assert np.allclose(
        sk_family.link.link_deriv(mu_sk),
        sm_familiy.link.deriv(mu_sm)
    )

    assert np.allclose(
         np.multiply(
            sk_family.inv_variance(mu_sk), 
            np.square(sk_family.link.inv_link_deriv(eta))
        ),
        sm_familiy.weights(mu_sm)
    )
 
    