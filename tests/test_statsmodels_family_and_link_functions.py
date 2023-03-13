import skinnyglms as skinny
from skinnyglms.mappings.statsmodels import STATSMODELS_MAPPING
import statsmodels.api as sm
import numpy as np
import pytest
from tests.utils import DISTRIBUTIONS, ETA_BOUNDS
from itertools import product

eta = np.linspace(ETA_BOUNDS[0], ETA_BOUNDS[1], 10)
eta = eta.reshape((len(eta), 1))

test_params = list()

for distn in DISTRIBUTIONS:
    test_params += product(
        (STATSMODELS_MAPPING[distn]['families'],),
        STATSMODELS_MAPPING[distn]['links']
    )


@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement_inv_link(families, links):   
    sk_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    mu_sk = sk_family.link.inv_link(eta)
    mu_sm = sm_familiy.fitted(eta)

    assert np.allclose(mu_sk, mu_sm)


@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement_link_deriv(families, links):
    sk_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    mu_sk = sk_family.link.inv_link(eta)
    mu_sm = sm_familiy.fitted(eta)

    assert np.allclose(
        sk_family.link.link_deriv(mu_sk),
        sm_familiy.link.deriv(mu_sm)
    )


@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement_variance(families, links):
    sk_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    mu_sk = sk_family.link.inv_link(eta)
    mu_sm = sm_familiy.fitted(eta)

    assert np.allclose(sk_family.inv_variance(mu_sk),  sm_familiy.variance(mu_sm)**-1)


@pytest.mark.parametrize("families, links", test_params)
def test_statsmodels_agreement_inv_link_deriv(families, links):
    sk_family = families[0](links[0]())
    sm_familiy = families[1](links[1]())
    mu_sm = sm_familiy.fitted(eta)

    assert np.allclose(sk_family.link.inv_link_deriv(eta), sm_familiy.link.deriv(mu_sm)**-1)

    