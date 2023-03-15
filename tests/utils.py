import skinnyglms as skinny
import statsmodels.api as sm

TOL = 1e-4
SEED = 1324

ETA_BOUNDS = (-5., 5.)

N_AND_P = [(10**n, 10**p if p >= 0 else 0) for n in range(1, 4) for p in range(-1, n)]

STATSMODELS_MAPPING = {
    'GAUSSIAN' : {
        'families' : (skinny.families.GaussianFamily, sm.families.Gaussian),
        'links' : [
            (skinny.links.IdentityLink, sm.genmod.families.links.identity),
        ],
    },
    'BINOMIAL' : {
        'families' : (skinny.families.BinomialFamily, sm.families.Binomial),
        'links' : [
            (skinny.links.LogitLink, sm.genmod.families.links.logit),
            # (skinny.links.ProbitLink, sm.genmod.families.links.probit),
            # (skinny.links.CLogLogLink, sm.genmod.families.links.cloglog),
        ],
    },
    'GAMMA' : {
        'families' : (skinny.families.GammaFamily, sm.families.Gamma),
        'links' : [
            (skinny.links.LogLink, sm.genmod.families.links.log),
        ],
    },
    'POISSON' : {
        'families' : (skinny.families.PoissonFamily, sm.families.Poisson),
        'links' : [
            (skinny.links.LogLink, sm.genmod.families.links.log),
        ],
    },
    'INVERSEGAUSSIAN' : {
        'families' : (skinny.families.InverseGaussianFamily, sm.families.InverseGaussian),
        'links' : [
            # (skinny.links.InverssGaussianCanonicalLink, sm.genmod.families.links.inverse_squared),
            (skinny.links.LogLink, sm.genmod.families.links.log)
        ],
    },
}

DISTRIBUTIONS = [k for k in STATSMODELS_MAPPING.keys()]