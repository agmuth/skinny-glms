import skinnyglms as skinny
import statsmodels.api as sm

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
            (skinny.links.ProbitLink, sm.genmod.families.links.probit),
            (skinny.links.CLogLogLink, sm.genmod.families.links.cloglog),
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
}