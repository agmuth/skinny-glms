import skinnyglms as skinny
import sklearn.linear_model as lm

SKLEARN_MAPPING = {
    'GAUSSIAN' : {
        'families' : (skinny.families.GaussianFamily, lm.LinearRegression),
        'links' : [
            (skinny.links.IdentityLink, None),
        ],
    },
    'BINOMIAL' : {
        'families' : (skinny.families.BinomialFamily, lm.LogisticRegression),
        'links' : [
            (skinny.links.LogitLink, None),
            # (skinny.links.ProbitLink, sm.genmod.families.links.probit),
            # (skinny.links.CLogLogLink, sm.genmod.families.links.cloglog),
        ],
    },
    'GAMMA' : {
        'families' : (skinny.families.GammaFamily, lm.GammaRegressor),
        'links' : [
            (skinny.links.LogLink, None),
        ],
    },
    'POISSON' : {
        'families' : (skinny.families.PoissonFamily, lm.PoissonRegressor),
        'links' : [
            (skinny.links.LogLink, None),
        ],
    },
}