import skinnyglms as skinny
import statsmodels.api as sm

STATSMODELS_MAPPING = {
    'GAUSSIAN' : {
        'families' : (skinny.families.GaussianFamily, sm.families.Gaussian),
        'links' : [
            (skinny.links.IdentityLink(), sm.genmod.families.links.identity())
        ],
    }
}