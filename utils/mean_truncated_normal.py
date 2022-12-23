import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects.packages import importr

utils = rpackages.importr('utils')
#utils.chooseCRANmirror(ind=0)

# Install packages
#packnames = ('tmvtnorm')
#utils.install_packages(StrVector(packnames))

importr('tmvtnorm')


def mean_truncated_normal_2d(mu, Sigma):
    mean = robjects.r('''
                mu <- c({}, {})
                sigma <- matrix(c( {}, {},
                {}, {}), 2, 2)
                a <- c(-Inf, 0)
                b <- c(Inf, Inf)
                # compute first moment
                mtmvnorm(mu, sigma, lower=a, upper=b, doComputeVariance=FALSE)['tmean']
                '''.format(mu[0,0], mu[1,0],
                Sigma[0,0], Sigma[0,1], Sigma[1,0], Sigma[1,1]))
    
    return robjects.FloatVector(mean[0])