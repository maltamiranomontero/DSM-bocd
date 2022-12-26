import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

from utils.mean_truncated_normal import mean_truncated_normal_2d
from utils.truncated_mvn_sampler import TruncatedMVN
class GaussianUnknownMean:
    
    def __init__(self, mean0, var0, varx):
        """Initialize model.
        
        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])
    
    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        x = data[t-1] 
        post_means = self.mean_params[indices]
        post_stds  = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)
    
    def update_params(self, t, data):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        x = data[t-1] 
        new_prec_params  = self.prec_params + (1/self.varx)
        # See eq. 24 in (Murphy 2007).
        new_mean_params  = (self.mean_params * self.prec_params + \
                            (x / self.varx)) / new_prec_params
        
        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx

class GaussianUnknownMeanVariance:
    
    def __init__(self, mu0, kappa0, alpha0, beta0):
        """Initialize model.
        """
        self.alpha = self.alpha0 = np.array([alpha0])
        self.beta = self.beta0 = np.array([beta0])
        self.kappa = self.kappa0 = np.array([kappa0])
        self.mu = self.mu0 = np.array([mu0])
    
    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1] 
        df = 2 * self.alpha[indices]
        loc = self.mu[indices]
        scale = np.sqrt(self.beta[indices] * (self.kappa[indices] + 1) / (self.alpha[indices] * self.kappa[indices]))

        return stats.t.logpdf(x=x,df=df, loc=loc, scale=scale )
    
    def update_params(self, t, data):
        """Upon observing a new datum x at time t, update all run length 
        hypotheses.
        """
        x = data[t-1] 
        
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + x) / (self.kappa + 1))
        )
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.0))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (
                self.beta0,
                self.beta
                + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0

class SMGaussianUnknownMean:
    
    def __init__(self, data, grad_t, lap_t, grad_b, beta, mean0, var0, varx):
        """Initialize model.
        """
        self.data = data

        self.grad_t = grad_t
        self.lap_t = lap_t
        self.grad_b = grad_b
        self.beta = beta

        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])


    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1] 
        post_means = self.mean_params[indices]
        post_stds  = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def A(self, x):
        return self.grad_t(x).T@self.grad_t(x)
    
    def v(self, x):
        v1 = self.grad_t(x).T@self.grad_b(x)
        v2 = self.lap_t(x)
        return v1+v2

    def update_params(self, t, data):

        x = data[t-1] 
        new_prec_params  = self.prec_params + 2*self.beta*self.A(x)
        new_mean_params  = (1/new_prec_params)*(self.prec_params*self.mean_params-2*self.beta*self.v(x))

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1/self.prec_params + self.varx

class DSMGaussianUnknownMean:
    
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mean0, var0, varx, p=1, d=1):
        """Initialize model.
        """
        self.data = data
        self.p = p
        self.d = d
        
        self.m = m
        self.grad_m = grad_m
        self.grad_t = grad_t
        self.hess_t = hess_t
        self.grad_b = grad_b
        self.beta = beta

        self.mean0 = mean0
        self.var0  = var0
        self.varx  = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])


    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1] 
        post_means = self.mean_params[indices]
        post_stds  = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def A(self, x):
        return self.grad_t(x).T@self.m(x)@self.m(x).T@self.grad_t(x)
    
    def v(self, x):
        v1 = (self.grad_t(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = np.array([[np.sum([[self.grad_m(x)[i,j,i]*self.m(x)[p,j] + self.grad_m(x)[p,j,i]*self.m(x)[i,j]  for i in range(self.d)]for j in range(self.d)])] for p in range(self.p)])
        v2 = div_mm.T@self.grad_t(x)
        v3 = np.array([[np.trace(self.m(x)@self.m(x).T@self.hess_t(x)[:,:,p])] for p in range(self.p)])
        return v1+v2+v3

    def update_params(self, t, data):

        x = data[t-1] 
        new_prec_params  = self.prec_params + 2*self.beta*self.A(x)
        new_mean_params  = (1/new_prec_params)*(self.prec_params*self.mean_params-2*self.beta*self.v(x))

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1/self.prec_params + self.varx

class DSMBase(ABC):
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, d, p):
        """Initialize model.
        """
        self.data = data
        self.p = p
        self.d = d
        
        self.m = m
        self.grad_m = grad_m
        self.grad_t = grad_t
        self.hess_t = hess_t
        self.grad_b = grad_b
        self.beta = beta

        self.mu0 = np.asarray([mu0])
        self.mu = np.asarray([mu0])

        self.Sigma0Inv =  np.asarray([np.linalg.inv(Sigma0)])
        self.SigmaInv = np.asarray([np.linalg.inv(Sigma0)])

        self.Sigma0 = np.asarray([Sigma0])
        self.Sigma = np.asarray([Sigma0])

    def A(self, x):
        return self.grad_t(x).T@self.m(x)@self.m(x).T@self.grad_t(x)
    
    def v(self, x):
        v1 = (self.grad_t(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = np.asarray([[np.sum([[self.grad_m(x)[i,j,i]*self.m(x)[p,j] + self.grad_m(x)[p,j,i]*self.m(x)[i,j]  for i in range(self.d)]for j in range(self.d)])] for p in range(self.d)])
        v2 = self.grad_t(x).T@div_mm
        v3 = np.asarray([[np.trace(self.m(x)@self.m(x).T@self.hess_t(x)[:,:,p])] for p in range(self.p)])
        return v1+v2+v3

    def update_params(self, t, data):
        x = data[t-1] 

        new_SigmaInv  = self.SigmaInv + 2*self.beta*self.A(x)
        new_Sigma = np.asarray([np.linalg.inv(new_SigmaInv[i]) for i in range(t)], dtype='float')
        new_mu  = new_Sigma@(self.SigmaInv@self.mu-2*self.beta*self.v(x))

        self.SigmaInv = np.concatenate((self.Sigma0Inv,new_SigmaInv))
        self.Sigma = np.concatenate((self.Sigma0,new_Sigma))
        self.mu = np.concatenate((self.mu0, new_mu))
    
    @abstractmethod
    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        raise NotImplementedError("subclasses should implement this!") 

class DSMExponentialGaussianSampling(DSMBase):
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, d=2, p=3)
        self.b = b

    def log_pred_prob(self, t, data, indices):
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([0, -np.inf, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0,:]
            eta2 = eta[1,:]
            eta3 = eta[2,:]
            sample_exp = stats.expon(scale = 1/eta1).pdf(x[0])
            sample_norm = stats.norm(loc=eta2/eta3,scale=np.sqrt(1/eta3)).pdf(x[1])
            log_pred_prob[k] = np.log(np.average(sample_norm*sample_exp))
        return log_pred_prob

class DSMGaussianSampling(DSMBase):
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, d=1, p=2)
        self.b = b

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([-np.inf, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0,:]
            eta2 = eta[1,:]
            log_pred_prob[k] = np.log(np.average(stats.norm(loc=eta1/eta2,scale=np.sqrt(1/eta2)).pdf(x)))
        return log_pred_prob

class DSMGaussianApprox(DSMBase):
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, b=20, T=True):
        super().__init__(data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, d=1, p=2)
        self.b = b
        self.T = T

    def log_pred_prob(self, t, data, indices):
        x = data[t-1]

        eta1 = []
        eta2 = []
        for i in indices:
            mean = mean_truncated_normal_2d(self.mu[i],self.Sigma[i])
            eta1.append(mean[0])
            eta2.append(mean[1])
        eta1 = np.array(eta1)
        eta2 = np.array(eta2)

        loc=eta1/eta2
        scale=np.sqrt(1/eta2)
        df = indices+1
        if self.T:
            return stats.t(loc=loc, scale=scale, df=df).logpdf(x)
        else:
            return stats.norm(loc=loc, scale=scale).logpdf(x)

class DSMGammaSampling(DSMBase):
    def __init__(self, data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, grad_t, hess_t, grad_b, beta, mu0, Sigma0, d=1, p=2)
        self.b = b
    
    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([-1, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0,:]
            eta2 = eta[1,:]
            log_pred_prob[k] = np.log(np.average(stats.gamma(a = eta1+1, scale=1/eta2).pdf(x)))
        return log_pred_prob