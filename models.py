import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

from utils.truncated_mvn_sampler import TruncatedMVN

# Standard Bayes models


class GaussianUnknownMean:
    def __init__(self, mean0, var0, varx):
        """Initialize model, for standard Bayes.
        Prior: Normal
        Likelihood: Normal known variance
        Predictive posterior: GaussNormalian
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, data):
        """Upon observing a new datum x at time t,
        update all run length hypotheses.
        """
        x = data[t-1]
        new_prec_params = self.prec_params + (1/self.varx)
        new_mean_params = (self.mean_params * self.prec_params +
                           (x / self.varx)) / new_prec_params

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1./self.prec_params + self.varx


class MultivariateGaussianUnkownMean:
    def __init__(self, mu0, Sigma0, Cov, d):
        self.d = d

        self.mu0 = np.asarray([mu0])
        self.mu = np.asarray([mu0])

        self.Sigma0Inv = np.asarray([np.linalg.inv(Sigma0)])
        self.SigmaInv = np.asarray([np.linalg.inv(Sigma0)])

        self.Sigma0 = np.asarray([Sigma0])
        self.Sigma = np.asarray([Sigma0])

        self.Cov = Cov
        self.CovInv = np.linalg.inv(Cov)

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.empty(len(indices))
        for k, i in enumerate(indices):
            post_mean = self.mu[i].reshape(self.d)
            post_cov = self.Sigma[i] + self.Cov
            log_pred_prob[k] = stats.multivariate_normal.logpdf(x=x,
                                                                mean=post_mean,
                                                                cov=post_cov)

        return log_pred_prob

    def update_params(self, t, data):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        x = data[t-1]
        new_SigmaInv = self.SigmaInv + self.CovInv
        new_Sigma = np.asarray([np.linalg.inv(new_SigmaInv[i]) for
                                i in range(t)], dtype='float')
        new_mu = new_Sigma@(self.SigmaInv@self.mu +
                            self.CovInv@x.reshape((self.d, 1)))

        self.SigmaInv = np.concatenate((self.Sigma0Inv, new_SigmaInv))
        self.Sigma = np.concatenate((self.Sigma0, new_Sigma))
        self.mu = np.concatenate((self.mu0, new_mu))


class Gaussian:
    def __init__(self, mu0, kappa0, alpha0, omega0):
        """Initialize model, for standard Bayes.
        Prior: Normal-inverse gamma
        Likelihood: Normal
        Predictive posterior: t-student
        """
        self.alpha = np.array([alpha0])
        self.alpha0 = np.array([alpha0])

        self.omega = np.array([omega0])
        self.omega0 = np.array([omega0])

        self.kappa = np.array([kappa0])
        self.kappa0 = np.array([kappa0])

        self.mu = np.array([mu0])
        self.mu0 = np.array([mu0])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        df = 2 * self.alpha[indices]
        loc = self.mu[indices]
        scale = np.sqrt(self.omega[indices] * (self.kappa[indices] + 1) /
                        (self.alpha[indices] * self.kappa[indices]))

        return stats.t.logpdf(x=x, df=df, loc=loc, scale=scale)

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
        omegaT0 = np.concatenate(
            (
             self.omega0,
             self.omega
             + (self.kappa * (x - self.mu) ** 2) / (2.0 * (self.kappa + 1.0)),
            )
        )

        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.omega = omegaT0


class MultivariateGaussian:
    def __init__(self, dof=0, kappa=1, mu=-1, scale=-1, d=1):
        """Initialize model, for standard Bayes.
        Prior: Normal-inverse-Wishart
        Likelihood: Multivariate normal
        Predictive posterior: Multivariate t-student
        """
        # We default to the minimum possible degrees of freedom,
        # which is 1 greater than the dimensionality
        if dof == 0:
            dof = d + 1
        # The default mean is all 0s
        if mu == -1:
            mu = [0] * d
        else:
            mu = [mu] * d

        # The default covariance is the identity matrix.
        # The scale is the inverse of that, which is also the identity
        if scale == -1:
            scale = np.identity(d)
        else:
            scale = np.identity(scale)

        # The dimensionality of the dataset (number of variables)
        self.d = d

        # Each parameter is a vector of size 1 x t, where t is time.
        # Therefore each vector grows with each update.
        self.dof = np.array([dof])
        self.kappa = np.array([kappa])
        self.mu = np.array([mu])
        self.scale = np.array([scale])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        t_dof = self.dof - self.d + 1
        expanded = np.expand_dims((self.kappa * t_dof) /
                                  (self.kappa + 1), (1, 2))
        log_pred_prob = np.empty(len(indices))
        for k, i in enumerate(indices):
            df = t_dof[i]
            loc = self.mu[i]
            shape = np.linalg.inv(expanded[i] * self.scale[i])
            log_pred_prob[k] = stats.multivariate_t.logpdf(x=x, df=df,
                                                           loc=loc,
                                                           shape=shape)
        return log_pred_prob

    def update_params(self, t, data):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        x = data[t-1]
        centered = x - self.mu

        self.scale = np.concatenate(
            [
                self.scale[:1],
                np.linalg.inv(
                    np.linalg.inv(self.scale)
                    + np.expand_dims(self.kappa / (self.kappa + 1), (1, 2))
                    * (np.expand_dims(centered, 2) @ np.expand_dims(centered,
                                                                    1))
                ),
            ]
        )
        self.mu = np.concatenate(
            [
                self.mu[:1],
                (np.expand_dims(self.kappa, 1) * self.mu + x)
                / np.expand_dims(self.kappa + 1, 1),
            ]
        )
        self.dof = np.concatenate([self.dof[:1], self.dof + 1])
        self.kappa = np.concatenate([self.kappa[:1], self.kappa + 1])

# DSM Bayes modes


class SMGaussianUnknownMean:
    def __init__(self, data, omega, mean0, var0, varx):
        """Initialize model.
        """
        self.data = data
        self.omega = omega

        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def grad_r(self, x):
        return np.eye(1)

    def grad_b(self, x):
        return -x

    def lap_r(self, x):
        return np.zeros(1)

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def A(self, x):
        return self.grad_r(x).T@self.grad_r(x)

    def v(self, x):
        v1 = self.grad_r(x).T@self.grad_b(x)
        v2 = self.lap_r(x)
        return v1+v2

    def update_params(self, t, data):

        x = data[t-1]
        new_prec_params = self.prec_params + 2*self.omega*self.A(x)
        new_mean_params = (1/new_prec_params)*(self.prec_params *
                                               self.mean_params -
                                               2*self.omega*self.v(x))

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1/self.prec_params + self.varx


class DSMGaussianUnknownMean:
    def __init__(self, data, m, grad_m, omega, mean0,
                 var0, varx):
        """Initialize model, for DSM Bayes.
        Prior: Normal
        Likelihood: Normal
        Predictive posterior: Normal
        """
        self.data = data
        self.p = 1
        self.d = 1

        self.m = m
        self.grad_m = grad_m
        self.omega = omega

        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1/var0])

    def grad_r(self, x):
        return np.eye(1)/self.varx

    def grad_b(self, x):
        return -np.array([x])/self.varx

    def hess_r(self, x):
        return np.zeros([1, 1, 1])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        post_means = self.mean_params[indices]
        post_stds = np.sqrt(self.var_params[indices])
        return stats.norm(post_means, post_stds).logpdf(x)

    def A(self, x):
        return self.grad_r(x).T@self.m(x)@self.m(x).T@self.grad_r(x)

    def v(self, x):
        v1 = (self.grad_r(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = np.array([[np.sum([[self.grad_m(x)[i, j, i]*self.m(x)[p, j] +
                                     self.grad_m(x)[p, j, i]*self.m(x)[i, j]
                                     for i in range(self.d)]
                                    for j in range(self.d)])]
                          for p in range(self.p)])
        v2 = div_mm.T@self.grad_r(x)
        v3 = np.array([[np.trace(self.m(x)@self.m(x).T @
                                 self.hess_r(x)[:, :, p])]
                       for p in range(self.p)])
        return v1+v2+v3

    def update_params(self, t, data):

        x = data[t-1]
        new_prec_params = self.prec_params + 2*self.omega*self.A(x)
        new_mean_params = (1/new_prec_params)*(self.prec_params *
                                               self.mean_params -
                                               2*self.omega*self.v(x))

        self.mean_params = np.append([self.mean0], new_mean_params)
        self.prec_params = np.append([1/self.var0], new_prec_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1/self.prec_params + self.varx


class DSMBase(ABC):
    """
    Abstract class for DSM-Bayes model.
    """
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, d, p):
        """
        Initialize model.
        """

        self.data = data
        self.p = p
        self.d = d

        self.m = m
        self.grad_m = grad_m

        self.omega = omega

        self.mu0 = np.asarray([mu0])
        self.mu = np.asarray([mu0])

        self.Sigma0Inv = np.asarray([np.linalg.inv(Sigma0)])
        self.SigmaInv = np.asarray([np.linalg.inv(Sigma0)])

        self.Sigma0 = np.asarray([Sigma0])
        self.Sigma = np.asarray([Sigma0])

    @abstractmethod
    def grad_r(self, x):
        raise NotImplementedError("subclasses should implement this!")

    @abstractmethod
    def hess_r(self, x):
        raise NotImplementedError("subclasses should implement this!")

    @abstractmethod
    def grad_b(self, x):
        raise NotImplementedError("subclasses should implement this!")

    def A(self, x):
        """
        Function to calculate Lambda(x).
        """
        return self.grad_r(x).T@self.m(x)@self.m(x).T@self.grad_r(x)

    def v(self, x):
        """
        Function to calculate nu(x).
        """
        v1 = (self.grad_r(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = (np.sum(self.grad_m(x)@self.m(x).T, axis=(0, 2)) +
                  np.sum(self.grad_m(x)@self.m(x).T, axis=(0, 1))
                  ).reshape(self.d, 1)
        v2 = self.grad_r(x).T@div_mm
        v3 = np.trace((self.m(x)@self.m(x).T@self.hess_r(x))
                      ).reshape(self.p, 1)
        return v1+v2+v3

    def update_params(self, t, data):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        x = data[t-1]
        new_SigmaInv = self.SigmaInv + 2*self.omega*self.A(x)
        new_Sigma = np.asarray([np.linalg.inv(new_SigmaInv[i])
                                for i in range(t)], dtype='float')
        new_mu = new_Sigma@(self.SigmaInv@self.mu-2*self.omega*self.v(x))

        self.SigmaInv = np.concatenate((self.Sigma0Inv, new_SigmaInv))
        self.Sigma = np.concatenate((self.Sigma0, new_Sigma))
        self.mu = np.concatenate((self.mu0, new_mu))

    @abstractmethod
    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        raise NotImplementedError("subclasses should implement this!")


class DSMExponentialGaussian(DSMBase):
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, omega, mu0, Sigma0, d=2, p=3)
        self.b = b

    def grad_r(self, x):
        grad_t1 = np.zeros((1, 3))
        grad_t1[:, 0] = -1
        grad_t2 = np.zeros((1, 3))
        grad_t2[:, 1] = 1
        grad_t2[:, 2] = -x[1]
        return np.concatenate((grad_t1, grad_t2), axis=0)

    def hess_r(self, x):
        hess_t = np.zeros((2, 2, 3))
        hess_t[1, 1, 2] = -1
        return hess_t

    def grad_b(self, x):
        return np.asarray([[0], [0]])

    def log_pred_prob(self, t, data, indices):
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([0, -np.inf, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),
                               self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0, :]
            eta2 = eta[1, :]
            eta3 = eta[2, :]
            sample_exp = stats.expon(scale=1/eta1).pdf(x[0])
            sample_norm = stats.norm(loc=eta2/eta3,
                                     scale=np.sqrt(1/eta3)
                                     ).pdf(x[1])
            log_pred_prob[k] = np.log(np.average(sample_norm*sample_exp))
        return log_pred_prob


class DSMGaussian(DSMBase):
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, omega, mu0, Sigma0, d=1, p=2)
        self.b = b
        """Initialize model, for DSM Bayes.
        Prior: Squared exponential
        Likelihood: Normal
        Predictive posterior: Approximated by sampling.
        """

    def grad_r(self, x):
        return np.asarray([[1, -x[0]]], dtype='float')

    def hess_r(self, x):
        return np.asarray([[[0, -1]]])

    def grad_b(self, x):
        return np.asarray([[0]])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([-np.inf, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),
                               self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0, :]
            eta2 = eta[1, :]
            log_pred_prob[k] = np.log(np.average(
                stats.norm(loc=eta1/eta2, scale=np.sqrt(1/eta2)).pdf(x)))
        return log_pred_prob


class DSMGamma(DSMBase):
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, b=20):
        super().__init__(data, m, grad_m, omega, mu0, Sigma0, d=1, p=2)
        self.b = b
        """Initialize model, for DSM Bayes.
        Prior: Squared exponential
        Likelihood: Gamma
        Predictive posterior: Approximated by sampling.
        """

    def grad_r(self, x):
        return np.asarray([[1/x[0], -1]], dtype='float')

    def hess_r(self, x):
        return np.asarray([[[-1/(x[0]**2), 0]]], dtype='float')

    def grad_b(self, x):
        return np.asarray([[0]])

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.asarray([-1, 0])
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),
                               self.Sigma[i], lb, ub).sample(self.b)
            eta1 = eta[0, :]
            eta2 = eta[1, :]
            log_pred_prob[k] = np.log(np.average(stats.gamma(a=eta1+1,
                                                             scale=1/eta2
                                                             ).pdf(x[0])))
        return log_pred_prob


class DSMMultivariateGaussian(DSMBase):
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, d, b=20):
        super().__init__(data, m, grad_m, omega, mu0, Sigma0, d=d, p=2*d)
        self.b = b

    def grad_r(self, x):
        grad_r = np.zeros((self.d, self.p))
        for i in range(self.d):
            grad_r[i, 2*i] = 1
            grad_r[i, 2*i+1] = -x[i]
        return grad_r

    def hess_r(self, x):
        hess_r = np.zeros((self.d, self.d, self.p))
        for i in range(self.d):
            hess_r[i, i, 2*i+1] = -1
        return hess_r

    def grad_b(self, x):
        return np.zeros((self.d, 1))

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.zeros(len(indices))
        lb = np.zeros(self.p)
        for i in range(self.d):
            lb[2*i] = -np.inf
        ub = np.ones(self.p) * np.inf
        for k, i in enumerate(indices):
            eta = TruncatedMVN(self.mu[i].reshape(self.p),
                               self.Sigma[i], lb, ub).sample(self.b)
            samples = np.empty((self.d, self.b))
            for j in range(self.d):
                eta1 = eta[2*j, :]
                eta2 = eta[2*j+1, :]
                samples[j] = stats.norm(loc=eta1/eta2,
                                        scale=np.sqrt(1/eta2)).pdf(x[j])
            log_pred_prob[k] = np.log(np.average(np.prod(samples,
                                                         axis=0)))
        return log_pred_prob


class DSMMultivariateGaussianUnkownMean(DSMBase):
    def __init__(self, data, m, grad_m, omega, mu0, Sigma0, Cov, d, b=20):
        super().__init__(data, m, grad_m, omega, mu0, Sigma0, d=d, p=d)
        self.b = b
        self.Cov = Cov
        self.CovInv = np.linalg.inv(Cov)

    def grad_r(self, x):
        return self.CovInv

    def hess_r(self, x):
        return np.zeros((self.d, self.d, self.d))

    def grad_b(self, x):
        x = x.reshape((self.d, 1))
        return -self.CovInv@x

    def log_pred_prob(self, t, data, indices):
        """Compute predictive probabilities pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        x = data[t-1]
        log_pred_prob = np.empty(len(indices))
        for k, i in enumerate(indices):
            post_mean = self.mu[i].reshape(self.p)
            post_cov = self.Sigma[i] + self.Cov
            log_pred_prob[k] = stats.multivariate_normal.logpdf(x=x,
                                                                mean=post_mean,
                                                                cov=post_cov)

        return log_pred_prob
