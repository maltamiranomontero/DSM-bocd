import numpy as np
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp
import optax
from functools import partial

dist = tfp.distributions

class OmegaEstimatorGamma:
    def __init__(self, data, m, grad_m, mu0, Sigma0):
        self.data = data
        self.p = 2
        self.d = 1
        self.n = self.data.shape[0]
        ## DSM 
        self.m = m
        self.grad_m = grad_m

        self.mu0 = mu0

        self.Sigma0Inv =  np.linalg.inv(Sigma0)

        self.Sigma0 = Sigma0

        self.A = np.sum([self.Ax(x) for x in self.data],axis=0)/self.n
        self.v = 2*np.sum([self.vx(x) for x in self.data],axis=0)/self.n
    
    
    def grad_r(self,x):
        return np.asarray([[1/x,-1]],dtype='float')

    def hess_r(self,x):
        return np.asarray([[[-1/(x**2),0]]],dtype='float')

    def grad_b(self,x):
        return np.asarray([[0]])
    
    def Ax(self, x):
        return self.grad_r(x).T@self.m(x)@self.m(x).T@self.grad_r(x)
    
    def vx(self, x):
        v1 = (self.grad_r(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = (np.sum(self.grad_m(x)@self.m(x).T,axis=(0,2))+ np.sum(self.grad_m(x)@self.m(x).T,axis=(0,1))).reshape(self.d,1)
        v2 = self.grad_r(x).T@div_mm
        v3 = np.trace((self.m(x)@self.m(x).T@self.hess_r(x))).reshape(self.p,1)
        return v1+v2+v3
    
    def var_dsm_full(self, b):
        return jnp.linalg.inv((self.Sigma0Inv + 2*b*self.n*self.A))

    def mu_dsm_full(self, b):
        return self.var_dsm_full(b)@(self.Sigma0Inv@self.mu0 - b*self.n*self.v)
    
    def log_gamma_posterior(self, alpha, beta, prior_parameters):

        p, q, r, s = prior_parameters
        p = p * jnp.prod(self.data)
        q = q + jnp.sum(self.data)
        r = self.n + r
        s = self.n + s
        return (alpha-1)*jnp.log(p)-beta*q - r*jax.scipy.special.gammaln(alpha)+alpha*s*jnp.log(beta)
    
    @partial(jax.jit, static_argnums=(0,))
    def kl(self, omega, prior_parameters = [0.01, 5, 1, 1], n_samples=1000, key = jax.random.PRNGKey(1)):
        mu = self.mu_dsm_full(omega)
        var = self.var_dsm_full(omega)
        q1 = dist.TruncatedNormal(loc=mu[0], scale=jnp.sqrt(var[0,0]), low = -1.0, high = 1.0e10)
        q2 = dist.TruncatedNormal(loc=mu[1], scale=jnp.sqrt(var[1,1]), low = 0.0, high = 1.0e10)

        sample_set_1 = q1.sample(
            seed=key,
            sample_shape=[
                n_samples,
            ],
        )
        sample_set_2 = q2.sample(
            seed=key,
            sample_shape=[
                n_samples,
            ],
        )
        return jnp.mean(q1.log_prob(sample_set_1)+q2.log_prob(sample_set_2) - self.log_gamma_posterior(sample_set_1+1, sample_set_2, prior_parameters))
    
    def omega(self,omega0, lr = 0.01, niter = 1000, prior_parameters = [0.01, 5, 1, 1]):
        param = jnp.array([omega0])
        optimizer = optax.sgd(learning_rate=lr)
        opt_state = optimizer.init(param)
        costs = np.empty(niter)
        params = np.empty(niter)
        key = jax.random.PRNGKey(1)
        grad_loss = jax.grad(self.kl, argnums=(0))
        
        for i in range(niter):
            key, subkey = jax.random.split(key)
            cost_val = self.kl(param, prior_parameters=prior_parameters, key = subkey)
            costs[i] = cost_val
            grads = grad_loss(param, prior_parameters=prior_parameters)
            updates, opt_state = optimizer.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            params[i] = param
        return params, costs

class OmegaEstimatorGaussian:
    def __init__(self, data, m ,grad_m, mu0, Sigma0):
        self.data = data
        self.p = 2
        self.d = 1
        self.n = self.data.shape[0]
        ## DSM 
        self.m = m
        self.grad_m = grad_m

        self.mu0 = mu0

        self.Sigma0Inv =  np.linalg.inv(Sigma0)

        self.Sigma0 = Sigma0

        self.A = np.sum([self.Ax(x) for x in self.data],axis=0)/self.n
        self.v = 2*np.sum([self.vx(x) for x in self.data],axis=0)/self.n


    def grad_r(self,x):
        return np.asarray([[1,-x]],dtype='float')

    def hess_r(self,x):
        return np.asarray([[[0,-1]]])

    def grad_b(self,x):
        return np.asarray([[0]])

    
    def Ax(self, x):
        return self.grad_r(x).T@self.m(x)@self.m(x).T@self.grad_r(x)
    
    def vx(self, x):
        v1 = (self.grad_r(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = (np.sum(self.grad_m(x)@self.m(x).T,axis=(0,2))+ np.sum(self.grad_m(x)@self.m(x).T,axis=(0,1))).reshape(self.d,1)
        v2 = self.grad_r(x).T@div_mm
        v3 = np.trace((self.m(x)@self.m(x).T@self.hess_r(x))).reshape(self.p,1)
        return v1+v2+v3
    
    def var_dsm_full(self, b):
        return jnp.linalg.inv((self.Sigma0Inv + 2*b*self.n*self.A))

    def mu_dsm_full(self, b):
        return self.var_dsm_full(b)@(self.Sigma0Inv@self.mu0 - b*self.n*self.v)
    
    def log_norminvgamma_posterior(self, mu, sigma2, prior_parameters):

        alpha, beta, u, k = prior_parameters
        sample_mean = jnp.mean(self.data)
        
        alpha = alpha + self.n/2
        beta = beta + 0.5*(jnp.sum((self.data-sample_mean)**2)+(self.n*k*(sample_mean-u)**2/(k + self.n)))
        u = (k*u+self.n*sample_mean)/(k+self.n)
        k = k + self.n 

        t1 =0.5 * (jnp.log(k) - jnp.log(2*jnp.pi*sigma2))
        t2 = alpha*jnp.log(beta)-jax.scipy.special.gammaln(alpha)
        t3 = -(alpha + 1)*jnp.log(sigma2)
        t4 = -(2*beta + k*(mu-u)**2)/(2*sigma2)

        return t1+t2+t3+t4
    
    @partial(jax.jit, static_argnums=(0,))
    def kl(self, omega, prior_parameters = [1, 1, 0, 1], n_samples=1000, key = jax.random.PRNGKey(1)):
        mu = self.mu_dsm_full(omega)
        var = self.var_dsm_full(omega)
        q1 = dist.Normal(loc=mu[0], scale=jnp.sqrt(var[0,0]))
        q2 = dist.TruncatedNormal(loc=mu[1], scale=jnp.sqrt(var[1,1]), low = 0.0, high = 1.0e10)

        sample_set_1 = q1.sample(
            seed=key,
            sample_shape=[
                n_samples,
            ],
        )
        sample_set_2 = q2.sample(
            seed=key,
            sample_shape=[
                n_samples,
            ],
        )
        return jnp.mean(q1.log_prob(sample_set_1)+q2.log_prob(sample_set_2) - self.log_norminvgamma_posterior(sample_set_1/sample_set_2, 1/sample_set_2, prior_parameters))
    
    def omega(self,omega0, lr = 0.01, niter = 1000, prior_parameters= [1, 1, 0, 1]):
        param = jnp.array([omega0])
        optimizer = optax.sgd(learning_rate=lr)
        opt_state = optimizer.init(param)
        costs = np.empty(niter)
        params = np.empty(niter)
        key = jax.random.PRNGKey(1)
        grad_loss = jax.grad(self.kl, argnums=(0))
        
        for i in range(niter):
            key, subkey = jax.random.split(key)
            cost_val = self.kl(param, prior_parameters=prior_parameters, key = subkey)
            costs[i] = cost_val
            grads = grad_loss(param, prior_parameters=prior_parameters)
            updates, opt_state = optimizer.update(grads, opt_state)
            param = optax.apply_updates(param, updates)
            params[i] = param
        return params, costs


