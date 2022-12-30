import numpy as np
from scipy.optimize import minimize

class BetaEstimatorFisher:
    
    def __init__(self,data, m, grad_m, grad_t, hess_t, grad_b, mu0, Sigma0, B = 1000, d=1, p=1):
        self.data = data
        self.p = p
        self.d = d
        self.n = self.data.shape[0]
        
        self.B = B

        self.m = m
        self.grad_m = grad_m
        self.grad_t = grad_t
        self.hess_t = hess_t
        self.grad_b = grad_b

        self.mu0 = mu0
        self.Sigma0 = Sigma0

        self.A = np.sum([self.Ax(x) for x in self.data],axis=0)/self.n
        self.v = 2*np.sum([self.vx(x) for x in self.data],axis=0)/self.n

    def Ax(self, x):
        return self.grad_t(x).T@self.m(x)@self.m(x).T@self.grad_t(x)
    
    def vx(self, x):
        v1 = (self.grad_t(x).T@self.m(x)@self.m(x)@self.grad_b(x))
        div_mm = (np.sum(self.grad_m(x)@self.m(x).T,axis=(0,2))+ np.sum(self.grad_m(x)@self.m(x).T,axis=(0,1))).reshape(self.d,1)
        v2 = self.grad_t(x).T@div_mm
        v3 = np.trace((self.m(x)@self.m(x).T@self.hess_t(x))).reshape(self.p,1)
        return v1+v2+v3

    def thetab(self,data):
        A = np.sum([self.Ax(x) for x in data],axis=0)/self.n
        v = 2*np.sum([self.vx(x) for x in data],axis=0)/self.n
        return np.linalg.solve(2*A,v)

    def thetaB(self):
        idx = np.random.randint(0,self.n,(self.n,self.B))
        out = self.data[idx].swapaxes(0,1)
        thetaB = []
        for i in range(self.B):
            thetaB.append(self.thetab(out[i]))
        return np.array(thetaB)
    
    def beta(self):
        thetaB = self.thetaB()
        nume = np.sum([(2*self.A@thetab+self.v).T@np.linalg.solve(self.Sigma0,thetab-self.mu0)+2*np.trace(self.A) for thetab in thetaB]) 
        deno = np.sum([np.linalg.norm(2*self.A@thetab+self.v)**2 for thetab in thetaB])
        return (nume/deno), thetaB


class BetaEstimatorKL:
    
    def __init__(self,data, m, grad_m, grad_t, hess_t, grad_b, mean0, var0, varx, d=1, p=1):
        self.data = data
        self.p = p
        self.d = d
        self.n = self.data.shape[0]
        
        self.m = m
        self.grad_m = grad_m
        self.grad_t = grad_t
        self.hess_t = hess_t
        self.grad_b = grad_b

        self.mean0 = np.array([mean0])
        self.sigma0 = np.array([1/var0])
        self.varx = varx

        self.A = np.sum([self.Ax(x) for x in self.data],axis=0)/self.n
        self.v = 2*np.sum([self.vx(x) for x in self.data],axis=0)/self.n

        self.var_sm = (self.sigma0+(self.n/self.varx))**-1
        self.mu_sm = self.var_sm*((self.mean0*self.sigma0)+(np.sum(self.data,axis=0)/self.varx))


    def Ax(self, x):
        return self.grad_t(x).T@self.m(x)@self.m(x).T@self.grad_t(x)
    
    def vx(self, x):
        v1 = (self.grad_t(x).T@self.m(x)@self.m(x).T@self.grad_b(x))
        div_mm = np.array([[np.sum([[self.grad_m(x)[i,j,i]*self.m(x)[p,j] + self.grad_m(x)[p,j,i]*self.m(x)[i,j]  for i in range(self.d)]for j in range(self.d)])] for p in range(self.p)])
        v2 = div_mm.T@self.grad_t(x)
        v3 = np.array([[np.trace(self.m(x)@self.m(x).T@self.hess_t(x)[:,:,p])] for p in range(self.p)])
        return v1+v2+v3

    def var_dsm(self,b):
        return (self.sigma0 + 2*b*self.n*self.A)**-1

    def mu_dsm(self,b):
        return self.var_dsm(b)*(self.mean0*self.sigma0-b*self.n*self.v)
    
    def KLb(self,b):
        return (np.log(self.var_dsm(b)/self.var_sm)/2 + (self.var_sm + (self.mu_sm-self.mu_dsm(b))**2)/(2*self.var_dsm(b))-1/2)[0][0]
    
    def KLf(self,b):
        return (np.log(self.var_sm/self.var_dsm(b))/2 + (self.var_dsm(b) + (self.mu_dsm(b)-self.mu_sm)**2)/(2*self.var_sm)-1/2)[0][0]
        
    def beta(self, beta0=0, b=True):
        if b:
            res = minimize(self.KLb, beta0)
        else:
            res = minimize(self.KLf, beta0)
        return res.x




