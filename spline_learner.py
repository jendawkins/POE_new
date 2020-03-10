from helper import *
import numpy as np
import scipy.stats as st

class SplineLearner():
    def __init__(self, true_a, true_b, MEAS_VAR, PROC_VAR, THETA_VAR, BVAR, GIBBS_VAR, NSAMPS, NPTSPERSAMP):
        self.theta_var = THETA_VAR
        self.mvar = MEAS_VAR
        self.pvar = PROC_VAR
        self.gvar = GIBBS_VAR
        self.true_a = true_a
        self.true_b = true_b
        self.states, self.observations, self.time, self.f = generate_data(
            self.true_a, self.true_b, self.mvar, self.pbar, self.true_a, NSAMPS, NPTSPERSAMP)

    def bsplines(self, xi, x, num_knots, k=3):
        knots = np.linspace(x.min() + .01, x.max()-.01, num_knots)
        bmat = np.zeros((len(knots), k))
        for ki in range(k):
            for i in np.arange(len(knots)-ki-2, 0, -1):
                if ki == 0:
                    if knots[i] <= xi <= knots[i+1]:
                        bmat[i, ki] = 1
                    else:
                        bmat[i, ki] = 0
                else:
                    bmat[i, ki] = ((xi-knots[i])*bmat[i, ki-1])/(knots[i+ki+1-1]-knots[i]) + (
                        (knots[i+ki+1] - xi)*bmat[i+1, ki-1])/(knots[i+ki+1]-knots[i+1])
        return bmat

    def calc_bmat(self,x,nknots,k=3):
        bmat = np.zeros((len(x), nknots))
        for i, xi in enumerate(x):
            bmat[i, :] = self.bsplines(xi, np.array(x), nknots)[:, k-1]
        return bmat

    def update_theta(self, states, obs):
        bmat = self.calc_bmat(states[:-1], len(states)-2)
        sig_theta = np.linalg.inv(
            (self.pvar/self.theta_var)*np.eye((bmat.shape[1])) + (bmat.T@bmat))
        # import pdb; pdb.set_trace()
        # mu_theta = (1/self.pvar)*(sig_theta@(bmat.T@np.expand_dims(obs, 1)))
        mu_theta = (1/self.pvar)*(sig_theta@(bmat.T@np.expand_dims(states[1:], 1)))
        return mu_theta, sig_theta

    def px(self,x,y,x0,betas,prior_var=.5, k=3):
        bmat = self.calc_bmat(x[:-1],len(x)-2)
        part1 = (-.5*((x[0]-x0)**2)*(1/prior_var))
        part2 = (np.sum((x[1:]-bmat@betas)**2)*(-.5)*(1/self.pvar))
        part3 = (-0.5*((y-x).T@(y-x))*(1/self.mvar))
        return part1 + part2 + part3

    def update_x(self,x,y,x0,betas):
        x = x.astype(float)
        xp = x.copy()
        for i in range(len(x)): 
            xnext = x[i] + np.random.normal(0,np.sqrt(self.gvar))
            xp[i] = xnext
            num=self.px(xp,y,x0,betas)
            dem = self.px(x,y,x0,betas)
            prob_keep = np.exp(num - dem)

            if prob_keep > 1:
                x[i]=xp[i]
        return x
    
    # def gen_true_betas():

    
    def run(self,gibbs_steps=500):
        trace_theta = []
        trace_x = []
        for i, y in enumerate(self.observations):
            x0 = self.states[i][0]
            x = np.random.normal(x0,10,size = y.shape)
            thetavec = []
            xxvec = []
            for s in range(gibbs_steps):
                mu_theta, sig_theta = self.update_theta(x,y)
                betas = st.multivariate_normal(
                    mu_theta.squeeze(), sig_theta).rvs()
                xnew = self.update_x(x,y,x0,betas)
                thetavec.append((mu_theta, sig_theta))
                xxvec.append(xnew)
                x = xnew
            trace_theta.append(thetavec)
            trace_x.append(xxvec)
        return trace_x, trace_theta
