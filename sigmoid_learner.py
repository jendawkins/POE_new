from helper import *
import numpy as np
import scipy.stats as st

class SigLearner():
    def __init__(self,true_a,true_b,MEAS_VAR,PROC_VAR,AVAR,BVAR,GIBBS_VAR,NSAMPS,NPTSPERSAMP):
        self.avar = AVAR
        self.bvar = BVAR
        self.mvar = MEAS_VAR
        self.pvar = PROC_VAR
        self.gvar = GIBBS_VAR
        self.true_a = true_a
        self.true_b = true_b
        self.states, self.observations, self.time, self.f = generate_data(
            self.true_a, self.true_b, self.mvar, self.pvar, self.true_a,NSAMPS,NPTSPERSAMP)

    def ptheta(self, a,b,x):
        p1 = -0.5*((a-1)**2)*(1/self.avar) + -0.5*((b-1)**2)*(1/self.bvar)
    #     p1 = np.log((st.norm(1,avar).pdf(a)) + (st.norm(1,bvar).pdf(b)))
        p2 = 0
        for i,xi in enumerate(range(len(x)-1)):
            p2 += -.5*((sigmoid(xi,yscale = a, xscale = b)-x[i+1])**2)*(1/self.pvar)
    #         p2 += np.log(st.norm(sigmoid(xi,yscale = a, xscale = b),pv).pdf(x[i+1]))
        
        return p1 + p2

    def update_theta(self, aold,bold,x):
        anew = aold + np.random.normal(0,np.sqrt(self.gvar))
        pold = self.ptheta(aold,bold,x)
        pnew = self.ptheta(anew,bold,x)
        prob_keep = np.exp(pnew - pold) 
    #     u = np.random.rand()
        if prob_keep > 1:
            aold = anew
        
        bnew = bold + np.random.normal(0,np.sqrt(self.gvar))
        pold = self.ptheta(aold,bold,x)
        pnew = self.ptheta(aold,bnew,x)
        prob_keep = np.exp(pnew - pold) 
        
        if prob_keep > 1:
            bold = bnew
        return aold,bold

    def px(self, x,y,a,b,x0,prior_var=.5, k=3):
        part1 = (-.5*((x[0]-x0)**2)*(1/prior_var))
        part2 = (np.sum((y-x)**2)*(-.5)*(1/self.mvar))
        part3 = 0
        for i in range(len(x)-1):
            part3 += (-0.5*((x[i+1]-sigmoid(x[i],a,b))**2)*(1/self.pvar))
        return part1 + part2 + part3 

    def update_x(self, x,y,a,b,x0):
        x = x.astype(float)
        xp = x.copy()
        for i in range(len(x)): 
            xnext = x[i] + np.random.normal(0,np.sqrt(self.gvar))
            xp[i] = xnext
            num=self.px(xp,y,a,b,x0)
            dem = self.px(x,y,a,b,x0)
            prob_keep = np.exp(num - dem)

    #         u = np.random.rand()
    #         import pdb; pdb.set_trace()
            if prob_keep > 1:
                x[i]=xp[i]
        return x
    
    def run(self,gibbs_steps=500):
        trace_theta = []
        trace_x = []
        a = np.random.normal(1, np.sqrt(self.avar))
        b = np.random.normal(1, np.sqrt(self.bvar))
        for i, y in enumerate(self.observations):
            x0 = self.states[i][0]
            x = np.random.normal(x0,10,size = y.shape)
            thetavec = []
            xxvec = []
            for s in range(gibbs_steps):
                anew, bnew = self.update_theta(a, b, x)
                xnew = self.update_x(x, y, anew, bnew, x0)
                thetavec.append((anew, bnew))
                xxvec.append(xnew)
                x = xnew
                a = anew
                b = bnew
            trace_theta.append(thetavec)
            trace_x.append(xxvec)
        return trace_x, trace_theta
