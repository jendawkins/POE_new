from data_loader import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import scipy
import random
import time

from matplotlib.patches import Ellipse
import scipy as sp
import seaborn as sns
from theano import tensor as T

from scipy.stats import multivariate_normal
from scipy.stats import invgamma
import scipy.stats as st

dr = dataReader()

def diag_block_mat_boolindex(L):
    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp))==1
    out = np.zeros(np.asarray(shp)*len(L),dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out

# Make data matrix 
dt = np.subtract(dr.times[1:],dr.times[:-1])    
x = np.zeros((len(dt)*len(dr.mouse_names), len(dr.bacteria.keys())))
y = []
for i,bact in enumerate(dr.bacteria.keys()):
    b = dr.bacteria[bact].values
    b = b/1e5
    # x1 = (dr.bacteria[bact].values.flatten(order = 'F'))
    x[:,i] = (dr.bacteria[bact].values[:-1,:].flatten(order = 'F'))*np.tile(dt,len(dr.mouse_names))
    y.append((dr.bacteria[bact].values[1:,:] - dr.bacteria[bact].values[:-1,:]).flatten(order = 'F'))

Y = np.concatenate(y)
numX = len(Y)/x.shape[0]
tupX = (x,)*int(numX)
X = diag_block_mat_boolindex(tupX)

def k2d2(x,xd,sigma = 1, l = 1):
    val = (sigma**2)*np.exp(-((x-xd).T@(x-xd))/(2*(l**2)))
    # if val == np.inf:
    #     import pdb; pdb.set_trace()
    return val

def covfunc2d(x,xd):
    C = np.zeros((x.shape[0],xd.shape[0]))
    for i in range(x.shape[0]):
        for j in range(xd.shape[0]):
            if len(X.shape)>1 and X.shape[1]>1:
                C[i][j] = k2d2(x[i,:],xd[j,:])
            else:
                C[i][j] = k2d2(x[i],xd[j])
    return C

def pcov(X,sig):
    return np.linalg.inv(np.linalg.inv(covfunc2d(X,X)+ .00005*np.eye(X.shape[0])) + np.linalg.inv(np.eye(X.shape[0])*sig))
def pmean(X,Y, sig):
    return pcov(X,sig)@np.linalg.inv(np.eye(X.shape[0])*sig)@Y

def isDiag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)


# Joint Model
X = X[:100,:20]
Y = Y[:100]

I = len(dr.bacteria.keys())
SAMPLES = 2
N = 100
trace = []

alpha = len(X)/2
alpha_y = alpha

muy = 1.1
beta_y = muy*(alpha-1)

py = st.invgamma(alpha,scale=beta_y)

mua = 1
beta_a = mua*(alpha-1)
pa = st.invgamma(alpha, scale=beta_a)
pb = st.invgamma(alpha, scale=beta_a)
alpha_a = alpha

y_samp = py.rvs()
a_samp = pa.rvs(size = I)
b_samp = pb.rvs(size = I)

poe = py.rvs()

muGP = pmean(X,Y,y_samp)
covGP = pcov(X,y_samp)

muGPpost = muGP
cGPpost = covGP

I = len(dr.bacteria.keys())

f1 = np.zeros(Y.shape)



for i in range(SAMPLES):

# posterior on f2
    covGPpost = np.linalg.inv((1/poe)*np.eye(len(X)) + pcov(X,y_samp))
    muGPpost = covGPpost@((1/poe)*f1 + covGP@pmean(X,Y,y_samp))
    
    import pdb; pdb.set_trace()
    f2 = st.multivariate_normal(muGPpost.squeeze(), covGPpost).rvs()
    f2 = np.expand_dims(f2,1)
# Posterior on A
    covA = np.linalg.inv(X.T @((1/y_samp)*np.eye(len(X)))@ X + (np.diag(1/a_samp)) + X.T@(np.eye(len(X))*(1/poe)@X))
    muA = (X.T@ ((1/y_samp)*np.eye(len(X))) @Y +  (1/poe)*(X.T@f2)).T@covA

    A = st.multivariate_normal(muA.squeeze(), covA).rvs()
    A = np.expand_dims(A,1)
    f1 = X@A

    trace.append((cGPpost, muGPpost, muA, covA, f1, f2))  