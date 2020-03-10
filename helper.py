import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

def sigmoid(t, yscale=20, xscale=.1):
    y = yscale/(1+np.exp(-xscale*(t)))
    return y

def sigmoid2d(x,a,b):
    ar = np.array([np.sum([sigmoid(x[:, i]*x[:, j],a[i,j],b[i,j]) for j in range(a.shape[1])], 0) for i in range(a.shape[0])]).T
    return ar

def mm(x,y,a,b,dt,i):
    return x*y + dt*(a*x*y)/(b + i*x*y)


def michaelis_menten(x, a, b, ii):
    if len(x.shape) < 2:
        x = np.expand_dims(x,0)
#     ar = np.array([np.sum([a[i,j]*x[:,i]*x[:,j]/(b[i,j] + ii*x[:,i]*x[:,j]) for j in range(a.shape[0])],1) for i in range(a.shape[0])]).T
    out2 = np.array([[(a[i, j]*x[:, i]*x[:, j]*b[i, j])/(b[i, j] + ii*x[:, i]) for j in range(a.shape[0])] for i in range(a.shape[0])])

    for k in range(out2.shape[-1]):
        np.fill_diagonal(out2[:, :, k], np.array(
            [a[i, i]*(x[k, i]**2) for i in range(a.shape[0])]))
    return np.sum(out2, 1).T

def generate_data_MM(a, b, r, xinn, resolution, ii, mvar, pvar, nsamps, nptspersample, seed=4):

    # a2 = 0
    # b2 = np.sum(b, 1)
    num_bugs = a.shape[0]
    # a2, b2 = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
    xtot = []
    ytot = []
    for nn in range(nsamps):
        np.random.seed(seed)
        xin = xinn[nn]

        x_all = [xin]
        y_all = [xin + np.random.normal(0, np.sqrt(mvar))]
        for n in range(nptspersample):
            xout = xin + resolution*(xin*r[nn] + michaelis_menten(xin, a, b, ii)) 
            # pvar = pvar*abs(xout - xin)*10
            # pvar = 0
            # mvar = mvar*abs(xout - xin)*10
            xout = xout + np.random.normal(0,scale = np.sqrt(pvar))
            yout = xout + np.random.normal(0,np.sqrt(mvar))

            zidxs = np.where(xout < 0)
            xout[zidxs] = 0
            yout[zidxs] = 0
            x_all.append(xout)
            y_all.append(yout)
            xin = xout

        x_all = np.concatenate(x_all)
        y_all = np.concatenate(y_all)
        xtot.append(x_all)
        ytot.append(y_all)
    return xtot,ytot

def generate_data(a, b, mvar, pvar, xrange, nsamples=25, nptspersample=20, resolution=.1):
    t = np.linspace( - np.abs(xrange)/2, np.abs(xrange)/2,np.abs(xrange)/resolution)
    f = sigmoid(t, a, b) - xrange/2
    xvec = []
    yvec = []
    ns = 0
    while ns < nsamples:
        xin = [(f.max()-f.min())*np.random.rand() + f.min()]
        # xout = []
        for iterate in range(nptspersample):
            xin.append(xin[-1] + resolution*sigmoid(xin[-1],a,b) + np.random.normal(0,np.sqrt(pvar)))
        #     if iterate < 99:
        #         xin.append(xout[-1])
            # import pdb; pdb.set_trace()

        xin = np.array(xin)
        y = xin + np.random.normal(0, np.sqrt(mvar), size=xin.shape)
        if np.var(xin)/np.var(y) > .5:
            ns += 1
            xvec.append(xin)
            yvec.append(y)

    return xvec, yvec, t, f

def diag_block_mat(X, n):
    numX = n/X.shape[0]
    L = (X,)*int(numX)

    shp = L[0].shape
    mask = np.kron(np.eye(len(L)), np.ones(shp)) == 1
    out = np.zeros(np.asarray(shp)*len(L), dtype=int)
    out[mask] = np.concatenate(L).ravel()
    return out


def generate_data4D(a, b, mvar, pvar, xrange, gr=.1, nsamples=25, nptspersample=20, num_bugs=3, resolution=.1):
    t = np.linspace(- np.abs(xrange)/2, np.abs(xrange) /
                    2, np.abs(xrange)/resolution)

    f = [[sigmoid(t, a[i, j], b[i, j]) - xrange/2 for i in range(a.shape[0])]
         for j in range(a.shape[1])]
    f = np.array(f)
    xvec = []
    yvec = []
    ns = 0
    while ns < nsamples:
        xin = [(f.max()-f.min())*np.random.rand() + f.min()]
        # xout = []
        xall = []
        xin = (f.max(1).max(1) - f.min(1).min(1)) * \
            np.random.rand(a.shape[1]) + f.min(1).min(1)
        for iterate in range(nptspersample):
            xout = np.zeros((a.shape[0]))
            for i in range(a.shape[0]):
                xout[i] = xin[i] + resolution*(gr*xin[i] + np.sum(
                    [sigmoid(xin[i]*xin[j], a[i, j], b[i, j]) for j in range(a.shape[1])]))
            xall.append(xout)
            xin = xout

        xall = np.array(xall)
        y = xall + np.random.normal(0, np.sqrt(mvar), size=xall.shape)
        if np.var(xin)/np.var(y) > .5:
            ns += 1
            xvec.append(xall)
            yvec.append(y)

    xvec = xvec
    yvec = yvec
    # import pdb; pdb.set_trace()
    return xvec, yvec, t, f

def spline_change_pts(states):
    nvec = []
    for s in states:
        nn = np.where([np.abs(s[i] - s[i+1]) < .0001 for i in range(len(s)-1)])[0]
        if len(nn) == 0:
            n = len(s)
        elif len(nn) > 1:
            #         import pdb; pdb.set_trace()
            n = nn[0]
        else:
            n = nn
        nvec.append(n)
    return np.mean(nvec)/len(states)


def diag_mat(X):
    out = np.zeros((len(X)*X[0].shape[0], len(X)*X[0].shape[1]))
    for i in range(len(X)):
        out[i*X[0].shape[0]:(i+1)*X[0].shape[0], i *
            X[0].shape[1]:(i+1)*X[0].shape[1]] = X[i]
    return out


def plot_states(outdir,xnew, true_states, observations, xold  = None, proposed_xnew = None,ob = None, f2 = None, f1 = None):
    num_bugs = xnew.shape[1]
    fig, axes = plt.subplots(
        num_bugs, 1, sharex=True, figsize=(15, 15))
    for bb in range(num_bugs):
        axes[bb].plot(xnew[:, bb], label='New Inferred states')
        axes[bb].plot(true_states[:, bb],label='True states')
        axes[bb].plot(observations[:, bb],
                        label='Observations')
        # if xold is not None:
        #     axes[bb].plot(xold[:, bb], label='Old Inferred states')
        if proposed_xnew is not None:
            axes[bb].plot(proposed_xnew[:, bb], label='Proposed Inferred states')
        if f2 is not None:
            axes[bb].plot(f2[:,bb], label = 'f2')
        if f1 is not None:
            axes[bb].plot(f1[:,bb], label = 'f1')
        if ob is not None:
            axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
        else:
            axes[bb].set_title('Bug ' + str(bb))
        plt.xlabel('Time (t)')
        plt.ylabel('States (x)')
        axes[bb].legend()
    plt.savefig(outdir + '_states_out.png')

def plot_orig(outdir, states, observations):
    num_bugs = states.shape[1]
    for ob in range(states.shape[-1]):
        fig, axes = plt.subplots(
            num_bugs, 1, sharex=True, figsize=(15, 15))
        for k in range(num_bugs):
            axes[k].plot(states[:, k,ob],
                            label='True states', c='g')
            axes[k].plot(observations[:, k,ob],
                            label='Observations', c='r')
            plt.xlabel('Time (t)')
            plt.ylabel('States (x)')
            axes[k].set_title('Bug ' + str(k))
            axes[k].legend()
        plt.savefig(outdir + '_states_in.png')

def plot_f1(outdir, xplot, bmat_plot, mu_theta, sig_theta, dt, gr, true_states,ob):
    betas_plot = st.multivariate_normal(
        mu_theta.squeeze(), sig_theta).rvs(size=100)

    num_bugs = xplot.shape[-1]

    g1_plot = [np.reshape(betas_plot[bp,:]@bmat_plot.T, xplot.shape, order = 'F') for bp in range(betas_plot.shape[0])]
    f1_plot = [xplot + xplot*dt*gr +
                dt*g1_plot[bp] for bp in range(len(betas_plot))]
    g1_mean = np.reshape(mu_theta@bmat_plot.T, xplot.shape, order = 'F')
    f1_mean = xplot + xplot*dt*gr + dt*g1_mean

    f1_true = true_states[1:,:]
    g1_true = (f1_true - true_states[:-1, :] - true_states[:-1, :]*gr*dt)/dt
    # g1_true = np.reshape(true_betas@true_bmat.T, xplot.shape, order = 'F')
    # f1_true = xplot + dt*xplot*gr + dt*g1_true

    fig, axes = plt.subplots(1,
        num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        for bp in range(len(f1_plot)):
            axes[bb].plot(xplot[:,bb],f1_plot[bp][:,bb], c = '0.75', linewidth = .5)
        
        axes[bb].plot(xplot[:,bb], f1_mean[:,bb], c = 'r',label = r'Inferred $f_{1}$')
        axes[bb].plot(xplot[:, bb], f1_true[:, bb],
                        c='g', label=r'True $f_{1}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$f_{1}(x)$')
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
        axes[bb].legend()

    plt.savefig(outdir + '_f1.png')

    fig, axes = plt.subplots(1,
        num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        for bp in range(len(f1_plot)):
            axes[bb].plot(xplot[:, bb], g1_plot[bp]
                            [:, bb], c='0.75', linewidth=.5)
        axes[bb].plot(xplot[:, bb], g1_mean[:, bb],
                        c='r', label=r'Inferred $g_{1}$')
        axes[bb].plot(xplot[:, bb], g1_true[:, bb],
                        c='g', label=r'True $g_{1}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$g_{1}(x)$')

        axes[bb].legend()
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
    plt.savefig(outdir + '_g1.png')


def plot_f2_linear(outdir, xin, mu2, sig2, true_theta, use_mm, dt, gr,ob):
    xplot = xin

    num_bugs = xin.shape[1]
    g2_mean = michaelis_menten(
        xplot, np.reshape(mu2, (num_bugs, num_bugs), order='F'), np.ones((num_bugs,num_bugs)), use_mm)

    theta2_all = st.multivariate_normal(mu2, sig2).rvs(100)
    theta2_all = [np.reshape(theta2_all[i,:],(num_bugs, num_bugs), order = 'C') for i in range(100)]

    f2_mean = xplot + dt*(gr*xplot + g2_mean)

    g2_plot=[michaelis_menten(
            xin, theta2_all[ii], np.ones((num_bugs, num_bugs)), use_mm) for ii in range(100)]
    f2_plot = [xplot + dt*(gr*xplot + g2_plot[ii]) for ii in range(100)]

    g2_true = michaelis_menten(
        xplot, true_theta[0], true_theta[1], use_mm)
    f2_true = xplot + dt*(gr*xplot + g2_true)

    fig, axes = plt.subplots(1,
                            num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        axes[bb].plot(xplot[:, bb], f2_mean[:, bb],
                        c='r', label=r'Inferred $f_{2}$')
        for bp in range(len(f2_plot)):
            axes[bb].plot(xplot[:, bb], f2_plot[bp]
                        [:, bb], c='0.75', linewidth=.5)

        axes[bb].plot(xplot[:, bb], f2_true[:, bb],
                    c='g', label=r'True $f_{2}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$f_{1}(x)$')
        axes[bb].legend()
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))

    plt.savefig(outdir + '_f2_linear.png')

    fig, axes = plt.subplots(1,
                            num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        for bp in range(len(f2_plot)):
            axes[bb].plot(xplot[:, bb], g2_plot[bp]
                        [:, bb], c='0.75', linewidth=.5)
        axes[bb].plot(xplot[:, bb], g2_mean[:, bb],
                    c='r', label=r'Inferred $g_{2}$')
        axes[bb].plot(xplot[:, bb], g2_true[:, bb],
                    c='g', label=r'True $g_{2}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$g_{1}(x)$')
        axes[bb].legend()
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
    plt.savefig(outdir + '_g2_linear.png')


def plot_f2(outdir, xin, theta2, theta_true, use_mm, dt, gr,ob):
    xplot = xin
    num_bugs = xin.shape[1]
    g2_plot = michaelis_menten(
        xin, theta2[0], theta2[1], use_mm)
    f2_plot = xplot + dt*(gr*xplot + g2_plot)
    g2_true = michaelis_menten(
        xplot, theta_true[0], theta_true[1], use_mm)
    f2_true = xplot + dt*(gr*xplot + g2_true)
    fig, axes = plt.subplots(1,num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        axes[bb].plot(xplot[:, bb], f2_plot[:, bb],
                        c='r', label=r'Inferred $f_{2}$')
        axes[bb].plot(xplot[:, bb], f2_true[:, bb],
                        c='g', label=r'True $f_{2}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$f_{2}(x)$')
        axes[bb].legend()
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
    plt.savefig(outdir + '_f2.png')

    fig, axes = plt.subplots(1,num_bugs, figsize=(15, 5))
    for bb in range(num_bugs):
        axes[bb].plot(xplot[:, bb], g2_plot[:, bb],
                        c='r', label=r'Inferred $g_{2}$')
        axes[bb].plot(xplot[:, bb], g2_true[:, bb],
                        c='g', label=r'True $g_{2}$')
        plt.xlabel('x (latent states)')
        plt.ylabel(r'$g_{2}(x)$')
        axes[bb].legend()
        axes[bb].set_title('Bug ' + str(bb) + ', Observation ' + str(ob))
    plt.savefig(outdir + '_g2.png')
