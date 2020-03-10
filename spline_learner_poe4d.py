from helper import *
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import itertools

class SplineLearnerPOE_4D():
    def __init__(self, num_bact=3, MEAS_VAR=5, PROC_VAR=100, THETA_VAR=100, AVAR=100, BVAR=100, POE_VAR=100, NSAMPS=2, NPTSPERSAMP=10,DT=.1, gr = .1):
        
        self.true_a = -100*np.random.randn(num_bact,num_bact) + 10
        self.true_b = 1 *np.random.randn(num_bact, num_bact) + 1e-7
        self.mvar = MEAS_VAR
        self.pvar = PROC_VAR
        self.avar = AVAR
        self.bvar = BVAR
        self.dt = DT
        self.plot_iter = 10
        self.num_mice = NSAMPS
        self.num_states = NPTSPERSAMP
        self.num_bugs = num_bact
        self.gr = gr
        self.theta_var = THETA_VAR
        self.gibbs_var = 100

        self.alpha = 2
        
        self.k = 3
        self.states, self.observations, self.time, self.f = generate_data4D(
            self.true_a, self.true_b, self.mvar, self.pvar*self.dt, np.max(np.abs(self.true_a.flatten())), self.gr, self.num_mice, self.num_states, self.num_bugs, self.dt)

        self.X1 = np.concatenate(self.states,0)

        # self.X = diag_block_mat(self.X1, len(self.Y))
        
        self.num_knots = self.num_states-self.k

        self.mu_betas = np.array([[np.mean(self.f[i]*self.f[j])*np.ones(self.num_knots) for i in range(self.num_bugs)] for j in range(self.num_bugs)]).T

        # self.states --> bugs x mice x timepoints

        # all_states = [np.concatenate(self.states[i,:,:]) for i in range(self.num_bugs)]        
        self.knots = np.array([[np.linspace(min(self.X1[:,i]*self.X1[:,j]) + .01, max(self.X1[:,i]*self.X1[:,j])-.01, self.num_knots) for i in range(self.num_bugs)] for j in range(self.num_bugs)])

        self.poe_var = POE_VAR*np.eye(num_bact*(NPTSPERSAMP-1))
        self.beta_poevar = self.poe_var / (self.alpha - 1)

        self.pvar = self.pvar*np.eye(num_bact*(NPTSPERSAMP-1))
        self.mvar = self.mvar*np.eye(num_bact*(NPTSPERSAMP))

        self.beta_mvar = self.mvar / (self.alpha - 1)
        self.beta_pvar = self.pvar / (self.alpha - 1)

        self.states = np.transpose(self.states, (1,2,0))
        self.observations = np.transpose(self.observations, (1,2,0))
        self.true_bmat1 = [self.calc_bmat(self.states[:-1,:,i]) for i in range(self.num_mice)]
        # self.true_betas = np.linalg.lstsq(self.true_bmat1, self.Y)

        Y_est = [(self.states[1:,:,i]-self.states[:-1, :,i] - self.gr*self.states[:-1,:, i]*self.dt)/self.dt for i in range(self.num_mice)]
        Y_est = [Y_est[i].flatten(order='F') for i in range(self.num_mice)]
        self.true_betas = [np.linalg.lstsq(self.true_bmat1[i], Y_est[i])[0] for i in range(self.num_mice)]



    def bsplines(self, xi, x, bug1, bug2, k=3):
        # knots = np.linspace(x.min() + .01, x.max()-.01, num_knots)
        bmat = np.zeros((len(self.knots[bug1,bug2]), k))
        for ki in range(k):
            for i in np.arange(len(self.knots[bug1,bug2])-ki-2, 0, -1):
                if ki == 0:
                    if self.knots[bug1, bug2][i] <= xi <= self.knots[bug1, bug2][i+1]:
                        bmat[i, ki] = 1
                    else:
                        bmat[i, ki] = 0
                else:
                    bmat[i, ki] = ((xi-self.knots[bug1, bug2][i])*bmat[i, ki-1])/(self.knots[bug1, bug2][i+ki+1-1]-self.knots[bug1, bug2][i]) + (
                        (self.knots[bug1, bug2][i+ki+1] - xi)*bmat[i+1, ki-1])/(self.knots[bug1, bug2][i+ki+1]-self.knots[bug1, bug2][i+1])
        return bmat

    def calc_bmat(self,X,k=3):
        bmat = np.zeros((X.shape[0],self.num_bugs,self.num_bugs,self.num_knots))
        for t in range(X.shape[0]):
            try:
                bmat_mini = np.array([[self.bsplines(X[t,i]*X[t,j], np.array(X),i,j)[:,k-1] for i in range(X.shape[1])] for j in range(X.shape[1])])
            except:
                import pdb; pdb.set_trace()
            bmat[t,:,:,:] = bmat_mini
        bmat_full = []
        for i in range(self.num_bugs):
            r1 = np.array([bmat[t, i, :, :].flatten()
                           for t in range(X.shape[0])])
            bmat_full.append(r1)
        bmat_fin = diag_mat(bmat_full)
        return bmat_fin

    # Redo overleaf by keeping f1 and f2
    def update_theta(self, states, f2):
        x = states
        bmat = self.calc_bmat(x[:-1])
        mu_thetas = self.mu_betas.flatten(order='F')

        # g1 = bmat@mu_thetas
        # import pdb; pdb.set_trace()
        sig_post = np.linalg.inv((1/self.theta_var)*np.eye(bmat.shape[1]) + self.dt*bmat.T@np.linalg.inv(self.pvar*self.dt)@bmat*self.dt + self.dt*bmat.T@np.linalg.inv(self.poe_var)@bmat*self.dt)

        Y_est = x[1:, :]-x[:-1, :] - self.gr*x[:-1, :]*self.dt
        Y_est = Y_est.flatten(order='F')

        f2_flat = f2.flatten(order = 'F')
        try:
            mu_post = (((1/self.theta_var)*np.eye(bmat.shape[1]))@mu_thetas + (bmat.T@np.linalg.inv(self.pvar*self.dt))@(
                Y_est*self.dt) + (bmat.T@np.linalg.inv(self.poe_var))@((self.dt**2)*f2_flat))@sig_post
        except:
            import pdb; pdb.set_trace()
        return mu_post, sig_post

    def px(self, x, y, x0, betas, theta_2, prior_var=.5, k=3):
        bmat = self.calc_bmat(x[:-1,:])
###############################################################################
        # g1 = betas@bmat.T
        # g1 = np.reshape(g1,(x.shape[0]-1,x.shape[1]),order = 'F')

        # f1 = (x[:-1, :] + x[:-1, :]*self.dt * self.gr) + self.dt*g1
        # f2 = x[:-1, :] + x[:-1, :]*self.dt*self.gr + self.dt * \
        #     sigmoid2d(x[:-1, :], theta_2[0], theta_2[1])
        # xy = ((x[1:, :] - x[:-1, :] - x[:-1, :]*self.gr*self.dt)/self.dt)
        # part1 = -.5*((x[0, :]-x0)**2)*(1/prior_var)
        # pvar=[self.pvar[(xy.shape[0])*i: (xy.shape[0])*(i+1), (xy.shape[0])*i: (xy.shape[0])*(i+1)] for i in range(self.num_bugs)]
        # part2 = np.array([(-.5)*(xy[:, i]-g1[:, i]).T@(np.linalg.inv(self.dt * pvar[i]))@(xy[:, i]-g1[:, i]) for i in range(self.num_bugs)])
        # mvar = [self.mvar[(x.shape[0])*i: (x.shape[0])*(i+1), (x.shape[0])
        #                   * i: (x.shape[0])*(i+1)] for i in range(self.num_bugs)]
        # part3 = np.array([-0.5*((y[:,i]-x[:,i]).T@(np.linalg.inv(mvar[i]))@(y[:,i]-x[:,i])) for i in range(self.num_bugs)])
        # poevar = [self.poe_var[(xy.shape[0])*i: (xy.shape[0])*(i+1), (xy.shape[0])
        #                   * i: (xy.shape[0])*(i+1)] for i in range(self.num_bugs)]
        # part4 = np.array([(-.5)*(f1[:, i]-f2[:, i]).T@(np.linalg.inv(self.dt * poevar[i]))
        #          @(f1[:, i]-f2[:, i]) for i in range(self.num_bugs)])

        # try1 = np.array([part1, part2, part3, part4])
##################################################################################

        f1 = (x[:-1,:] + x[:-1,:]*self.dt*self.gr).flatten(order = 'F') + self.dt*(betas@bmat.T)
        f2 = x[:-1, :] + x[:-1, :]*self.dt*self.gr + self.dt * \
            sigmoid2d(x[:-1,:],theta_2[0],theta_2[1])
        f2 = f2.flatten(order = 'F')
        xy = ((x[1:,:] - x[:-1,:] - x[:-1,:]*self.gr*self.dt)/self.dt).flatten(order = 'F')
        # import pdb
        # pdb.set_trace()
        part1 = -.5*((x[0,:]-x0)@(x[0,:]-x0))*(1/prior_var)
        # import pdb; pdb.set_trace()
        part2 = (-.5)*(xy-betas@bmat.T)@(np.linalg.inv(self.dt*self.pvar))@(xy-betas@bmat.T)
        y = y.flatten(order = 'F')
        x = x.flatten(order = 'F')
        part3 = -0.5*((y-x).T@(np.linalg.inv(self.mvar))@(y-x))
        part4 = -.5*((f1-f2).T@np.linalg.inv(self.poe_var)@(f1-f2))

        try2 = np.array([part1, part2, part3, part4])

        return try2

    def update_x(self, x, y, x0, betas, theta_2):
        x = x.astype(float)
        xp = x.copy()

        # import pdb; pdb.set_trace()
        mvar = np.delete(self.mvar, list(range(self.num_states-1,self.mvar.shape[0],self.num_states-1)),axis=0)
        mvar = np.delete(mvar, list(
            range(self.num_states-1, self.mvar.shape[1], self.num_states-1)), axis=1)

        sig = np.linalg.inv(np.linalg.inv(self.pvar*self.dt) +  np.linalg.inv(mvar))
        # sig = 1/(1/(self.pvar*self.dt) + 1/self.mvar)

        x1next = np.random.normal(x[0,:], np.sqrt(self.gibbs_var))
        xp[0,:] = x1next

        num = self.px(xp, y, x0, betas, theta_2)
        dem = self.px(x, y, x0, betas, theta_2)

        prob_keep = np.exp(np.sum(num,0) - np.sum(dem,0))

        # idxs = np.where(prob_keep > 1)
        # x[0,idxs] = xp[0,idxs]

        if prob_keep > 1:
            x[0,:] = xp[0,:]

        # import pdb; pdb.set_trace()
        proposed_x = np.zeros(x.shape)
        proposed_x[0,:] = x1next
        for i in range(1, x.shape[0]):
            mx = np.expand_dims(x[i-1,:],0)
            mvar1 = self.mvar[i-1:self.mvar.shape[0]:self.num_states,i-1:self.mvar.shape[1]:self.num_states]
            pvar1 = self.pvar[i-1:self.pvar.shape[0]:self.num_states -1, i-1:self.pvar.shape[1]:self.num_states-1]
            
            sig1 = sig[i-1:sig.shape[0]:self.num_states -1, i-1:sig.shape[1]:self.num_states-1]
            mu_xi = (np.linalg.inv(pvar1*self.dt)@(x[i-1, :] + self.dt *(x[i-1, :]*self.gr + betas@self.calc_bmat(mx).T)) + (np.linalg.inv(mvar1)@y[i]))@sig1

            # xnext = x[i] + np.random.normal(0,np.sqrt(self.gvar))
            xnext = st.multivariate_normal(mu_xi, sig1).rvs()
            # xnext = (np.random.normal(x[i-1],np.sqrt(self.pvar)) + np.random.normal(y[i],np.sqrt(self.mvar)))/2
            xp[i,:] = xnext
            
            num,num2 = self.px(xp, y, x0, betas, theta_2)
            dem,dem2 = self.px(x, y, x0, betas, theta_2)
            prob_keep = np.exp(np.sum(num, 0) - np.sum(dem, 0))
            # if i ==0:
            # idxs = np.where(prob_keep > 1)
            # x[i, idxs] = xp[i, idxs]
            if prob_keep > 1:
                x[i,:] = xp[i,:]
            # import pdb; pdb.set_trace()
            proposed_x[i, :] = xnext
        return x, proposed_x
    
    def px2(self,states,theta,f1):

        xin = states[:-1,:]
        xin_f = xin.flatten(order = 'F')
        f1a = ((f1 - xin_f)/self.dt - xin_f*self.gr)
        a=theta[0]
        b=theta[1]
        part1=-.5*(sigmoid2d(xin, a, b).flatten(order = 'F')-f1a).T@np.linalg.inv(
            self.poe_var)@(sigmoid2d(xin, a, b).flatten(order='F')-f1a)

        part2 = 0
        #######################################
        # xin = states[:-1,:]
        # f1b = np.reshape(f1,xin.shape,order = 'F')
        # f1a = ((f1b - xin)/self.dt - xin*self.gr)

        # a=theta[0]
        # b=theta[1]
        # poevar = [self.poe_var[(xin.shape[0])*i: (xin.shape[0])*(i+1), (xin.shape[0])
        #                     * i: (xin.shape[0])*(i+1)] for i in range(self.num_bugs)]
        # import pdb; pdb.set_trace()
        # part1=[-.5*(sigmoid2d(xin, a, b)-f1a)[:,i].T@np.linalg.inv(
        #     poevar[i])@(sigmoid2d(xin, a, b)-f1a)[:,i] for i in range(self.num_bugs)]

        # part2 = 0
        # import pdb; pdb.set_trace()
        return part1 + part2

    def update_f2(self,states,theta,f1):
        anew = theta[0] + np.random.normal(0,self.avar, size = (self.num_bugs,self.num_bugs))
        pold = self.px2(states,theta,f1)
        pnew = self.px2(states,[anew, theta[1]], f1)
        prob_keep = np.exp(pnew - pold)
        import pdb; pdb.set_trace()
        if prob_keep > 1:
            theta[0] = anew
        bnew = theta[1] + \
            np.random.normal(0, self.bvar, size=(self.num_bugs, self.num_bugs))
        pold=self.px2(states, theta, f1)
        pnew=self.px2(states, [theta[0], bnew], f1)
        prob_keep = np.exp(pnew-pold)

        import pdb; pdb.set_trace()
        if prob_keep > 1:
            theta[1] = bnew
        return theta

    def update_poe(self,f1, f2):
        # import pdb; pdb.set_trace()
        f1 = np.expand_dims(f1,0)
        f2 = np.expand_dims(f2,0)
        beta_post = self.beta_poevar + 0.5*(f1-f2).T@(f1-f2)
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()*np.eye(len(f1))

    def update_pvar(self,states, f1):

        f1 = np.expand_dims(f1,0)
        states = np.expand_dims(states.flatten(order = 'F'),0)
        beta_post = self.beta_pvar + 0.5*(states-f1).T@(states-f1)
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()

    def update_mvar(self, states, obs):
        beta_m_post = self.beta_mvar + 0.5*(obs - states).T@(obs - states)
        alpha_m_post = self.alpha + 1/2
        return st.invgamma(alpha_m_post, beta_m_post).rvs()

    
    def run(self,gibbs_steps=500, train_x = True, train_f1 = True, train_f2 = True, train_var = True, plot = True):
        self.trace_a = []
        self.trace_b =[]
        self.trace_x=[]
        self.trace_f1=[]
        self.trace_f2=[]
        self.trace_beta=[]
        
        for i in range(self.observations.shape[-1]):
            y = self.observations[:,:,i]
            x0 = np.random.normal(self.observations[0, :, i], np.sqrt(100))
            x = np.random.normal(x0, 10, size=(self.num_states, self.num_bugs))


            f1 = np.random.normal(0, np.sqrt(self.theta_var),
                                size=((self.num_states-1)*self.num_bugs))
            f2 = np.random.normal(0, np.sqrt(self.theta_var),
                                  size=((self.num_states-1)*self.num_bugs))

            theta2 = [np.random.normal(0, self.avar, size=(self.num_bugs, self.num_bugs)),
                       np.random.normal(0, self.bvar, size=(self.num_bugs, self.num_bugs))]

            if train_var:
                # import pdb; pdb.set_trace()
                # self.mvar = self.update_mvar(x,y)
                self.pvar = self.update_pvar(
                    x[1:, :].flatten(order='F'), f1.flatten(order='F'))
                self.poe_var = self.update_poe(
                    f1.flatten(order='F'), f2.flatten(order='F'))
                # import pdb
                # pdb.set_trace()

            if plot:
                fig, axes = plt.subplots(self.num_bugs,1,sharex = True, figsize = (15,15))
                # plt.plot(x,label = 'Random starting states')
                for k in range(self.num_bugs):
                    axes[k].plot(self.states[:,k,i], label='True states', c = 'g')
                    axes[k].plot(self.observations[:,k,i],label = 'Observations', c = 'r')
                    plt.xlabel('Time (t)')
                    plt.ylabel('States (x)')
                    axes[k].set_title('Bug ' + str(k))
                    axes[k].legend()
                plt.show()
                    
            # x = np.insert(self.observations[i][1:], 0, x0) + np.random.normal(0, self.mvar, size=self.states[0].shape[0])
            if not train_x:
                # x = np.insert(self.observations[i][1:],0,x0) + np.random.normal(0, 1,size = self.states[0].shape[0])
                x = self.states[:,:,i]
            # self.poe_var = (1e7)*np.eye(len(self.states[0]))
            betavec = []
            xxvec = []
            f2vec = []
            avec=[]
            bvec=[]
            f1vec=[]
            import time
            end_ = [True]
            for s in range(gibbs_steps):
                end_.append(True)
                start = time.time()
                # Update theta 1
                end1 = time.time()
                if train_f1:
                    # self.pvar = 1
                    mu_theta_n, sig_theta = self.update_theta(x,f2)
                    betas_plot = st.multivariate_normal(mu_theta_n.squeeze(), sig_theta).rvs(size = 100)
                    betas = betas_plot[0,:]
                    if s > 0:
                        if sum(abs(mu_theta_n-mu_theta))>=.0001:
                            end_[-1] = False

                    mu_theta = mu_theta_n

                else:
                    mu_theta = self.true_betas[i]

                    betas_plot = st.multivariate_normal(
                        mu_theta.squeeze(), 10*np.eye(len(mu_theta))).rvs(size=100)

                    betas = mu_theta
                # print('update theta: ' + str(end1-start))
            
                if s % self.plot_iter == 0 and train_f1 and plot:
                    
                    fig, axes = plt.subplots(self.num_bugs, self.num_bugs, sharex = True, figsize = (15,15))

                    m_plot = np.reshape(mu_theta, (self.num_knots, self.num_bugs, self.num_bugs), order='F')
                    b_plot = np.reshape(betas_plot, (100,self.num_knots, self.num_bugs, self.num_bugs), order = 'F')
                    tru_plot = np.reshape(
                        self.true_betas[i], (self.num_knots, self.num_bugs, self.num_bugs), order = 'F')

                    # for ii,jj in itertools.product(*[np.arange(self.num_bugs),np.arange(self.num_bugs)]):
                        
                    #     axes[ii, jj].plot(
                    #         b_plot[:, :, jj, ii].T,  linewidth=1, color='0.75')
                    #     axes[ii, jj].plot(m_plot[:, jj, ii],
                    #             label=r'Inferred $\mu_{\theta_{1}}$', c='r', linewidth = 2)
                    #     axes[ii, jj].plot(tru_plot[:,jj,ii], label=r'True $\theta_{1}$', c = 'g')
                    #     axes[ii,jj].set_title('Bugs ' + str(ii) + str(jj))
                    #     plt.xlabel(r'$\theta_{1}$ index')
                    #     plt.ylabel(r'$\theta_{1}$')
                    #     axes[ii, jj].legend()
                    # plt.show()

                if train_x:
                    # import pdb; pdb.set_trace()
                    xnew,proposed_xnew = self.update_x(x, y, x0, betas, theta2)
                    x = xnew
                    x0 = x[0,:]
                    if sum(sum(abs(x-xnew)))>=.0001:
                        end_[-1] = False

                if s == 0:
                    xold = x
                    # plt.title('States, Observation ' + str(i) + ', Step ' + str(s))
                if train_x and s % self.plot_iter == 0:
                    fig, axes = plt.subplots(self.num_bugs, 1, sharex=True, figsize=(15, 15))
                    for bb in range(self.num_bugs):
                        axes[bb].plot(xnew[:,bb],label = 'New Inferred states')
                        axes[bb].plot(xold[:, bb], label='Old Inferred states')
                        axes[bb].plot(proposed_xnew[:, bb], label='Proposed Inferred states')
                        axes[bb].plot(self.states[:, bb, i],label='True states')
                        axes[bb].plot(self.observations[:, bb, i],
                                      label='Observations')
                        axes[bb].set_title('Bug ' + str(bb))
                        plt.xlabel('Time (t)')
                        plt.ylabel('States (x)')
                        axes[bb].legend()
                    plt.show()
                    xold = xnew
                # print('update x: ' + str(end2-start))
                # Calculate f1 based on x and theta 1

                f1 = (x[:-1, :] + x[:-1, :]*self.dt *
                      self.gr).flatten(order='F') + self.dt*(betas@self.calc_bmat(x[:-1,:]).T)
                
                xplot = self.states[:-1, :, i]

                bmat_plot = self.calc_bmat(xplot)
                g1_plot = [np.reshape(betas_plot[bp]@bmat_plot.T, xplot.shape, order = 'F') for bp in range(len(betas_plot))]
                f1_plot = [xplot + xplot*self.dt*self.gr + self.dt*g1_plot[bp] for bp in range(len(betas_plot))]

                g1_mean = np.reshape(mu_theta@bmat_plot.T, xplot.shape, order = 'F')
                f1_mean = xplot + xplot*self.dt*self.gr + self.dt*g1_mean

                g1_true = np.reshape(self.true_betas[i]@self.true_bmat1[i].T, xplot.shape, order = 'F')
                f1_true = xplot + self.dt*xplot*self.gr + self.dt*g1_true

                if s % self.plot_iter == 0 and train_f1 and plot:
                    fig, axes = plt.subplots(1,
                        self.num_bugs, figsize=(15, 15))
                    for bb in range(self.num_bugs):
                        for bp in range(len(f1_plot)):
                            axes[bb].plot(xplot[:,bb],f1_plot[bp][:,bb], c = '0.75', linewidth = .5)
                        
                        axes[bb].plot(xplot[:,bb], f1_mean[:,bb], c = 'r',label = r'Inferred $f_{1}$')
                        axes[bb].plot(xplot[:, bb], f1_true[:, bb],
                                      c='g', label=r'True $f_{1}$')
                        plt.xlabel('x (latent states)')
                        plt.ylabel(r'$f_{1}(x)$')
                        axes[bb].legend()

                    plt.show()

                    fig, axes = plt.subplots(1,
                        self.num_bugs, figsize=(15, 15))
                    for bb in range(self.num_bugs):
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
                    plt.show()

                end_bmat = time.time()
                # print('make f1: ' + str(end_bmat-start))
                # Update f2 based on f1, x
                
                if train_f2:
                    # xin = x[:-1,:]
                    theta2_n = self.update_f2(x,theta2,f1)
                    # f2 = xin + self.dt*(self.gr*xin + sigmoid2d(xin, theta2[0], theta2[1]))
                    
                    if sum(sum(abs(np.array(theta2_n)-np.array(theta2)))).all()>=0.0001:
                        end_[-1] = False
                        # import pdb
                        # pdb.set_trace()
                    theta2 = theta2_n
                else:
                    theta2 = [self.true_a, self.true_b]
                    # xin = xplot
                    # f2 = xin + self.dt * (self.gr*xin + sigmoid2d(xin, self.true_a, self.true_b))
                
                xin = x[:-1, :]
                f2 = xin + self.dt * \
                    (self.gr*xin + sigmoid2d(xin, theta2[0], theta2[1]))

                end3 = time.time()
                # print('update theta 2: ' + str(end3-start))

                g2_plot = sigmoid2d(xplot, theta2[0], theta2[1])
                f2_plot = xplot + self.dt*(self.gr*xplot + g2_plot)

                g2_true = sigmoid2d(xplot, self.true_a, self.true_b)
                f2_true = xplot + self.dt*(self.gr*xplot + g2_true)
                if s % self.plot_iter == 0 and train_f2 and plot:
                    fig, axes = plt.subplots(1,
                        self.num_bugs, figsize=(15, 5))
                    for bb in range(self.num_bugs):
                        axes[bb].plot(xplot[:, bb], f2_plot[:, bb],
                                      c='r', label=r'Inferred $f_{2}$')
                        axes[bb].plot(xplot[:, bb], f2_true[:, bb],
                                      c='g', label=r'True $f_{2}$')
                        plt.xlabel('x (latent states)')
                        plt.ylabel(r'$f_{2}(x)$')
                        axes[bb].legend()
                    plt.show()

                    fig, axes = plt.subplots(1,
                        self.num_bugs, figsize=(15, 5))
                    for bb in range(self.num_bugs):
                        axes[bb].plot(xplot[:, bb], g2_plot[:, bb],
                                    c='r', label=r'Inferred $g_{2}$')
                        axes[bb].plot(xplot[:, bb], g2_true[:, bb],
                                    c='g', label=r'True $g_{2}$')
                        plt.xlabel('x (latent states)')
                        plt.ylabel(r'$g_{2}(x)$')
                        axes[bb].legend()
                    plt.show()

                # fig, axes = plt.subplots(
                #     self.num_bugs, 1, sharex=True, figsize=(15, 15))
                # for bb in range(self.num_bugs):
                #     axes[bb].plot(xplot[:, bb], g2_plot[:, bb],
                #                 label=r'Inferred $g_{2}$')
                #     axes[bb].plot(xplot[:, bb], g2_true[:, bb],
                #                 label=r'True $g_{2}$')
                #     for bp in range(len(f1_plot)):
                #         axes[bb].plot(xplot[:, bb], g1_plot[bp]
                #                         [:, bb], c='0.75', linewidth=.5)
                #     axes[bb].plot(xplot[:, bb], g1_mean[:, bb],
                #                     label=r'Inferred $g_{1}$')
                #     axes[bb].plot(xplot[:, bb], g1_true[:, bb],
                #                     label=r'True $g_{1}$')
                
                #     plt.xlabel('x (latent states)')
                #     plt.ylabel(r'$g_{2}(x)$')
                #     axes[bb].legend()
                # plt.show()

                # import pdb; pdb.set_trace()
                f2 = f2.flatten(order='F')
                if s % self.plot_iter == 0 and train_var:
                    print('Step ' + str(s))
                    print('Meas Var:' + str(np.mean(self.mvar)))
                    print('Proc Var:' + str(np.mean(self.pvar)))
                    print('POE Var:' + str(np.mean(self.poe_var)))
                if train_var:
                    # self.mvar = self.update_mvar(x, y)

                    self.pvar = self.update_pvar(x[1:,:], f1)
                    self.poe_var = self.update_poe(f1, f2)

                betavec.append(betas)
                f1vec.append(f1)
                avec.append(theta2[0])
                bvec.append(theta2[1])
                xxvec.append(x)
                f2vec.append(f2)
                end = time.time()
                # print(str(s) + ' gibbs loop: ' + str(end-start))
                # print('step ' + str(s))
                if len(end_)>50:
                    if all(end_[-50:]) == True:
                        print('Converged')
                        break
                # import pdb; pdb.set_trace()
            self.trace_a.append(avec)
            self.trace_b.append(bvec)
            self.trace_x.append(xxvec)
            self.trace_f1.append(f1vec)
            self.trace_f2.append(f2vec)
            self.trace_beta.append(betavec)
            # import pdb; pdb.set_trace()
            print('Observation ' + str(i) + ' Complete')

