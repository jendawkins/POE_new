from helper import *
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import interpolate


class SplineLearnerPOE_old():
    def __init__(self, true_a=-10, true_b=.1, MEAS_VAR=5, PROC_VAR=100, THETA_VAR=100, AVAR=100, BVAR=100, POE_VAR=100, NSAMPS=10, NPTSPERSAMP=20,DT=.1):
        self.theta_var = THETA_VAR
        self.mvar = MEAS_VAR
        self.pvar = PROC_VAR
        self.avar = AVAR
        self.bvar = BVAR
        self.true_a = true_a
        self.true_b = true_b
        self.dt = DT
        self.plot_iter = 10

        self.alpha = 2
        self.beta_mvar = self.mvar / (self.alpha - 1)
        self.beta_pvar = self.pvar / (self.alpha - 1)
        
        self.k = 3
        self.states, self.observations, self.time, self.f = generate_data(
            self.true_a, self.true_b, self.mvar, self.pvar*self.dt, self.true_a, NSAMPS, NPTSPERSAMP,DT)

        f = [interpolate.interp1d(np.arange(len(self.states[i])),self.states[i]) for i in range(len(self.states))]
        self.plot_states = [f[i](np.linspace(0, len(self.states[i])-1, 100)) for i in range(len(self.states))]

        num_knots = len(self.states[0])-self.k
        self.mu_betas = np.mean(self.f)*np.ones(num_knots)
        # import pdb; pdb.set_trace()
        all_states = np.concatenate(self.states)
        # import pdb; pdb.set_trace()
        self.knots = np.linspace(
            min(all_states) + .01, max(all_states)-.01, num_knots)

        self.poe_var = POE_VAR*np.eye(len(self.states[0]))
        self.beta_poevar = self.poe_var / (self.alpha - 1)

        self.true_bmat1 = [self.calc_bmat(sta[:-1]) for sta in self.states]
        self.true_bmat2 = [self.calc_bmat(sta) for sta in self.states]
        # y = beta*bmat
        self.ys = [(self.states[i][1:]-self.states[i][:-1])/self.dt for i in range(len(self.states))]
        # import pdb; pdb.set_trace()
        self.true_betas = [np.linalg.lstsq(self.true_bmat1[i],self.ys[i])[0] for i in range(len(self.true_bmat1))]

    def bsplines(self, xi, x, k=3):
        # knots = np.linspace(x.min() + .01, x.max()-.01, num_knots)
        bmat = np.zeros((len(self.knots), k))
        for ki in range(k):
            for i in np.arange(len(self.knots)-ki-2, 0, -1):
                if ki == 0:
                    if self.knots[i] <= xi <= self.knots[i+1]:
                        bmat[i, ki] = 1
                    else:
                        bmat[i, ki] = 0
                else:
                    bmat[i, ki] = ((xi-self.knots[i])*bmat[i, ki-1])/(self.knots[i+ki+1-1]-self.knots[i]) + (
                        (self.knots[i+ki+1] - xi)*bmat[i+1, ki-1])/(self.knots[i+ki+1]-self.knots[i+1])
        return bmat

    def calc_bmat(self,x,k=3):
        nknots = len(self.knots)
        bmat = np.zeros((len(x), nknots))
        for i, xi in enumerate(x):
            bmat[i, :] = self.bsplines(xi, np.array(x), nknots)[:, k-1]
        return bmat

    # Redo overleaf by keeping f1 and f2
    def update_theta(self, states, f2):
        bmat = self.calc_bmat(states)
        bmat2 = self.calc_bmat(states[:-1])

        sig_theta = np.linalg.inv(np.eye(bmat.shape[1])*(1/self.theta_var) + (self.dt)*(bmat2.T@(
            (1/(self.pvar*self.dt))*np.eye(bmat2.shape[0]))@bmat2) + (self.dt**2)*bmat.T@np.linalg.inv(self.poe_var)@bmat)

        mu_theta = sig_theta@(self.mu_betas@np.eye(bmat.shape[1])*(1/self.theta_var) + ((states[1:] - states[:-1]))@(
            np.eye(bmat2.shape[0])*(1/(self.pvar*self.dt)))@bmat2 + (self.dt**2)*(bmat.T@np.linalg.inv(self.poe_var)@f2))

        return mu_theta, sig_theta

    def px(self,x,y,x0,betas,prior_var=.5, k=3):
        bmat = self.calc_bmat(x[:-1])
        xy = (x[1:] - x[:-1])/self.dt
        part1 = (-.5*((x[0]-x0)**2)*(1/prior_var))
        part2 = (np.sum((xy-betas@bmat.T)**2)*(-.5)*(1/(self.dt*self.pvar)))
        part3 = (-0.5*((y-x).T@(y-x))*(1/self.mvar))
        # import pdb; pdb.set_trace()
        return part1 + part2 + part3

    def update_x(self,x,y,x0,betas):
        x = x.astype(float)
        xp = x.copy()

        sig = 1/(1/(self.pvar*self.dt) + 1/self.mvar)
        for i in range(1,len(x)): 
            mu_xi = ((1/(self.pvar*self.dt))*(x[i-1] + self.dt*(betas@self.calc_bmat([x[i-1]]).T).item()) + (1/self.mvar)*y[i])*sig
            # xnext = x[i] + np.random.normal(0,np.sqrt(self.gvar))
            xnext = st.norm(mu_xi, np.sqrt(sig)).rvs()
            # xnext = (np.random.normal(x[i-1],np.sqrt(self.pvar)) + np.random.normal(y[i],np.sqrt(self.mvar)))/2
            xp[i] = xnext
            num=self.px(xp,y,x0,betas)
            dem = self.px(x,y,x0,betas)
            prob_keep = np.exp(num - dem)
            # if i ==0:
                # import pdb; pdb.set_trace()
            if prob_keep > 1:
                x[i]=xp[i]
        return x
    
    def px2(self,states,theta,f1):
        f1a = (f1 - states)/self.dt
        a=theta[0]
        b=theta[1]
        part1=-.5*(sigmoid(states, a, b)-f1a).T@np.linalg.inv(
            self.poe_var)@(sigmoid(states, a, b)-f1a)
        # plt.figure()
        # plt.plot(sigmoid(states, a, b),label = 'sigmoid')
        # plt.plot(f1a,label = 'f1')
        # plt.title('Predicted a,b:' + str(theta))
        # plt.show()
        # part2=-.5*(((a+100)**2)*(1/self.avar) + ((b-.1)**2)*(1/self.bvar))
        part2 = 0
        return part1 + part2

    def update_f2(self,states,theta,f1):
        anew = theta[0] + np.random.normal(1,np.sqrt(self.avar))
        pold = self.px2(states,theta,f1)
        pnew = self.px2(states,[anew, theta[1]], f1)
        prob_keep = np.exp(pnew - pold)
        # import pdb; pdb.set_trace()
        if prob_keep > 1:
            theta[0] = anew
        bnew = theta[1] + np.random.normal(1,np.sqrt(self.bvar))
        pold=self.px2(states, theta, f1)
        pnew=self.px2(states, [theta[0], bnew], f1)
        prob_keep = np.exp(pnew-pold)
        # import pdb
        # pdb.set_trace()
        if prob_keep > 1:
            theta[1] = bnew
        return theta

    def update_poe(self,f1, f2):
        beta_post = self.beta_poevar + 0.5*(f1-f2).T@(f1-f2)
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()*np.eye(len(f1))

    def update_pvar(self,states, f1):
        beta_post = self.beta_pvar + 0.5*(states[1:]-f1[:-1]).T@(states[1:]-f1[:-1])
        alpha_post = self.alpha + 1/2
        return st.invgamma(alpha_post, beta_post).rvs()

    def update_mvar(self, states, obs):
        beta_m_post = self.beta_mvar + 0.5*(obs - states).T@(obs - states)
        alpha_m_post = self.alpha + 1/2
        return st.invgamma(alpha_m_post, beta_m_post).rvs()

    
    def run(self,gibbs_steps=50, train_x = True, train_f1 = True, train_f2 = True, train_var = True, plot = True):
        self.trace_a = []
        self.trace_b =[]
        self.trace_x=[]
        self.trace_f1=[]
        self.trace_f2=[]
        self.trace_beta=[]
        
        for i, y in enumerate(self.observations):
            # x0 = self.states[i][0]
            x0 = np.random.normal(self.observations[i][0], np.sqrt(100))
            
            x = np.random.normal(x0, 10, size=self.states[0].shape[0])
            f1 = np.random.normal(0, np.sqrt(self.theta_var), size=self.states[0].shape[0])
            f2 = np.random.normal(0, np.sqrt(self.theta_var), size=self.states[0].shape[0])
            if train_var:
                # import pdb; pdb.set_trace()
                # self.mvar = self.update_mvar(x,y)
                self.pvar = self.update_pvar(x,f1)
                self.poe_var = self.update_poe(f1,f2)
                # import pdb
                # pdb.set_trace()

            if plot:
                plt.figure()
                plt.title('Random starting states')
                plt.plot(x,label = 'Random starting states')
                plt.plot(self.states[i], label='True states')
                plt.plot(self.observations[i],label = 'Observations')
                plt.xlabel('Time (t)')
                plt.ylabel('States (x)')
                plt.legend()
                plt.show()
            # x = np.insert(self.observations[i][1:], 0, x0) + np.random.normal(0, self.mvar, size=self.states[0].shape[0])
            if not train_x:
                # x = np.insert(self.observations[i][1:],0,x0) + np.random.normal(0, 1,size = self.states[0].shape[0])
                x = self.states[i]
            theta2 = [np.random.normal(0,np.sqrt(self.avar)), np.random.normal(0,np.sqrt(self.bvar))]
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
                    betas = st.multivariate_normal(mu_theta_n.squeeze(), sig_theta).rvs()
                    betas_plot = st.multivariate_normal(
                        mu_theta_n.squeeze(), sig_theta).rvs(100)
                    if s > 0:
                        if sum(abs(mu_theta_n-mu_theta))>=.0001:
                            end_[-1] = False

                    mu_theta = mu_theta_n

                else:
                    betas = self.true_betas[i]
                # print('update theta: ' + str(end1-start))
                if s % self.plot_iter == 0 and train_f1 and plot:
                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))

                    plt.plot(range(len(betas)), betas, color='0.75', linewidth=1,
                             label=r'Sampled $\theta_{1}$ from inferred $\mu_{\theta_{1}}$ and $\Sigma_{\theta_{1}}$')
                    plt.plot(range(len(betas)), betas_plot.T,
                             color='0.75', linewidth=1)
                    plt.plot(
                        self.true_betas[i], label=r'True $\theta_{1}$', color='g')
                    plt.plot(
                        mu_theta, label=r'Inferred $\mu_{\theta_{1}}$', color='r')
                    plt.xlabel(r'$\theta_{1}$ index')
                    plt.ylabel(r'$\theta_{1}$')
                    plt.legend()
                    plt.show()

                # self.poe_var = 1*np.eye(len(self.states[0]))
                # Update x based on theta 1

                # f1_part = betas@self.calc_bmat(x[:-1]).T
                if train_x:
                    # if s < 20:
                    #     self.pvar = 1e7
                    xnew = self.update_x(x, y, x0, betas)
                    if sum(abs(x-xnew))>=.0001:
                        end_[-1] = False
                        # import pdb; pdb.set_trace()
                    x = xnew

                f1 = x + self.dt*(betas@self.calc_bmat(x).T)                
                end2 = time.time()
                if s % self.plot_iter == 0 and train_x and plot:
                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    plt.plot(xnew, label='Inferred states')
                    plt.plot(self.states[i], label='True states')
                    plt.plot(self.observations[i], label='Observations')
                    plt.xlabel('Time (t)')
                    plt.ylabel('States (x)')
                    plt.legend()
                    plt.show()
                # print('update x: ' + str(end2-start))
                # Calculate f1 based on x and theta 1

                if s % self.plot_iter == 0 and (train_f1 or train_x) and plot:
                    xplot = self.plot_states[i]
                    f1_plot = [xplot + (self.dt)*(betas_plot[kk, :]@self.calc_bmat(xplot).T)
                               for kk in range(len(betas_plot))]
                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    plt.plot(xplot, np.array(f1_plot).T,
                             color='0.75', linewidth=1)
                    plt.plot(xplot, xplot + (self.dt)*(betas@self.calc_bmat(
                        xplot).T), label=r'Inferred $f_{1}$ Samples', color='0.75', linewidth=1)
                    plt.plot(xplot, xplot + (self.dt)*(
                        self.true_betas[i]@self.calc_bmat(xplot).T), label=r'True $f_{1}$')
                    plt.plot(xplot, xplot + (self.dt)*(mu_theta@self.calc_bmat(
                        xplot).T), label=r'Inferred $f_{1}$ Mean', color='r', linewidth=1)
                    # plt.plot(self.ys[i], label=r'True $f_{1}$')
                    plt.xlabel('True x (latent states)')
                    plt.ylabel(r'$f_{1}(x)$')
                    plt.legend()
                    plt.show()

                    # plt.figure()
                    # plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    # plt.plot((x[1:]-x[:-1])/self.dt, betas@self.calc_bmat(x[:-1]).T,label = r'Inferred $\theta_{1} B_{xin}$')
                    # plt.plot((self.states[i][1:]-self.states[i][:-1])/self.dt, self.true_betas[i]
                    #          @self.true_bmat1[i].T, label=r'True $\theta_{1} B_{xin}$')
                    # # plt.plot(self.ys[i], label=r'True $f_{1}$')
                    # plt.xlabel(r'$(x_{i+1}-x_{i})/\Delta t)$')
                    # plt.ylabel(r'$\theta_{1} B_{xin}$')
                    # plt.legend()
                    # plt.show()

                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    plt.plot(xplot[:-1], (betas_plot[0, :]@self.calc_bmat(xplot[:-1]).T),
                             label=r'Inferred $\theta_{1} B_{xin}$ Samples', color='0.75', linewidth=1)
                    for bet in range(1, betas_plot.shape[0]):
                        plt.plot(
                            xplot[:-1], (betas_plot[bet, :]@self.calc_bmat(xplot[:-1]).T), linewidth=1, color='0.75')
                    plt.plot(xplot[:-1], (mu_theta@self.calc_bmat(xplot[:-1]).T),
                             label=r'Inferred $\theta_{1} B_{xin}$ Mean', color='r')
                    plt.plot(xplot[:-1], self.true_betas[i]@self.calc_bmat(
                        xplot[:-1]).T, label=r'True $\theta_{1} B_{xin}$', color='g')
                    # plt.plot(self.true_betas[i]@self.true_bmat1[i].T, label=r'True $\theta_{1} B_{xin}$')
                    # plt.plot(self.ys[i], label=r'True $f_{1}$')
                    plt.xlabel('True x')
                    plt.ylabel(r'$\theta_{1} B_{xin}$')
                    plt.legend()
                    plt.show()

                end_bmat = time.time()
                # print('make f1: ' + str(end_bmat-start))
                # Update f2 based on f1, x
                if train_f2:
                    theta2_n = self.update_f2(x,theta2,f1)
                    f2 = x + self.dt*sigmoid(x, theta2[0], theta2[1])
                    if sum(abs(np.array(theta2_n)-np.array(theta2)))>=0.0001:
                        end_[-1] = False
                        # import pdb
                        # pdb.set_trace()
                    theta2 = theta2_n
                else:
                    f2 = x + self.dt*sigmoid(x, self.true_a, self.true_b)
                end3 = time.time()
                # print('update theta 2: ' + str(end3-start))
                if s % self.plot_iter == 0 and train_f2 and plot:
                    xplot = self.plot_states[i]
                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    plt.plot(xplot, xplot + (self.dt)*sigmoid(xplot,
                                                              theta2[0], theta2[1]), label=r'Inferred $f_{2}$')
                    plt.plot(xplot, xplot + (self.dt)*sigmoid(xplot,
                                                              self.true_a, self.true_b), label=r'True $f_{2}$')
                    plt.xlabel('True x')
                    plt.ylabel('$f_{2}$')
                    plt.legend()
                    plt.show()

                    # plt.figure()
                    # plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    # plt.plot((x[1:]-x[:-1])/self.dt, sigmoid(x[:-1],theta2[0],theta2[1]),
                    #          label=r'Inferred')
                    # plt.plot((self.states[i][1:]-self.states[i][:-1])/self.dt, sigmoid(
                    #     self.states[i][:-1], self.true_a, self.true_b), label=r'True')
                    # plt.xlabel(r'$(x_{i+1}-x_{i})/\Delta t)$')
                    # plt.ylabel(r'$sigmoid(x_{i},\theta_{2})$')
                    # plt.legend()
                    # plt.show()

                    plt.figure()
                    plt.title('Observation ' + str(i) + ', Step ' + str(s))
                    plt.plot(xplot[:-1], sigmoid(xplot[:-1],
                                                 theta2[0], theta2[1]), label=r'Inferred')
                    plt.plot(xplot[:-1], sigmoid(xplot[:-1],
                                                 self.true_a, self.true_b), label='True')
                    # plt.plot(sigmoid(self.states[i][:-1], self.true_a, self.true_b), label=r'True')
                    plt.xlabel('True x')
                    plt.ylabel(r'$sigmoid(x_{i},\theta_{2})$')
                    plt.legend()
                    plt.show()
                    print(theta2)
                if s % self.plot_iter == 0 and train_var:
                    print('Step ' + str(s))
                    print('Meas Var:' + str(self.mvar))
                    print('Proc Var:' + str(self.pvar))
                    print('POE Var:' + str(self.poe_var[0][0]))
                if train_var:
                    # self.mvar = self.update_mvar(x, y)
                    self.pvar = self.update_pvar(x, f1)
                    self.poe_var = self.update_poe(f1, f1)

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

            self.trace_a.append(avec)
            self.trace_b.append(bvec)
            self.trace_x.append(xxvec)
            self.trace_f1.append(f1vec)
            self.trace_f2.append(f2vec)
            self.trace_beta.append(betavec)
            # import pdb; pdb.set_trace()
            print('Observation ' + str(i) + ' Complete at ' + str(s) + 'steps')
