# Deconvolution example from the lecture of week 13
import numpy as np
import matplotlib.pyplot as plt

noiselevel = .1

# Simulate measurement data using a dense grid
N1 = 150
s1 = np.linspace(1/N1/2,1-1/N1/2,N1)
t1 = s1
omega = .5
T1,S1 = np.meshgrid(t1,s1)
A = np.exp(-1/(2*omega**2)*(T1-S1)**2)/N1 
x_true1 = 8*t1**3-16*t1**2+8*t1
y_01 = A@x_true1

# Interpolate the data onto a coarser grid and calculate A using the
# coarser grid
N = 120
s = np.linspace(1/N/2,1-1/N/2,N)
t = s
T,S = np.meshgrid(t,s)
A = np.exp(-1/(2*omega**2)*(T-S)**2)/N
x_true = 8*t**3-16*t**2+8*t
y_0 = np.interp(s,s1,y_01)
np.random.seed(100); # reproducible experiments
sigma = noiselevel*np.max(np.abs(y_0)) # std of the noise
y = y_0 + sigma*np.random.normal(size=y_0.shape) # measurement with noise
Gamma_noise = sigma**2*np.eye(N) # covariance of the noise
moro = sigma*np.sqrt(N) #  Expected noise level
eta_0 = np.zeros((N,)); # mean of the noise

## White noise prior
print('Reconstructions with the white noise prior\n')
for gamma in [.2,0.6864,2]:
    # Since we have a linear model with additive Gaussian noise and
    # Gaussian prior, the posterior is also Gaussian with explicitly
    # known mean and covariance. Let us draw some samples from both the
    # prior and posterior distributions using the coloring transform.
    # Moreover, we draw the CM estimate of the posterior distribution
    # and draw the 2-sigma credibility envelopes.
    x_0 = np.zeros((N,)) # prior mean
    Gamma_pr = gamma**2 * np.eye(N) # prior covariance
    G = A@Gamma_pr@A.T+Gamma_noise # Kalman gain
    x_bar = x_0 + Gamma_pr @ A.T @ np.linalg.inv(G) @ (y-A@x_0-eta_0) # posterior mean
    Gamma_post = Gamma_pr - Gamma_pr @ A.T @ np.linalg.inv(G) @ A @ Gamma_pr # posterior covariance
    Gamma_post = (Gamma_post + Gamma_post.T)/2

    # Remark: The "inv" command is used rather liberally in what follows.
    # As a rule of thumb, it is usually better to use matrix decompositions
    # such as LU or Cholesky decomposition or numerical linear system solvers.

    # Draw some samples from the prior
    R_pr = np.linalg.cholesky(np.linalg.inv(Gamma_pr)).T # compute _upper triangular_ Cholesky factor
    samples_pr = np.linalg.solve(R_pr,np.random.normal(size=(N,10))) # coloring transform
    #samples_pr = np.random.multivariate_normal(x_0,Gamma_pr,10).T # sampling a Gaussian density using a generic numpy routine
    fig,ax = plt.subplots()
    ax.plot(t,samples_pr)
    ax.set_title('Samples drawn from the white noise prior, $\gamma = $' + str(gamma))
    plt.show()

    # Draw some samples from the posterior
    R_post = np.linalg.cholesky(np.linalg.inv(Gamma_post)).T # compute _upper triangular_ Cholesky factor
    samples_post = x_bar.reshape((x_bar.size,1)) + np.linalg.solve(R_post,np.random.normal(size=(N,10))) # coloring transform
    #samples_post = np.random.multivariate_normal(x_bar,Gamma_post,10).T # sampling a Gaussian density using a generic numpy routine

    # We can marginalize the posterior w.r.t. each component in order
    # to obtain the 2-sigma credibility envelopes.
    variances = np.diag(Gamma_post)
    fig,ax = plt.subplots()
    ax.plot(t,x_true,'r',t,x_bar,'b',t,x_bar+2*np.sqrt(variances),'b',t,x_bar-2*np.sqrt(variances),'b',linewidth=2)
    ax.plot(t,samples_post,linewidth=1)
    ax.set_title('Samples drawn from the posterior with white noise prior, $\gamma = $' + str(gamma))
    ax.legend(['ground truth','posterior mean $\pm 2\sigma$'],loc='upper right')
    plt.show()

    print('gamma = ' + str(gamma))
    print('Expected noise level: ' + str(moro))
    print('Obtained discrepancy: ' + str(np.linalg.norm(A@x_bar-y)))

## Smoothness prior
L = -np.diag(np.ones(N-1),-1) + 2*np.diag(np.ones(N)) - np.diag(np.ones(N-1),1)
print('\nReconstructions with the smoothness prior\n')
for gamma in [0.001,0.0064,0.02]:
    # Since we have a linear model with additive Gaussian noise and
    # Gaussian prior, the posterior is also Gaussian with explicitly
    # known mean and covariance. Let us draw some samples from both the
    # prior and posterior distributions using the coloring transform.
    # Moreover, we draw the CM estimate of the posterior distribution
    # and draw the 2-sigma credibility envelopes.
    x_0 = np.zeros((N,)); # prior mean
    Gamma_pr = gamma**2 * np.linalg.inv(L.T@L) # prior covariance
    Gamma_pr = (Gamma_pr + Gamma_pr.T)/2
    G = A@Gamma_pr@A.T + Gamma_noise # Kalman gain
    x_bar = x_0 + Gamma_pr @ A.T @ np.linalg.inv(G) @ (y-A@x_0-eta_0) # posterior mean
    Gamma_post = Gamma_pr - Gamma_pr @ A.T @ np.linalg.inv(G) @ A @ Gamma_pr # posterior covariance
    Gamma_post = (Gamma_post + Gamma_post.T)/2

    # Remark: The "inv" command is used rather liberally in what follows.
    # As a rule of thumb, it is usually better to use matrix decompositions
    # such as LU or Cholesky decomposition or numerical linear system solvers.

    # Draw some samples from the prior
    R_pr = np.linalg.cholesky(np.linalg.inv(Gamma_pr)).T # compute _upper triangular_ Cholesky factor
    samples_pr = np.linalg.solve(R_pr,np.random.normal(size=(N,10))) # coloring transform
    #samples_pr = np.random.multivariate_normal(np.zeros(N),Gamma_pr,10).T # sampling a Gaussian density using a generic numpy routine
    fig,ax = plt.subplots()
    ax.plot(t,samples_pr)
    ax.set_title('Samples drawn from the smoothness prior, $\gamma = $' + str(gamma))
    plt.show()

    # Draw some samples from the posterior
    R_post = np.linalg.cholesky(np.linalg.inv(Gamma_post)).T # compute _upper triangular_ Cholesky factor
    samples_post = x_bar.reshape((x_bar.size,1)) + np.linalg.solve(R_post,np.random.normal(size=(N,10))) # coloring transform
    #samples_post = np.random.multivariate_normal(x_bar,Gamma_post,10).T # sampling a Gaussian density using a generic numpy routine

    # We can marginalize the posterior w.r.t. each component in order
    # to obtain the 2-sigma credibility envelopes.
    variances = np.diag(Gamma_post)
    fig,ax = plt.subplots()
    ax.plot(t,x_true,'r',t,x_bar,'b',t,x_bar+2*np.sqrt(variances),'b',t,x_bar-2*np.sqrt(variances),'b',linewidth=2)
    ax.plot(t,samples_post,linewidth=1)
    ax.set_title('Samples drawn from the posterior with white noise prior, $\gamma = $' + str(gamma))
    ax.legend(['ground truth','posterior mean $\pm 2\sigma$'],loc='upper right')
    plt.show()

    print('gamma = ' + str(gamma))
    print('Expected noise level: ' + str(moro))
    print('Obtained discrepancy: ' + str(np.linalg.norm(A@x_bar-y)))