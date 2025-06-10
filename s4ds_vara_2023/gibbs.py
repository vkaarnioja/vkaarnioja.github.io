# A simple Gibbs sampler algorithm
import numpy as np
import matplotlib.pyplot as plt

p = lambda x,y: np.exp(-10*(x**2-y)**2-(y-1/4)**4)
x = np.array([0,0])
nsamples = 5000
samples = []
xx = np.linspace(-2,2,10000)
for jj in range(1,nsamples):
    u = np.random.uniform()
    P = np.cumsum(p(xx,x[1]))
    P = P/P[-1]
    ind = np.where(u<=P)[0][0]
    x1 = xx[ind]
    u = np.random.uniform()
    P = np.cumsum(p(x1,xx))
    P = P/P[-1]
    ind = np.where(u<=P)[0][0]
    x2 = xx[ind]
    x = np.array([x1,x2])
    samples.append(x)
# At this point, one would usually discard some number of
# the initial samples (the burn-in period). Here, we omit
# this step.
samples = list(zip(*samples)) # transpose list
X,Y = np.meshgrid(np.linspace(-2,2),np.linspace(-2,2))
plt.contour(X,Y,p(X,Y))
plt.plot(samples[0],samples[1],'.',color='black',markersize=2)
plt.title('Single component Gibbs sampler with ' + str(nsamples) + ' samples')
plt.show()

fig,ax = plt.subplots(2,1)
ax[0].plot(samples[0],linewidth=.5)
ax[0].set_xlabel('Sample history of $x_1$')
ax[0].xaxis.set_ticks_position('top')
ax[0].set_ylabel('$x_1$')
ax[1].plot(samples[1],linewidth=.5)
ax[1].set_xlabel('Sample history of $x_2$')
ax[1].set_ylabel('$x_2$')
plt.show()

# Function to compute the autocovariance
def autocovariance(f):
    N = len(f)
    gamma2 = np.zeros(N-1)
    f_c = f-np.mean(f)
    gamma2_0 = np.mean(f_c**2)
    for k in np.arange(N-1,0,-1):
        jj = np.arange(1,N-k+1)
        gamma2[k-1] = 1/(gamma2_0*(N-k)) * np.sum(f_c[jj-1]*f_c[jj+k-1])
    return gamma2

# Note: autocovariances are computed after burn-in is removed!
ac1 = autocovariance(samples[0])
ac2 = autocovariance(samples[1])
N_ac = 100
plt.plot(range(0,N_ac+1),ac1[0:N_ac+1],'o',markerfacecolor='none',markeredgecolor='red',label='horizontal component')
plt.plot(range(0,N_ac+1),ac2[0:N_ac+1],'o',markerfacecolor='none',markeredgecolor='blue',label='vertical component')
plt.legend()
plt.show()