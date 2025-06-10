import numpy as np
import matplotlib.pyplot as plt

# Initializations
d = 1000 # try out different values of d; does the convegence rate w.r.t. n change?
true = 2**d * np.cos(d/2)*np.sin(1/2)**d
fun = lambda x: np.cos(np.sum(x,axis=0))
maxiter = 20
sums = []
means = []

# Compute a naïve Monte Carlo error
print('MC simulation number 1')
for i in np.arange(0,maxiter+1):
	nincr = 1 if i == 0 else 2**(i-1) # reuse the previous 2^(ii-1) points
	ntot = 2**i # total number of points
	sample = np.random.uniform(low=0.0,high=1.0,size=(d,nincr))
	sums.append(np.sum(fun(sample)))
	means.append(sum(sums)/ntot)
errors = abs(means-true)

# Least squares fit (see note on course page)
A = np.ones((maxiter+1,2))
A[0:maxiter+1,1] = np.log(2**np.arange(0,maxiter+1))
lsq = np.linalg.solve(np.transpose(A) @ A,np.transpose(A)@np.log(errors))
lsq[0] = np.exp(lsq[0])

# Visualize the results
fig, ax = plt.subplots()
x = 2**np.arange(0,maxiter+1)
ax.loglog(2**np.arange(0,maxiter+1),lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
ax.loglog(2**np.arange(0,maxiter+1), errors, '.r', label='data')
ax.set_title('Monte Carlo (d = ' + str(d) + ')', fontsize=15)
ax.set_xlabel('n', fontsize=13)
ax.set_ylabel('absolute error', fontsize=13)
ax.legend()
plt.show()

# Estimate the root-mean-square error by performing multiple Monte Carlo
# simulations and averaging the results
trials = 10 # let us try 10 Monte Carlo simulations
errors = errors**2 # reuse the computation from before
for trial in range(trials-1):
	print('MC simulation number ' + str(trial+2))
	sums = []
	means = []
	for i in np.arange(0,maxiter+1):
		nincr = 1 if i == 0 else 2**(i-1) # reuse the previous 2^(ii-1) points
		ntot = 2**i # total number of points
		sample = np.random.uniform(low=0.0,high=1.0,size=(d,nincr))
		sums.append(np.sum(fun(sample)))
		means.append(sum(sums)/ntot)
	errors = errors + (means-true)**2
errors = np.sqrt(errors/trials) # empirical root-mean-square error

# Least squares fit (see note on course page)
A = np.ones((maxiter+1,2))
A[0:maxiter+1,1] = np.log(2**np.arange(0,maxiter+1))
lsq = np.linalg.solve(np.transpose(A) @ A,np.transpose(A)@np.log(errors))
lsq[0] = np.exp(lsq[0])

# Visualize the results
fig, ax = plt.subplots()
x = 2**np.arange(0,maxiter+1)
ax.loglog(2**np.arange(0,maxiter+1),lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
ax.loglog(2**np.arange(0,maxiter+1), errors, '.r', label='data')
ax.set_title('Monte Carlo (d = ' + str(d) + ')', fontsize=15)
ax.set_xlabel('n', fontsize=13)
ax.set_ylabel('R.M.S. error', fontsize=13)
ax.legend()
plt.show()