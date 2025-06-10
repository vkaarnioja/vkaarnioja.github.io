# Backward heat equation example from the fifth lecture.
import numpy as np
from scipy import linalg, optimize
import matplotlib.pyplot as plt

# Initialize the parameters.
T = 0.1 # Time parameter.
n = 100 # Number of discretization points.
h = np.pi/n # Step size.
x = np.linspace(h,np.pi-h,n-1) # Discretization grid.
F_true = (x>=1) * (x<=2) * 1.0 # Ground truth.
Nvec = np.arange(1,31) # Use 30 Fourier terms to compute the measurements at time t=T
X,N = np.meshgrid(x,Nvec)
Fcoefs = 2*(np.cos(N)-np.cos(2*N))/N/np.pi
U_true = np.sum(np.multiply(Fcoefs,np.exp(-T*N**2)*np.sin(N*X)),axis=0) # Analytic solution of the PDE at time t=T

# Construct the system matrix.
# First, use the stencil for the second order spatial derivative.
B = np.diag(np.ones(n-2),-1) - 2*np.diag(np.ones(n-1)) + np.diag(np.ones(n-2),1)
B = B/h**2

# Create the forward operator for the temperature distribution at time T.
A = linalg.expm(T*B)

sigma = .01 # Standard deviation of normally distributed noise.
moro2 = (n-1)*sigma**2 # Noise level squared / Morozov discrepancy goal squared.

np.random.seed(321); # Reproducible experiments
# Simulate noisy measurements by adding normally distributed random noise
# with std sigma into the _analytic solution_ to avoid the inverse crime.
U = U_true + sigma * np.random.normal(size=U_true.shape)

# Tikhonov regularization
tikhonov_solution = lambda delta: linalg.solve(A.T @ A + delta * np.eye(n-1),A.T@U)
delta = optimize.fmin(func=lambda delta: np.abs(np.linalg.norm(A@tikhonov_solution(delta**2)-U)**2-moro2),x0=.01,ftol=1e-12,disp=0)[0]
delta = delta**2 # A lazy method to enforce a positivity constraint for the optimization problem...

# Solution with Morozov regularization parameter.
w = tikhonov_solution(delta)

# Check that the Morozov discrepancy goal was achieved.
print('Tikhonov regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@w-U)))
print('Regularization parameter: ' + str(delta))

# Plot the results
fig, ax = plt.subplots()
ax.plot(x,w,'-b',linewidth=2,label='Tikhonov reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('Tikhonov reconstruction with delta\nchosen using Morozov', fontsize=15)
plt.legend(loc = 'upper right')
plt.show()

# Investigate what happens when the regularization parameter is too small.
delta = .000001
w = tikhonov_solution(delta)
print('\nTikhonov regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@w-U)))
print('Regularization parameter: ' + str(delta))
fig, ax = plt.subplots()
ax.plot(x,w,'-b',linewidth=2,label='Tikhonov reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('Tikhonov reconstruction with delta\nchosen too small', fontsize=15)
plt.legend(loc = 'upper right')
plt.show()

# Investigate what happens when the regularization parameter is too large.
delta = 1
w = tikhonov_solution(delta)
print('\nTikhonov regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@w-U)))
print('Regularization parameter: ' + str(delta))
fig, ax = plt.subplots()
ax.plot(x,w,'-b',linewidth=2,label='Tikhonov reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('Tikhonov reconstruction with delta\nchosen too large', fontsize=15)
plt.legend(loc = 'upper right')
plt.show()
