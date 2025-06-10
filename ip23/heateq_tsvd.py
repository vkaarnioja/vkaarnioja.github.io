# Backward heat equation example from the fourth lecture.
import numpy as np
from scipy import linalg
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

# Sanity check
plt.plot(x,U_true,'-k',linewidth=2,label='Analytic solution')
plt.plot(x,A@F_true,'--r',linewidth=2,label='FDM solution')
plt.legend()
plt.show()

sigma = .01 # Standard deviation of normally distributed noise.
moro2 = (n-1)*sigma**2 # Noise level squared / Morozov discrepancy goal squared.

np.random.seed(321); # Reproducible experiments
# Simulate noisy measurements by adding normally distributed random noise
# with std sigma into the _analytic solution_ to avoid the inverse crime.
U = U_true + sigma * np.random.normal(size=U_true.shape)

# Singular values of system matrix A
u,s,vT = np.linalg.svd(A)
fig, ax = plt.subplots()
ax.semilogy(s)
ax.set_title('Singular values of the system matrix', fontsize=15)
plt.show()

# Minimum norm solution for noisy data (not regularized)
U_bad = np.linalg.pinv(A)@U
fig, ax = plt.subplots()
ax.plot(x,U_bad,'r')
ax.set_title('Minimum norm solution (not regularized)', fontsize=15)
plt.show()

# TSVD solution with Morozov principle
ind = 5
Sp1 = np.zeros(n-1)
Sp1[0:ind] = 1/s[0:ind]
Sp1 = np.diag(Sp1)
Ap1 = vT.T @ Sp1 @ u.T
res = Ap1 @ U
# Morozov discrepancy principle is approximately satisfied
print('TSVD regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@res-U)))
print('Spectral cut-off: k = ' + str(ind))
fig, ax = plt.subplots()
ax.plot(x,res,'-b',linewidth=2,label='TSVD reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('TSVD reconstruction (k = ' + str(ind) + ' chosen using Morozov)', fontsize=15)
plt.legend()
plt.show()

# TSVD solution with too small spectral cut-off
ind = 2
Sp1 = np.zeros(n-1)
Sp1[0:ind] = 1/s[0:ind]
Sp1 = np.diag(Sp1)
Ap1 = vT.T @ Sp1 @ u.T
res = Ap1 @ U
print('\nTSVD regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@res-U)))
print('Spectral cut-off: k = ' + str(ind))
fig, ax = plt.subplots()
ax.plot(x,res,'-b',linewidth=2,label='TSVD reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('TSVD reconstruction (k = ' + str(ind) + ' too small)', fontsize=15)
plt.legend()
plt.show()

# TSVD solution with too large spectral cut-off
ind = 8
Sp1 = np.zeros(n-1)
Sp1[0:ind] = 1/s[0:ind]
Sp1 = np.diag(Sp1)
Ap1 = vT.T @ Sp1 @ u.T
res = Ap1 @ U
print('\nTSVD regularization')
print('Morozov discrepancy goal: ' + str(np.sqrt(moro2)))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@res-U)))
print('Spectral cut-off: k = ' + str(ind))
fig, ax = plt.subplots()
ax.plot(x,res,'-b',linewidth=2,label='TSVD reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('TSVD reconstruction (k = ' + str(ind) + ' too large)', fontsize=15)
plt.legend()
plt.show()
