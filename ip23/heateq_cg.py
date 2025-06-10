# Backward heat equation example from the sixth lecture.
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

sigma = .01 # Standard deviation of normally distributed noise.
epsilon = np.sqrt((n-1))*sigma # Noise level / Morozov discrepancy goal.

np.random.seed(321); # Reproducible experiments
U = U_true + sigma * np.random.normal(size=U_true.shape)

# We solve the normal equation using the conjugate gradient method
sol = np.zeros(n-1)
residual = []
error = []
k = 0
r = A.T@U # RHS of the normal equation
s = r
residual.append(np.linalg.norm(r))
error.append(np.linalg.norm(U))
fig, ax = plt.subplots()

while error[k] > epsilon:
    hlp = A.T @ (A@s) # An auxiliary variable
    alpha = residual[k]**2/(np.dot(s,hlp)) # The line search in the direction s
    sol = sol + alpha*s # New CG iterate
    r = r-alpha*hlp # % Update the residual
    residual.append(np.linalg.norm(r)) # Update the residual error
    beta = residual[k+1]**2/residual[k]**2 # Compute the new search direction
    s = r + beta*s
    k = k+1
    error.append(np.linalg.norm(A@sol-U))
    ax.plot(x,sol,linewidth=2,label='k = ' + str(k))

# Check that the Morozov discrepancy goal was achieved.
print('CG reconstruction for A\'AF = A\'U')
print('Morozov discrepancy goal: ' + str(epsilon))
print('Obtained discrepancy: ' + str(np.linalg.norm(A@sol-U)))

# Plot the results
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('Evolution of CG iterates for $A^TAF = A^TU$', fontsize=15)
plt.legend(loc = 'upper right')
plt.show()

fig, ax = plt.subplots()
ax.plot(x,sol,'-b',linewidth=2,label='CG reconstruction')
ax.plot(x,F_true,'--r',linewidth=2,label='ground truth')
ax.set_title('CG reconstruction for $A^TAF = A^TU$\n(k = ' + str(k) + ' chosen using Morozov)',fontsize=15)
plt.legend(loc = 'upper right')
plt.show()

fig, ax = plt.subplots()
ax.semilogy(np.arange(1,k+1),error[1:])
ax.set_title('Residual $||AF_k-U||$',fontsize=15)
plt.show()