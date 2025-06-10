# Total variation regularization for X-ray tomography
import numpy as np
import scipy.io
from scipy import sparse
import matplotlib.pyplot as plt

# Import projection matrix A and the (noisy) sinogram m
data = scipy.io.loadmat('sino.mat')
A = data['A']
S = data['S']
N = int(data['N']) # dimension of the image
y = np.transpose(S).reshape((S.size,)) # vectorize the sinogram

# Visualize the sinogram data
fig, ax = plt.subplots()
ax.imshow(S,cmap='gray')
ax.axis('auto')
plt.show()

# Construct the discretized (image) gradient operator
block = sparse.spdiags(np.array([np.ones(N),-np.ones(N),np.ones(N)]),np.array([1-N,0,1]),N,N)
LH = sparse.block_diag([block]*N)
LV = sparse.spdiags(np.array([np.ones(N**2),-np.ones(N**2),np.ones(N**2)]),np.array([-N**2+N,0,N]),N**2,N**2)
D = sparse.vstack((LH,LV))

# Choose CP parameters wisely to ensure convergence
L = sparse.linalg.svds(sparse.vstack((A,D)),1,which='LM')[1][0]
tau = 1/L
sigma = 1/L
theta = 1.0
x = np.zeros(A.shape[1])
q = np.zeros(A.shape[0])
z = np.zeros(D.shape[0])
xhat = x

# Use the Chambolle-Pock algorithm to find the TV regularized solution
lam = .01 # educated guess for the regularization parameter
for ii in range(1000):
    # See pg. 30 of the slides
    q = (q+sigma*(A@xhat-y))/(1+sigma)
    z = lam*(z+sigma*D@xhat)/np.maximum(lam,np.abs(z+sigma*D@xhat))
    xold = x
    x = np.maximum(x-tau*A.T@q-tau*D.T@z,0)
    xhat = x+theta*(x-xold)

# Plot the reconstructed x on an NxN grid
plt.imshow(x.reshape((N,N)).T,cmap='gray')
plt.show()

# Compare with a Tikhonov regularized solution
xtik = sparse.linalg.lsqr(A,y,damp=np.sqrt(lam))[0]
plt.imshow(xtik.reshape((N,N)).T,cmap='gray')
plt.show()