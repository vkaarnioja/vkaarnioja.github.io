# A simple tomography matrix demo
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

n = 4
x = -1 + (2/n) * np.arange(0,n+1)
xd = np.kron(np.ones(n),x[0:n])
yd = np.kron(x[0:n],np.ones(n))
xu = np.kron(np.ones(n),x[1:n+1])
yu = np.kron(x[1:n+1],np.ones(n))
# Note that {(xd[k],yd[k]),(xu[k],yd[k]),(xu[k],yu[k]),(xd[k],yu[k])}
# are the corners of the "kth pixel"!

# Set up parallel-beam geometry
M = 180 # number of illumination angles
K = 150 # number of parallel rays per illumination angles
theta = np.pi/M * np.arange(0,M) # equally spaced illumination angles
s = -1+ 2/(K-1) * np.arange(0,K) # equally spaced "offset" for parallel rays
# Remark: instead of -1<=s<=1, we could more generally have -S<=s<=S.

# We want to construct the tomography matrix A in sparse format
I = []
J = []
val = []
eps = np.finfo(float).eps
for m in range(M): # loop over angles
    cc = np.cos(theta[m]) # precompute the cosine
    ss = np.sin(theta[m]) # and sine
    for k in range(K): # loop over rays
        if np.abs(cc) < eps: # horizontal ray
            aux = (yd <= s[k]) * (s[k] < yu) # which pixels encounter the X-ray?
            I.extend((m*K+k)*np.ones(np.sum(aux))) # labels of the X-rays
            J.extend(aux.nonzero()[0]) # labels of "active" pixels
            val.extend(xu[aux.nonzero()]-xd[aux.nonzero()])
        elif np.abs(ss) < eps: # vertical ray
            aux = (xd < s[k]) * (s[k] <= xu)
            I.extend((m*K+k)*np.ones(np.sum(aux)))
            J.extend(aux.nonzero()[0])
            val.extend(yu[aux.nonzero()]-yd[aux.nonzero()])
        elif cc>0: # X-ray with downward slope
            tmp = np.minimum((xu-s[k]*cc)/ss,(s[k]*ss-yd)/cc)-np.maximum((xd-s[k]*cc)/ss,(s[k]*ss-yu)/cc) # cf. pg. 23 of the slides
            aux = tmp>0 # apply ReLU; this is denoted by ()_+ in the slides
            I.extend((m*K+k)*np.ones(np.sum(aux)))
            J.extend(aux.nonzero()[0])
            val.extend(tmp[aux.nonzero()])
        elif cc<0: # X-ray with upward slope
            tmp = np.minimum((xu-s[k]*cc)/ss,(s[k]*ss-yu)/cc)-np.maximum((xd-s[k]*cc)/ss,(s[k]*ss-yd)/cc) # cf. pg. 24 of the slides
            aux = tmp>0 # apply ReLU; this is denoted by ()_+ in the slides
            I.extend((m*K+k)*np.ones(np.sum(aux)))
            J.extend(aux.nonzero()[0])
            val.extend(tmp[aux.nonzero()])

A = sparse.csr_array((val,(I,J)),shape=(M*K,n**2)) # put everything together

# Sanity check
# Let's consider the simple 4x4 phantom on the Wikipedia page
# https://en.wikipedia.org/wiki/Radon_transform

p = np.array([[0,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]])

# Plot the phantom
plt.imshow(p,cmap='gray')
plt.show() 

# Simulate the sinogram data
meas = A @ p.reshape((p.size,1))
sino = meas.reshape((M,K)) # form the sinogram
plt.imshow(sino,cmap='gray')
plt.show()
