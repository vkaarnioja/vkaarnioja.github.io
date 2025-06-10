# Let us try using the TSVD method to solve an X-ray tomography problem.
# We use the FIPS open data set https://zenodo.org/record/1254210
import numpy as np
import scipy.io
from scipy import sparse
import matplotlib.pyplot as plt

data = scipy.io.loadmat('DataFull_128x15.mat') # sparse angle tomography data
#data = scipy.io.loadmat('DataLimited_128x15.mat') # limited angle tomography data
A = data['A'] # extract projection matrix
m = np.transpose(data['m']) # extract sinogram data
# N.B. MATLAB transpose and Numpy transpose are different! We need to first transpose
# the data m so that the reshape operation is compliant with the indexing of elements
# in the projection matrix A (which has been constructed in MATLAB).

# Solving the linear system _without_ using regularization.
res_naive = sparse.linalg.lsqr(A,m.reshape((m.size,1)))[0]
plt.imshow(res_naive.reshape((128,128)).T,cmap='gray')
plt.show()
# The problem is very ill-conditioned, so the the presence of small
# measurement noise ruins the reconstruction.

# We form the truncated SVD of the system matrix corresponding to
# k singular values and solve the spectrally truncated equation.
for k in [1,10,100,1000]:
    u,s,vT = sparse.linalg.svds(A,k,which='LM')
    Sinv = sparse.diags(1/s)
    res = vT.T @ (Sinv @ (u.T @ m.reshape((m.size,1))))
    plt.imshow(res.reshape((128,128)).T,cmap='gray')
    plt.show()
# The reconstruction corresponding to 1000 largest singular values looks
# reasonable. While the TSVD method is simple to implement, the implementation 
# can be extremely slow if the (sparse) system matrix is very large in size.