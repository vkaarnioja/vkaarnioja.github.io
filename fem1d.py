import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def generateFEmesh(n1):
    # Create a regular partition of the unit interval (0,1)
    # Topology: FE nodes, mesh elements, interior, centers
    nodes = np.arange(0,n1)/(n1-1)
    element = np.array([np.arange(0,n1-1),np.arange(1,n1)]).T
    interior = np.arange(1,n1-1)
    centers = (nodes[:-1]+nodes[1:])/2
    return nodes,element,interior,centers

def generateFEmatrices(nodes,element):
    # FE matrices
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(element) # number of elements
    h = 1/(ncoord-1) # element width
    mass = sparse.lil_matrix((ncoord,ncoord))
    grad = sparse.lil_matrix((ncoord*ncoord,nelem))
    # Note: it would be more efficient to construct
    # the stiffness matrix directly in CSC or CSR format
    localmass = h*np.array([[1/3,1/6],[1/6,1/3]])
    localgrad = (1/h)*np.array([[1,-1],[-1,1]])
    for k in range(nelem):
        ind = element[k]
        g = nodes[ind]
        mass[np.ix_(ind,ind)] = mass[np.ix_(ind,ind)] + localmass
        dummy = sparse.lil_matrix((ncoord,ncoord)); dummy[np.ix_(ind,ind)] = localgrad
        grad[:,k] = grad[:,k] + dummy.reshape((ncoord*ncoord,1))
    grad = grad.tocsr()
    return grad,mass

def UpdateStiffness(grad,a):
    n = np.sqrt(grad.shape[0]).astype(int)
    vec = grad @ sparse.csr_matrix(a.reshape((a.size,1)))
    return sparse.csr_matrix.reshape(vec,(n,n)).tocsr()

if __name__ == '__main__':
    n = 101 # discretization level, corresponds to mesh width h = 1/(n-1)
    nodes,element,interior,centers = generateFEmesh(n) # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers)
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices

    # ODE example:
    # -(a(x)u'(x))' = f(x) in (0,1)
    # with Dirichlet BCs u(0)=u(1)=0
    
    # Set up the uncertain diffusion coefficient
    s = 100 # stochastic dimension
    decay = 2.0 # decay of the input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers) # precompute deterministic part
    a = lambda y: np.exp(y @ deterministic) # a(x,y) = exp(\sum_{j=1}^s y_j j^{-decay} sin(pi*j*x))

    f = lambda x: x-1/2 # source term
    rhs = mass[interior,:]@f(nodes) # precompute loading vector

    # Main loop (can be parallelized, e.g., using the joblib library)
    y = np.random.uniform(size=s) # realization of the random parameter
    aval = a(y) # realization of the input random field
    stiffness = UpdateStiffness(grad,aval) # assemble stiffness matrix corresponding to realization of y
    sol = np.zeros(ncoord) # initialize solution vector
    sol[interior] = sparse.linalg.spsolve(stiffness[np.ix_(interior,interior)],rhs) # solve the PDE

    # Visualize the solution
    fig = plt.subplots()
    plt.plot(nodes,sol)
    plt.show()
