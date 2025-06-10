import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

def generateFEmesh(level=5):
    # Create a regular uniform triangulation of the unit square (0,1)**2
    n1 = 2**level+1 # number of nodes in 1D
    # Topology: FE nodes, mesh elements, interior, centers
    X,Y = np.meshgrid(np.arange(0,n1)/(n1-1),np.arange(0,n1)/(n1-1))
    nodes = np.array([X.flatten(),Y.flatten()]).T
    element = []; interior = []
    for i in range(0,n1-1):
        for j in range(0,n1-1):
            element.append([j*n1+i,(j+1)*n1+i,j*n1+i+1])
            element.append([(j+1)*n1+i,(j+1)*n1+i+1,j*n1+i+1])
            if i < n1-2 and j < n1-2:
                interior.append((j+1)*n1+i+1)
    centers = np.mean(nodes[element[:]],axis=1)
    # Visualize the mesh
    #plt.triplot(nodes[:,0],nodes[:,1],triangles=element)
    #plt.show()
    # Sanity check:
    #for i in range(len(interior)):
    #   vals = [0]*len(nodes)
    #   vals[interior[i]] = 1
    #   fig = plt.figure(figsize=plt.figaspect(1.0))
    #   ax = fig.add_subplot(1,1,1,projection='3d')
    #   ax.plot_trisurf(nodes[:,0],nodes[:,1],vals,triangles=element,cmap=plt.cm.rainbow)
    #   plt.show()
    return nodes,element,interior,centers

def shoelace(g):
    # Compute the area of a triangle with
    # vertices g[0], g[1], and g[2]
    return abs(np.linalg.det([g[0],g[1]])
               + np.linalg.det([g[1],g[2]])
               + np.linalg.det([g[2],g[0]]))/2

def generateFEmatrices(nodes,element):
    # Assemble FE matrices
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(element) # number of elements
    mass_data = []; mass_rows = []; mass_cols = []
    grad_data = []; grad_rows = []; grad_cols = []
    localmass = np.array([[1/12,1/24,1/24],[1/24,1/12,1/24],[1/24,1/24,1/12]])
    for k in range(nelem):
        ind = element[k]
        g = nodes[ind]
        detB = abs(np.linalg.det([g[1]-g[0],g[2]-g[0]]))
        Dt = np.array([g[2]-g[1],g[0]-g[2],g[1]-g[0]])
        triarea = shoelace(g)
        localgrad = Dt@Dt.T/4/triarea
        for i in range(3):
            for j in range(3):
                # Mass matrix entries
                mass_rows.append(ind[i])
                mass_cols.append(ind[j])
                mass_data.append(detB*localmass[i,j])
                # Flattened stiffness tensor entries
                grad_rows.append(ind[i]*ncoord+ind[j])
                grad_cols.append(k)
                grad_data.append(localgrad[i,j])
    mass = sparse.csr_matrix((mass_data,(mass_rows,mass_cols)),shape=(ncoord,ncoord))
    grad = sparse.csr_matrix((grad_data,(grad_rows,grad_cols)),shape=(ncoord*ncoord,nelem))
    return grad,mass

def UpdateStiffness(grad,a):
    n = np.sqrt(grad.shape[0]).astype(int)
    vec = grad @ sparse.csr_matrix(a.reshape((a.size,1)))
    return sparse.csr_matrix.reshape(vec,(n,n)).tocsr()

if __name__ == '__main__':
    level = 5 # discretization level, corresponds to mesh width h = 1/2**level
    nodes,element,interior,centers = generateFEmesh(level) # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers) # number of elements
    ndof = len(interior) # number of degrees of freedom
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices

    # PDE example:
    # -div(a(x)grad(u(x))) = f(x) in (0,1)**2
    # with homogeneous zero Dirichlet BCs,
    # source term f(x) = x_1+x_2,
    # and diffusion coefficient a(x) = 1+x_1**2+x_2**2
    
    a = lambda x: 1+np.sum(x**2,axis=1) # diffusion coefficient
    f = lambda x: np.sum(x,axis=1) # source term
    rhs = mass[interior,:]@f(nodes) # precompute the loading vector
    aval = a(centers) # evaluate diffusion coefficient at element centers
    stiffness = UpdateStiffness(grad,aval) # assemble stiffness matrix corresponding to the diffusion coefficient
    sol = np.zeros(ncoord) # initialize solution vector
    sol[interior] = sparse.linalg.spsolve(stiffness[np.ix_(interior,interior)],rhs) # solve the PDE

    # Visualize the solution
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1],sol,triangles=element,cmap=plt.cm.rainbow)
    plt.show()