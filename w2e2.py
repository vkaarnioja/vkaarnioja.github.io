from fem import *

if __name__ == '__main__':
    nodes,element,interior,centers = generateFEmesh() # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers) # number of elements
    ndof = len(interior) # number of degrees of freedom
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices
    stiffness = UpdateStiffness(grad,np.ones(nelem)) # discrete Dirichlet-Laplacian
    f = lambda x: np.sum(x,axis=1) # source term
    g = lambda x: 1-x[:,0]**3-2*x[:,1] # boundary term
    loading = mass[interior,:]@f(nodes)

    # Method 1: nodal interpolation of the boundary values
    boundary = [i for i in range(ncoord) if i not in interior] # indices of boundary nodes
    sol1 = np.zeros(ncoord)
    sol1[boundary] = g(nodes[boundary]) # enforce the boundary condition
    sol1[interior] = sparse.linalg.spsolve(stiffness[np.ix_(interior,interior)],loading-stiffness[np.ix_(interior,boundary)]@sol1[boundary])
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1],sol1,triangles=element,cmap=plt.cm.rainbow)
    plt.show()

    # Method 2: Dirichlet lift
    Delta_g = lambda x: -6*x[:,0] # Laplacian of g
    rhs = loading + mass[interior,:]@Delta_g(nodes)
    sol2 = np.zeros(ncoord)
    sol2[interior] = sparse.linalg.spsolve(stiffness[np.ix_(interior,interior)],rhs)
    sol2 = sol2 + g(nodes)
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1],sol2,triangles=element,cmap=plt.cm.rainbow)
    plt.show()

    # Comparison
    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax = fig.add_subplot(1,1,1,projection='3d')
    ax.plot_trisurf(nodes[:,0],nodes[:,1],sol1-sol2,triangles=element,cmap=plt.cm.rainbow)
    plt.show()