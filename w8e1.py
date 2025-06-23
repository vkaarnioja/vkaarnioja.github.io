from fem import *
from scipy import interpolate
from scipy.sparse.linalg import spsolve

if __name__ == '__main__':
    fun = lambda x: x[:,0]
    # Find FE solution on a dense mesh
    nodes,element,interior,centers = generateFEmesh() # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers) # number of elements
    ndof = len(interior) # number of degrees of freedom
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices
    A = UpdateStiffness(grad,np.ones(nelem))
    rhs = mass[interior,:] @ fun(nodes)
    sol_dense = np.zeros(ncoord)
    tmp = spsolve(A[np.ix_(interior,interior)],rhs)
    sol_dense[interior] = tmp
    nodes_dense = nodes
    mass_dense = mass
    errors = []
    for i in range(1,5):
        nodes,element,interior,centers = generateFEmesh(i) # generate FE mesh
        ncoord = len(nodes) # number of coordinates (boundary + interior)
        nelem = len(centers) # number of elements
        ndof = len(interior) # number of degrees of freedom
        grad,mass = generateFEmatrices(nodes,element) # generate FE matrices
        # Find FE solution on a coarse mesh
        A = UpdateStiffness(grad,np.ones(nelem))
        rhs = mass[interior,:] @ fun(nodes)
        sol = np.zeros(ncoord)
        tmp = spsolve(A[np.ix_(interior,interior)],rhs)
        sol[interior] = tmp
        
        # Intepolate onto dense FE mesh
        sol_interp = interpolate.griddata((nodes[:,0],nodes[:,1]),sol,(nodes_dense[:,0],nodes_dense[:,1]))
        errors.append(np.sqrt((sol_dense-sol_interp).T @ mass_dense @ (sol_dense-sol_interp)))
    
    # Least squares fit
    h = 2.0**(-np.arange(1,5))
    A = np.ones((4,2))
    A[0:5,1] = np.log(h)
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])
    
    # Visualize the results
    fig, ax = plt.subplots(1,1,figsize=[7,7])
    ax.loglog(h,lsq[0]*h**lsq[1],'--b',label='slope: ' + str(lsq[1]))
    ax.loglog(h,errors,'.r',label='FE errors')
    ax.set_title('Finite element error',fontsize=15)
    ax.set_xlabel('h',fontsize=13)
    ax.set_ylabel('$L^2$ error',fontsize=13)
    ax.legend()
    plt.show()

