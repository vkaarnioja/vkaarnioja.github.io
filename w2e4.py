from fem import *
from scipy import io
from itertools import product

def Lshape(level=5):
    # Create a regular uniform triangulation of the unit square (0,1)**2
    n1 = 2**(level+1)+1

    # FE nodes
    # First form a Cartesian product of grid points in [0,2]**2
    nodes = list(product(np.linspace(0,2,n1,endpoint=True),repeat=2))
    # Use list comprehension to pick out the elements x in nodes for which min(x)<=1
    nodes = np.array([x for x in nodes if min(x)<=1])

    # Elements
    element = [];
    for ii in np.arange(1,n1):
        for jj in np.arange(1,(n1-1)/2+2):
            if ii <= (n1-1)/2 or jj <= (n1-1)/2:
                element.append(((jj-1)*n1+ii,jj*n1+ii,(jj-1)*n1+ii+1))
                element.append((jj*n1+ii,jj*n1+ii+1,(jj-1)*n1+ii+1))
            if ii <= (n1-1)/2 and jj <= (n1-3)/2:
                element.append((jj-1)*(n1+1)/2+(n1*(n1+1)/2+ii,n1*(n1+1)/2+ii+(n1-1)/2+1,n1*(n1+1)/2+ii+1))
                element.append((jj-1)*(n1+1)/2+(n1*(n1+1)/2+ii+1,(n1+1)*(n1+1)/2+ii,(n1+1)*(n1+1)/2+ii+1))

    # Convert to np.array and fix indexing (Python indexing starts from 0)
    element = -1 + np.array(element,dtype=int)
    interior = [i for i, (x1, x2) in enumerate(nodes) if (0 < x1 < 1 and 0 < x2 < 2) or (1 <= x1 < 2 and 0 < x2 < 1)]
    centers = np.mean(nodes[element[:]],axis=1)
    return nodes,element,interior,centers

nodes,element,interior,centers = Lshape()
#data = io.loadmat('femdata.mat')
#nodes = data['nodes']; element = data['element']; interior = data['interior'][0]; centers = data['centers']

# Visualize the mesh
plt.triplot(nodes[:,0],nodes[:,1],triangles=element)
plt.show()
ncoord = len(nodes)
ndof = len(interior)
nelem = len(centers)
grad,mass = generateFEmatrices(nodes,element)
stiffness = UpdateStiffness(grad,np.ones(nelem)) # stiffness matrix of the Dirichlet-Laplacian

# Solve the eigenvalue problem.
evals,evecs = sparse.linalg.eigsh(stiffness[np.ix_(interior,interior)],k=1,M=mass[np.ix_(interior,interior)],which='SM')
coef = np.sqrt(np.transpose(evecs) @ mass[np.ix_(interior,interior)] @ evecs) # compute the L2 norm
evecs = np.sign(evecs[interior[1]])*evecs/coef; # normalize the eigenfunction
sol = np.zeros(ncoord) # initialize the solution vector
sol[interior] = evecs.reshape((ndof,)) # plug in the eigenfunction values solved in the interior of the domain

# Plot the results over the actual FE mesh (note the use of "element" in trisurf!)
fig=plt.figure(figsize=plt.figaspect(1.0))
ax=fig.add_subplot(1,1,1,projection='3d')
ax.plot_trisurf(nodes[:,0],nodes[:,1],sol,triangles=element,cmap=plt.cm.rainbow)
plt.show()