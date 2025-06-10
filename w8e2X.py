from fem import *
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed

if __name__ == '__main__':
    nodes,element,interior,centers = generateFEmesh() # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers) # number of elements
    ndof = len(interior) # number of degrees of freedom
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices
    ones = np.ones(ndof)
    slist = 2**np.arange(1,12)
    n = 2**15
    gen = np.loadtxt('https://vesak90.userpage.fu-berlin.de/offtheshelf2048.txt')
    results = []
    for s in slist:
        decay = 2.0
        indices = np.arange(1,s+1).reshape((s,1))
        deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
        a = lambda y: 2 + y @ deterministic
        
        f = lambda x: x[:,0]
        rhs = mass[interior,:] @ f(nodes)

        z = gen[:s]
        def solve(i,n):
            qmcnode = np.mod(i*z/n,1)-1/2
            A = UpdateStiffness(grad,a(qmcnode))
            sol = spsolve(A[np.ix_(interior,interior)],rhs)
            return ones @ mass[np.ix_(interior,interior)] @ sol
        
        print('s = ' + str(s))
        tmp = Parallel(n_jobs=-1)(delayed(solve)(i,n) for i in range(n))
        result = np.mean(tmp)
        results.append(result)

    errors = np.abs(results[:-1]-results[-1])

    # Least squares fit
    A = np.ones((10,2))
    x = slist[:-1]
    A[0:10,1] = np.log(x)
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])
    
    # Visualize the results
    fig, ax = plt.subplots(1,1,figsize=[7,7])
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r','truncation errors')
    ax.set_title('Truncation error',fontsize=15)
    ax.set_xlabel('s',fontsize=13)
    ax.set_ylabel('error',fontsize=13)
    ax.legend()
    plt.show()
