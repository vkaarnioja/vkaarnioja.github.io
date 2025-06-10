from fem import *
from scipy.sparse.linalg import spsolve
from joblib import Parallel, delayed

if __name__ == '__main__':
    nodes,element,interior,centers = generateFEmesh(3) # generate FE mesh
    ncoord = len(nodes) # number of coordinates (boundary + interior)
    nelem = len(centers) # number of elements
    ndof = len(interior) # number of degrees of freedom
    grad,mass = generateFEmatrices(nodes,element) # generate FE matrices
    
    # Set up the diffusion coefficient
    s = 100 # stochastic dimension
    decay = 2.0 # decay of the input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
    a = lambda y: 2 + y @ deterministic
    
    # Set up the loading
    f = lambda x: x[:,0]
    rhs = mass[interior,:] @ f(nodes)

    gen = np.loadtxt('https://vesak90.userpage.fu-berlin.de/offtheshelf2048.txt')
    ones = np.ones(ndof)
    def solve(i,shift,n):
        qmcnode = np.mod(i*z/n+shift,1)-1/2
        A = UpdateStiffness(grad,a(qmcnode))
        sol = spsolve(A[np.ix_(interior,interior)],rhs)
        return ones @ mass[np.ix_(interior,interior)] @ sol
    
    z = gen[:s]
    R = 4
    rms = []
    with Parallel(n_jobs=-1) as parallel:
        for n in 2**np.arange(10,16):
            print('n = ' + str(n))
            results = []
            for r in range(R):
                shift = np.random.uniform(0,1,s)
                tmp = parallel(delayed(solve)(i,shift,n) for i in range(n))
                results.append(np.mean(tmp))
            qmcavg = np.mean(results)
            rmserror = np.linalg.norm(qmcavg-results)/np.sqrt(R*(R-1))
            rms.append(rmserror)

    # Least squares fit
    A = np.ones((6,2))
    nadjusted = R * 2**np.arange(10,16)
    A[0:6,1] = np.log(nadjusted)
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(rms))
    lsq[0] = np.exp(lsq[0])
    
    # Visualize the results
    fig, ax = plt.subplots(1,1,figsize=[7,7])
    ax.loglog(nadjusted,lsq[0]*nadjusted**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(nadjusted,rms,'.r','QMC error')
    ax.set_title('QMC error (s = ' + str(s) + ')',fontsize=15)
    ax.set_xlabel('n',fontsize=13)
    ax.set_ylabel('error',fontsize=13)
    ax.legend()
    plt.show()
