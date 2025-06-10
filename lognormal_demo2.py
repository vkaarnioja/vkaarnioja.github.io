# Quasi-Monte Carlo for PDE with lognormal diffusion coefficient
from fem import *
from scipy.stats import norm
from joblib import Parallel,delayed

if __name__ == '__main__':
    # Precompute the FEM data
    nodes,element,interior,centers = generateFEmesh()
    ncoord = len(nodes)
    ndof = len(interior)
    grad,mass = generateFEmatrices(nodes,element)
    s = 100 # stochastic dimension
    decay = 1.1 # the "decay parameter" \vartheta from the slides
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay)*np.sin(np.pi*indices*centers[:,0])*np.sin(np.pi*indices*centers[:,1])
    a = lambda y: np.exp(y@deterministic)

    # Compute results from n==2**miniter up to n==2**maxiter
    miniter = 10
    maxiter = 17

    # Initialize the source term and loading vector
    f = lambda x: x[:,0] # source term, in this case f(x_1,x_2) = x_1
    rhs = mass[interior,:] @ f(nodes) # loading term; not affected by the random diffusion coefficient

    # Use an off-the-shelf generating vector downloaded from
    # https://web.maths.unsw.edu.au/~fkuo/lattice/index.html
    z = np.loadtxt('https://vesak90.userpage.fu-berlin.de/lattice-39101-1024-1048576.3600',dtype=int)[:s,1]

    # Create a function which solves the PDE for a random realization of the input coefficient
    def solve(i,z,n,shift):
        qmcnode = norm.ppf(np.mod(i*z/n+shift,1)) # a QMC node transported using the inverse CDF
        A = UpdateStiffness(grad,a(qmcnode)) # assemble the stiffness matrix corresponding to a(y), y == qmcnode
        sol = sparse.linalg.spsolve(A[np.ix_(interior,interior)],rhs) # solve the PDE in the interior of the domain
        return sol

    print('QMC simulation number 1')
    sums = []
    means = []
    shift = np.random.uniform(size=s) # random shift
    # Range over an increasing number of QMC points
    with Parallel(n_jobs=-1) as parallel: # initialize the parallel pool
        for i in range(miniter,maxiter+1):
            # Since the generating vector is extensible, we can reuse
            # previous function evaluations
            if i == miniter:
                nincr = 2**i # number of "new" points
                n = nincr
                ind = np.arange(nincr)
            else:
                nincr = 2**(i-1) # number of "new" points
                n = 2**i # total number of points
                ind = np.arange(1, n, 2) # indices of new points
            tmp = parallel(delayed(solve)(ind[k],z,n,shift) for k in range(nincr)) # solve the PDE
            sums.append(np.sum(tmp,axis=0)) # store the partial sums
            means.append(np.sum(sums,axis=0)/n) # store the sample averages
            # If the generating vector is not extensible, then this loop can
            # be replaced with the following naive implementation
            #n = 2**i
            #tmp = parallel(delayed(solve)(k,z,n,shift) for k in range(n))
            #means.append(np.sum(tmp,axis=0)/n)

    # Use the solution corresponding to n == 2**maxiter as the reference solution
    maxind = len(means)-1
    ref = means[maxind] # reference solution
    errors = []
    # Compute the L2 errors of solutions corresponding to n == 2**i, i = 0,...,maxiter-1,
    # vis-a-vis the reference solution. (For the computation of the L2 norm of a FE function
    # using the mass matrix, see the lecture notes!)
    for i in range(maxind):
        errors.append(np.sqrt((means[i]-ref).T @ mass[np.ix_(interior,interior)] @ (means[i]-ref)))

    # Least squares fit for the errors (see note on the course page)
    A = np.ones((maxind,2))
    A[0:maxind,1] = np.log(2**np.arange(miniter,miniter+maxind))
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])

    # Plot the L2 errors as a log-log plot
    fig, ax = plt.subplots()
    x = 2**np.arange(miniter,miniter+maxind)
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r',label='errors')
    ax.legend()
    ax.set_title('Lognormal example (quasi-Monte Carlo)',fontsize=15)
    ax.set_xlabel(r'$n$',fontsize=13)
    ax.set_ylabel(r'$L^2$ error',fontsize=13)
    plt.show()

    # Estimate the root-mean-square error by averaging the results
    # over a number of random shifts
    R = 10 # let us average the results over 10 random shifts
    results = [means] # reuse the computation from before
    with Parallel(n_jobs=-1) as parallel: # initialize the parallel pool
        for r in range(R-1):
            print('QMC simulation number ' + str(r+2))
            shift = np.random.uniform(size=s)
            sums = []
            means = []
            for i in range(miniter,maxiter+1):
                # Since the generating vector is extensible, we can reuse
                # previous function evaluations
                if i == miniter:
                    nincr = 2**i # number of "new" points
                    n = nincr 
                    ind = np.arange(nincr)
                else:
                    nincr = 2**(i-1) # number of "new" points
                    n = 2**i # total number of points
                    ind = np.arange(1, n, 2) # indices of new points
                tmp = parallel(delayed(solve)(ind[k],z,n,shift) for k in range(nincr)) # solve the PDE
                sums.append(np.sum(tmp,axis=0)) # store the partial sums
                means.append(np.sum(sums,axis=0)/n) # store the sample averages
                # If the generating vector is not extensible, then this loop can
                # be replaced with the following naive implementation
                #n = 2**i
                #tmp = parallel(delayed(solve)(k,z,n,shift) for k in range(n))
                #means.append(np.sum(tmp,axis=0)/n)
            results.append(means)

    maxind = maxind+1
    qmcavg = np.mean(results,axis=0) # the QMC estimator averaged over the R random shifts
    errors = []
    # Estimate the R.M.S. error using the formula
    #   sqrt(sum(||\bar{Q}-Q_r||_{L^2}^2,r=1,...,R)/R/(R-1))
    for i in range(maxind):
        tmp = sum((qmcavg[i]-results[r][i]).T @ mass[np.ix_(interior,interior)] @ (qmcavg[i]-results[r][i]) for r in range(R))
        errors.append(np.sqrt(tmp/R/(R-1)))

    # Least squares fit for the errors (see note on the course page)
    A = np.ones((maxind,2))
    A[0:maxind,1] = np.log(2**np.arange(miniter,miniter+maxind))
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])

    # Plot the L2 errors as a log-log plot
    fig, ax = plt.subplots()
    x = 2**np.arange(miniter,miniter+maxind)
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r',label='errors')
    ax.legend()
    ax.set_title('Lognormal example (quasi-Monte Carlo)\naveraged over ' + str(R) + ' random shifts',fontsize=15)
    ax.set_xlabel(r'$n$',fontsize=13)
    ax.set_ylabel(r'$L^2$ R.M.S. error',fontsize=13)
    plt.show()