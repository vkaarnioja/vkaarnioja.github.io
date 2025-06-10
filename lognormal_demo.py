# Monte Carlo for PDE with lognormal diffusion coefficient
from fem import *
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

    # Use n == 2**maxiter as the "reference solution"
    maxiter = 16

    # Initialize the source term and loading vector
    f = lambda x: x[:,0] # source term, in this case f(x_1,x_2) = x_1
    rhs = mass[interior,:] @ f(nodes) # loading term; not affected by the random diffusion coefficient

    # Create a function which solves the PDE for a random realization of the input coefficient
    def solve(i):
        mcnode = np.random.normal(size=s) # a realization of the normally distributed parametric vector "y"
        A = UpdateStiffness(grad,a(mcnode)) # assemble the stiffness matrix corresponding to a(y), y == mcnode
        sol = sparse.linalg.spsolve(A[np.ix_(interior,interior)],rhs) # solve the PDE in the interior of the domain
        return sol

    print('MC simulation number 1')
    sums = []
    means = []
    # Range over an increasing number of random samples
    with Parallel(n_jobs=-1) as parallel: # initialize the parallel pool
        for i in range(maxiter+1):
            nincr = 1 if i==0 else 2**(i-1) # reuse the previous 2**(i-1) points
            n = 2**i # total number of points
            tmp = parallel(delayed(solve)(k) for k in range(nincr)) # solve the PDE
            sums.append(np.sum(tmp,axis=0)) # store the partial sums
            means.append(np.sum(sums,axis=0)/n) # store the sample averages

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
    A[0:maxind,1] = np.log(2**np.arange(maxind))
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])

    # Plot the L2 errors as a log-log plot
    fig, ax = plt.subplots()
    x = 2**np.arange(maxiter)
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r',label='errors')
    ax.legend()
    ax.set_title('Lognormal example (Monte Carlo)',fontsize=15)
    ax.set_xlabel(r'$n$',fontsize=13)
    ax.set_ylabel(r'$L^2$ error',fontsize=13)
    plt.show()

    # Estimate the root-mean-square error by performing multiple Monte Carlo
    # simulations and averaging the results
    trials = 10 # let us try 10 Monte Carlo simulations
    errors = [i**2 for i in errors] # reuse the computation from before
    # Range over an increasing number of random samples
    with Parallel(n_jobs=-1) as parallel: # initialize the parallel pool
        for trial in range(trials-1):
            print('MC simulation number ' + str(trial+2))
            sums = []
            means = []
            for i in range(0,maxiter+1):
                nincr = 1 if i==0 else 2**(i-1) # reuse the previous 2**(i-1) points
                n = 2**i # total number of points
                tmp = parallel(delayed(solve)(k) for k in range(nincr)) # solve the PDE
                sums.append(np.sum(tmp,axis=0)) # store the partial sums
                means.append(np.sum(sums,axis=0)/n) # store the sample averages
            ref = means[maxind] # reference solution
            # Compute the L2 errors of solutions vis-a-vis the reference solution
            for i in range(maxind):
                errors[i] = errors[i] + (means[i]-ref).T @ mass[np.ix_(interior,interior)] @ (means[i]-ref)
    errors = [np.sqrt(i/trials) for i in errors]

    # Least squares fit for the errors (see note on the course page)
    A = np.ones((maxind,2))
    A[0:maxind,1] = np.log(2**np.arange(maxind))
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])

    # Plot the L2 errors as a log-log plot
    fig, ax = plt.subplots()
    x = 2**np.arange(maxiter)
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r',label='errors')
    ax.legend()
    ax.set_title('Lognormal example (Monte Carlo)\naveraged over ' + str(trials) + ' trials',fontsize=15)
    ax.set_xlabel(r'$n$',fontsize=13)
    ax.set_ylabel(r'$L^2$ R.M.S. error',fontsize=13)
    plt.show()