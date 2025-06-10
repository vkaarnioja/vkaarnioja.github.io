from fem import *
from joblib import Parallel, delayed

if __name__ == '__main__':
    # FEM set-up
    nodes,element,interior,centers = generateFEmesh()
    ncoord = len(nodes)
    grad,mass = generateFEmatrices(nodes,element)

    # Set up the diffusion coefficient
    s = 100 # stochastic dimension
    decay = 2.0 # decay of input random field
    indices = np.arange(1,s+1).reshape((s,1))
    deterministic = indices**(-decay) * np.sin(np.pi * indices * centers[:,0]) * np.sin(np.pi * indices * centers[:,1])
    a = lambda y: 2 + y @ deterministic

    # Set up the loading
    f = lambda x: x[:,0] # source term
    rhs = mass[interior,:] @ f(nodes) # loading vector

    # Solve PDE for many realizations of a random vector
    def solve(i):
        mcnode = np.random.uniform(-1/2,1/2,s)
        A = UpdateStiffness(grad,a(mcnode))
        sol = np.zeros(ncoord)
        tmp = sparse.linalg.spsolve(A[np.ix_(interior,interior)],rhs)
        sol[interior] = tmp
        return sol

    sums = []
    means = []
    maxiter = 15
    # Open a pool of parallel workers
    with Parallel(n_jobs=-1) as parallel:
        for i in range(0,maxiter+1):
            print('n = ' + str(2**i))
            nincr = 1 if i == 0 else 2**(i-1)
            n = 2**i
            block = 10000 # solve up to 10000 tasks in parallel before combining the results
            iter = 0
            tempsum = np.zeros(ncoord)
            while iter*block < nincr:
                # Parallelize the PDE solves. We only compute _at most_ 10000 parallel tasks at the same time before combining the results to ensure that we do not use too much memory. Note that the "min" criterion ensures that we end up computing the correct number of PDE solutions.
                results = parallel(delayed(solve)(j) for j in range(iter*block,min((iter+1)*block,nincr)))
                tempsum += sum(results) # Combine the results
                iter +=1 # Continue loop until the correct number of PDE solves have been computed
            sums.append(tempsum) # Keep a running total of the sum
            means.append(sum(sums)/n) # Compute the Monte Carlo estimate

    # Use the result corresponding to n = 2**maxiter as the reference solution
    ref = means[maxiter]
    # Compute the approximate L2 errors
    errors = []
    for i in range(0,maxiter):
        errors.append(np.sqrt((means[i]-ref).T @ mass @ (means[i]-ref)))

    # Least squares fit
    A = np.ones((maxiter,2))
    A[0:maxiter,1] = np.log(2**np.arange(0,maxiter))
    lsq = np.linalg.solve(A.T @ A, A.T @ np.log(errors))
    lsq[0] = np.exp(lsq[0])

    # Visualize the results
    fig, ax = plt.subplots(1,1,figsize=[7,7])
    x = 2**np.arange(0,maxiter)
    ax.loglog(x,lsq[0]*x**lsq[1],'--b',linewidth=2,label='slope: ' + str(lsq[1]))
    ax.loglog(x,errors,'.r',label='data')
    ax.set_title('Source problem / Monte Carlo (s = ' + str(s) + ')', fontsize=15)
    ax.set_xlabel('n',fontsize=13)
    ax.set_ylabel('$L^2$ error',fontsize=13)
    ax.legend()
    plt.show()
