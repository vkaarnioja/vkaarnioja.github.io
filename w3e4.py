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

    # Solve PDE for many realizations of a random vector
    def solve(i):
        mcnode = np.random.uniform(-1/2,1/2,s)
        stiffness = UpdateStiffness(grad,a(mcnode))
        evals,evecs = sparse.linalg.eigsh(stiffness[np.ix_(interior,interior)],k=1,M=mass[np.ix_(interior,interior)],which='SM')
        coef = np.sqrt(evecs.T @ mass[np.ix_(interior,interior)] @ evecs)
        evecs = np.sign(evecs[0])*evecs/coef
        evecs_full = np.zeros(ncoord)
        evecs_full[interior] = evecs.T
        evals = evals[0]
        return evals,evecs_full

    evalsums = []
    evalmeans = []
    evecsums = []
    evecmeans = []
    maxiter = 15

    # Open a pool of parallel workers
    with Parallel(n_jobs=-1) as parallel:
        for i in range(0,maxiter+1):
            print('n = ' + str(2**i))
            nincr = 1 if i == 0 else 2**(i-1)
            n = 2**i
            block = 10000 # solve up to 10000 tasks in parallel before combining the results
            iter = 0
            evalsum = 0
            evecsum = np.zeros(ncoord)
            while iter*block < nincr:
                # Parallelize the PDE solves. We only compute _at most_ 10000 parallel tasks at the same time before combining the results to ensure that we do not use too much memory. Note that the "min" criterion ensures that we end up computing the correct number of PDE solutions.
                results = parallel(delayed(solve)(j) for j in range(iter*block,min((iter+1)*block,nincr)))
                results = list(zip(*results)) # transpose list
                evalsum += sum(results[0]) # Combine the eigenvalue results
                evecsum += sum(results[1]) # Combine the eigenvector results
                iter +=1 # Continue loop until the correct number of PDE solves have been computed
            evalsums.append(evalsum) # Keep a running total of the eigenvalue sum
            evalmeans.append(sum(evalsums)/n) # Compute the Monte Carlo estimate
            evecsums.append(evecsum) # Keep a running total of the eigenvector sum
            evecmeans.append(sum(evecsums)/n) # Compute the Monte Carlo estimate

    # Use the result corresponding to n = 2**maxiter as the reference solution
    evalref = evalmeans[maxiter]
    evecref = evecmeans[maxiter]
    # Compute the approximate L2 errors
    evalerrors = []
    evecerrors = []
    for i in range(0,maxiter):
        evalerrors.append(np.abs(evalmeans[i]-evalref))
        evecerrors.append(np.sqrt((evecmeans[i]-evecref).T @ mass @ (evecmeans[i]-evecref)))
        
    # Least squares fit
    A = np.ones((maxiter,2))
    A[0:maxiter,1] = np.log(2**np.arange(0,maxiter))
    evallsq = np.linalg.solve(A.T @ A, A.T @ np.log(evalerrors))
    evallsq[0] = np.exp(evallsq[0])
    eveclsq = np.linalg.solve(A.T @ A, A.T @ np.log(evecerrors))
    eveclsq[0] = np.exp(eveclsq[0])

    # Visualize the results
    fig, (ax,ax2) = plt.subplots(2,figsize=[7,7])
    x = 2**np.arange(0,maxiter)
    ax.loglog(x,evallsq[0]*x**evallsq[1],'--b',linewidth=2,label='slope: ' + str(evallsq[1]))
    ax.loglog(x,evalerrors,'.r',label='data')
    ax.set_title('Eigenvalues / Monte Carlo (s = ' + str(s) + ')', fontsize=15)
    #ax.set_xlabel('n',fontsize=13)
    ax.set_ylabel('absolute error',fontsize=13)
    ax.legend()
    
    ax2.loglog(x,eveclsq[0]*x**eveclsq[1],'--b',linewidth=2,label='slope: ' + str(eveclsq[1]))
    ax2.loglog(x,evecerrors,'.r',label='data')
    ax2.set_title('Eigenvectors / Monte Carlo (s = ' + str(s) + ')', fontsize=15)
    ax2.set_xlabel('n',fontsize=13)
    ax2.set_ylabel('$L^2$ error',fontsize=13)
    ax2.legend()
    plt.show()

