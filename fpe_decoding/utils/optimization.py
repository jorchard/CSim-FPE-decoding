import numpy as np
import scipy

def couple_by_distribution(A, couplings_adj, inv_cdf, only_diffs=False, max_iters=1000):
    '''
    Returns pairs of indices of SSP to couple together
    that follow the distribution with inverse cdf
    given by inv_cdf

    Parameters
    ----------
    A : (N,d) vector 
        Base phases
    couplings_adj : int
        The number of other coordinates to couple each entry in
        the SSP vector to
    inv_cdf : array_like, output of scipy.stats.rv_continuous.ppf
        Inverse cdf of distribution to follow
    only_diffs : bool 
        Only consider phase differences

    Returns
    -------
    I : list of tuples of length 3
        Indices of phases couple and whether to subtract or add them
    '''
    couplings = []

    ssp_dim = A.shape[0]
    n_axes = A.shape[1]

    for i in range(ssp_dim):
        # differences and sums to base phases at ssp index i
        distances = np.concatenate([A - A[i], A+A[i]], axis=0)
        distances[i] = np.inf # do not include coupling to self
        distances[ssp_dim+i] = np.inf

        added = 0 # add couplings_adj closest which are not already in couplings
        count = 0 # number of iterations
        while np.sum(np.abs(distances) < np.pi) > 0 and added < couplings_adj and count < max_iters:
            # select base phases which are closest in distance to
            # a (n_axes,) sample from the distribution until couplings_adj
            # couplings have been selected or there are no more base
            # phase options to select

            # sample with inverse transform sampling
            u = np.random.uniform(size=(n_axes,))
            dists = inv_cdf(u)

            if (np.abs(dists) > np.pi).any():
                # if not valid base phases in sample
                # repeat
                continue

            count += 1

            # closest to samples
            closest = np.argmin(np.sum(np.abs(distances - dists), axis=1))

            # get phase index and multiplier
            phase_ind = closest
            mult = -1

            # if bigger than ssp dim subtract ssp_dim and
            # set multiplier to 1 (for summation)
            if phase_ind >= ssp_dim:
                if only_diffs:
                    continue
                phase_ind = closest - ssp_dim
                mult = 1


            if (phase_ind, i, mult) not in couplings and (i,phase_ind, mult) not in couplings:
                # add couplings if coupling not included
                couplings.append((i, phase_ind, mult))
                added += 1
                distances[closest] = np.inf # no longer include as option
            elif (phase_ind, i, mult) in couplings:
                # if coupling included no longer include this coupling as option
                distances[closest] = np.inf

    return couplings


def get_coupling_matrix(N, couplings):
    ''' 
    Return coupling matrix where each row corresponds to an entry
    in couplings and each column corresponds to an index in the
    FHRR vector. For each coupling, there is a 1 in the column 
    corresponding to the first index and -1 in the column corresponding
    to the second index

    Parameters
    ----------
    N : int 
        Dimension of SSP
    couplings : list of length c of tuples of length 3
        Each tuple contains the index of phases to couple and
        whether to add or subtract the phases as the third entry

    Returns
    -------
    coupling_mat : (c,N) matrix
        Matrix encoding couplings of phases
    '''
    coupling_mat = np.zeros((len(couplings), N))
    for k,c  in enumerate(couplings):
        coupling_mat[k][c[0]] = 1
        coupling_mat[k][c[1]] = c[2]
    return coupling_mat

def maximize_direct_sim(ssp, A, x_est=None,
                        max_iters=1000,
                        kappa=0.1,
                        epsilon=0.001,
                        alpha=0,
                        scaling=False,
                        verbose=False):
    '''
    Performing gradient descent to yield maximum of direct circular distance
    regression problem (equivalent to real component of similarity between
    ssp and theta(A, x))

        argmax_{x} sum (over indices j from 1 to N) cos(Phi[j] - A[j]@x)

    Parameters
    ----------
    ssp : (N,) vector
        SSP to decode
    A : (N,d) matrix
        axis phase multipliers
    x_est : None | (d,) vector
        Initial x estimate
    max_iters : int
        Maximum number of iterations
    kappa : float
        Step size
    epsilon : float
        Threshold for stopping
    alpha : float
        Momentum parameter
    scaling : bool
        Whether to scale objective by number of items in the sum
    verbose : bool 

    Returns
    -------
    x : (d,) vector
        Decoded value
    iters : int
        Number of iterations
    '''
    ssp_dim = A.shape[0]
    n_axes = A.shape[1]

    Phi = np.angle(ssp) # phases

    # initial estimate of Ax
    if x_est is not None:
        x = np.atleast_1d(x_est.copy())
    else:
        x = np.zeros(n_axes,)
    
    # save x changes for momentum
    x_increment = np.zeros(n_axes,)
    scaling_factor = ssp_dim if scaling else 1

    # loop until converged to reach max_iters
    for iter in range(max_iters):
        err=1/scaling_factor*(np.sin(Phi-A@x)) # derivative
        x_increment = alpha*x_increment + kappa*(A.T@err) # change to x
        x += x_increment # perform update

        if verbose:
            print(f'Iter {iter}: x estimate {x}')

        # stop if changes smaller than epsilon
        if (np.abs(x_increment) < epsilon).all():
            break

    return x, iter

def maximize_phase_coupled_sim(ssp, A, couplings,
                               x_est=None,
                               max_iters=1000,
                               kappa=0.1,
                               epsilon=0.001,
                               alpha=0,
                               verbose=False,
                               scaling=False):
    ''' 
    Performing gradient descent to yield maximum of the phase-coupled least
    circular distance regression formulation (equivalent to real component of
    similarity between higher-dimensional SSP and theta(DeltaA, x))

        argmax_{x} sum (over indices k from 1 to c) cos((CPhi)[k] - (CA)[k]@x)

    Parameters
    ----------
    ssp : (N,) vector
        SSP to decode
    A : (N,d) matrix
        Axis phase multipliers
    couplings : list of length c of tuples of length 3
        :ist of phases to couple and how to couple them
    x_est : None | (d,) vector
        Initial x estimate
    max_iters : int
        Maximum number of iterations
    kappa : float
        Step size
    epsilon : float
        Threshold for stopping
    alpha : float
        Momentum parameter
    scaling : bool
        Whether to scale objective by number of items in the sum
    verbose : bool 

    Returns
    -------
    x : (d,) vector
        Decoded value
    iters : int
        Number of iterations
    '''
    ssp_dim = A.shape[0]
    n_axes = A.shape[1]

    Phi = np.angle(ssp) # phases
    CouplingMat = get_coupling_matrix(ssp_dim,couplings) # get coupling matrix
    scaling_factor = len(couplings) if scaling else 1 # scaling factor

    # initial estimate of Ax
    if x_est is not None:
        x = np.atleast_1d(x_est.copy())
    else:
        x = np.zeros(n_axes,)
    
    # save x changes for momentum
    x_increment = np.zeros(n_axes,)
    
    # loop until converged to reach max_iters
    for iter in range(max_iters):
        err=1/scaling_factor*(np.sin(CouplingMat@Phi-CouplingMat@A@x)) # derivative
        x_increment = alpha*x_increment + kappa*(A.T@CouplingMat.T@err) # change to x
        x += x_increment # perform update

        if verbose:
            print(f'Iter {iter}: x estimate {x}')

        # stop if changes smaller than epsilon
        if (np.abs(x_increment) < epsilon).all():
            break

    return x, iter

def csim_method(ssp, A, couplings,
                x_est=None,     
                max_iters=1000,
                kappas=[0.1,0.1],
                epsilon=0.001,
                alpha=0,
                verbose=False,
                scaling=False):
    ''' 
    Performs gradient descent first on the phase-coupled least
    circular distance regression, then on the direct circular
    distance regression problem

    Parameters
    ----------
    ssp : (N,) vector
        SSP to decode
    A : (N,d) matrix
        Axis phase multipliers
    couplings : list of length c of tuples of length 3
        :ist of phases to couple and how to couple them
    x_est : None | (d,) vector
        Initial x estimate
    max_iters : int
        Maximum number of iterations
    kappa : list or tuple of 2 floats
        Step size for LCD method and then direct method
    epsilon : float
        Threshold for stopping
    alpha : float
        Momentum parameter
    scaling : bool
        Whether to scale objective by number of items in the sum
    verbose : bool 

    Returns
    -------
    x : list of 2 (d,) vectors
        Decoded value of LCD method and direct method
    iters : list of 2 ints
        Number of iterations of LCD method and direct method
    '''
    coupled_est, coupled_iters = maximize_phase_coupled_sim(ssp, A, couplings,
                                        x_est=x_est,
                                        max_iters=max_iters,
                                        kappa=kappas[0],
                                        epsilon=epsilon,
                                        alpha=alpha,
                                        scaling=scaling,
                                        verbose=verbose)
    
    direct_est, direct_iters = maximize_direct_sim(ssp, A,
                                        x_est=coupled_est,
                                        max_iters=max_iters,
                                        kappa=kappas[1],
                                        epsilon=epsilon,
                                        alpha=alpha,
                                        scaling=scaling,
                                        verbose=verbose)
    

    return [coupled_est, direct_est], [coupled_iters, direct_iters]
