import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import special_ortho_group

def sample_uniform_phases(N, D=1):
    '''
    Returns (N,D) array of randomly generated phases
    in [-pi,pi]

    Parameters
    ----------
    N : int
        Dimension of ssp
    D : int
        Dimension of continuous value

    Returns
    A : (N,D) matrix
        Random base phases
    '''
    return np.random.uniform(low=-np.pi, high=np.pi, size=(N,D))

def theta(A, x=1):
    '''
    Returns FHRR representation of x using phases in A

    Parameters
    ----------
    A : (N,k) array | (N,) vector
        Base phases
    x : (k,) vector or scalar
        Continuous value to encode

    Returns
    -------
    U : (N,) vector
        SSP encoding x (unit-modulus complex)
    '''
    if type(x) in (np.ndarray, list):
        U = np.exp(1.j*A@x)
    else:
        U = np.exp(1.j*A*x)
    return U

def wrap_phases(A, x=1):
    '''
    Returns the phases from encoding x with phases A
    wrapped in range [-pi,pi]

    Parameters
    ----------
    A : (N,k) array | (N,) vector
        Base phases
    x : (k,) vector or scalar
        Continuous value encoded

    Returns
    -------
    U : (N,) vector 
        Phases in [-pi,pi]
    '''
    return np.angle(theta(A, x))


def similarity(A, B):
    '''
    Evaluates the similarity between two FHRR vectors
    with dimension N (note that the similarity can be
    complex-valued)

    Parameters
    ----------
    A : (N,) array
        FHRR vector (complex-valued)
    B : (N,) array
        FHRR vector (complex-valued)

    Returns
    -------
    s : complex
        Similarity between A and B
    '''
    return np.dot(A, np.conj(B).T) / len(A)

def create_axes(N, axes, conjugate_symm=True):
    '''
    Create base phases for each axis

    Parameters
    ----------
    N : int
        Dimension of SSP
    axes : int
        Number of axes
      
    conjugate_symm : bool
        Conjugate symmetry of base phases

    Returns
    -------
    A : (N, axes) array
        Base axis phases
    '''
    if conjugate_symm:
        phases = sample_uniform_phases((N-1)//2, axes)
        
        A=np.zeros((N,axes))
        A[1:(N+1)//2] = phases
        A[-1:N//2:-1] = -phases
    else:
        A = sample_uniform_phases(N, axes)

    return A

def extract_ssp(ssp, locs, A, k):
    '''
    If `ssp` is supposed to be the SSP obtained by
    binding the SSPs for `locs` (where each location
    corresponds to a dimension of the base phases A),
    return the SSP for the k-th encoded value

    `ssp` may be corrupted, hence why this process
    is required

    Parameters
    ----------
    ssp : (N,) vector
    locs : (d,) vector
        Encoded continuous values
    A :(N,d) vector
        Base phases
    k : int 
        Index of axis

    Returns
    -------
    projection : (N,) vector
        The vector obtained by unbinding the SSPs
        for all continuous values encoded other than
        the k-th one
    '''
    copy = ssp.copy()
    for a in range(A.shape[1]):
        if a != k:
            copy /= theta(A.T[a], locs[a])
    return copy

def get_random_location(n_axes, loc_bds, location_sample='unif_sphere', num_pts=1):
    '''
    Sample a random (n_axes)-dimensional location

    Parameters
    ----------
    n_axes : int
        Number of dimensions
    loc_bds : float | (d,) vector
        Bound for each dimension
    location_sample : str ('unif_sphere'|'unif_cube')
        Options: 'unif_sphere' (to sample from a uniform sphere with radius
        given by `loc_bds`) or 'unif_cube' (to sample from a uniform cube
        with dimensions given by `loc_bds`)
    '''
    if location_sample=='unif_cube':
        # sample uniformly from -loc_bds to loc_bds
        return np.random.uniform(-loc_bds, loc_bds, size=(num_pts,n_axes)).squeeze()
    elif location_sample=='unif_sphere':
        # sample uniformly from inside of sphere with radius loc_bds
        loc = np.random.uniform(-loc_bds, loc_bds, size=(num_pts,n_axes))
        while np.any(np.linalg.norm(loc, axis=-1) > loc_bds):
            # repeatedly sample until within sphere
            indices = np.linalg.norm(loc, axis=-1) > loc_bds
            loc[indices] = np.random.uniform(-loc_bds, loc_bds, size=(np.sum(indices),n_axes))
        return loc.squeeze(axis=0) if num_pts == 1 else loc
    else:
        raise NotImplementedError()
    
def create_ssp_bundle(A, locs=[], loc_bds=5,
                      location_sample='unif_sphere',
                      n_bundles=3, in_bundle=2, normalize_during=False):
    ''' 
    Return a bundle of n_bundles SSPs, where each SSP is obtained
    by binding in_bundle SSPs

    A location vector is included using A as base phases

    Parameters
    ----------
    A : (N,d) array
       Base phases
    locs : None | float | (d,) vector | (n_bundles,d) array
        Location to encode for some or all of items in bundle
    loc_bds : float | (d,) vector
        Bound for each dimension
    location_sample : str ('unif_sphere'|'unif_cube')
        Options: 'unif_sphere' (to sample from a uniform sphere with radius
        given by `loc_bds`) or 'unif_cube' (to sample from a uniform cube
        with dimensions given by `loc_bds`)
    n_bundles : int
        Number of bundles
    in_bundle : int
        Number of SSPs in each bundle
    normalize_during : bool 
        If true, normalize after adding a vector to the bundle; if false,
        normalize at the end

    Returns
    -------
    M : (N,) vector
        Bundle (unit-modulus complex)
    i_ssps : list of length `n_bundles` of list of length
     `in_bundle` of (N,) vectors
        The individual vectors bundled with the location for
        each item in the bundle
    b_ssps : list of length `n_bundles` of (N,) vectors
        The list of other vectors bundled with the location for
        each bundle item
    locs : list of length `n_bundles` of (d,) vectors
        The list of locations encoded as part of each item in the
        bundle
    '''
    # get dimensions
    ssp_dim = A.shape[0]
    num_axes = A.shape[1]

    # reshape locs
    locs = np.atleast_2d(locs).tolist()

    # save individual ssp's and binded ssp's
    i_ssps = []
    b_ssps = [] 

    # bundle to return
    M = np.zeros(ssp_dim,dtype=np.complex128)

    for i in range(n_bundles):
        vecs_to_bind = [] #list of ssp's to bind
        binded = np.ones(ssp_dim, dtype=np.complex128) # ssp obtained by binding

        # bind random ssps
        for _ in range(in_bundle):
            vec = theta(sample_uniform_phases(ssp_dim)).squeeze()
            vecs_to_bind.append(vec)
            binded = binded*vec
        
        # generate new location if missing
        if i >= len(locs):
            locs.append(get_random_location(num_axes, loc_bds=loc_bds,
                                  location_sample=location_sample))
        
        # bind location ssp
        binded = binded * theta(A, locs[i])

        # add ssp binded to bundle M
        M += binded
        if normalize_during:
            M /= abs(M)
    
        # save binded and individual ssp's
        b_ssps.append(binded)
        i_ssps.append(vecs_to_bind)

    # normalize if not done already
    if not normalize_during:
        M /= abs(M)

    return M, i_ssps, b_ssps, locs


def corrupt_ssp_distribution(A, loc,
                             corruption_dist='gaussian',
                             noise=0.5):
    ''' 
    Returns a corrupted ssp using corruption_method
    and distribution of noise corruption_dist

    The real and complex portions of the ssp are corrupted

    Parameters
    ----------
    A : (N,d) array
        Base phases
    loc : (d,) vector
        Location to encode
    corruption_dist : str ('gaussian'|'ssp')
        Options: 'phases' (corrupt the phases with the
        noise) or 'ssp' (corrupt the real and complex portions
        of the ssp each with noise/2)
    noise : float
        Amount of noise
    
    Output:
      ssp : (N,) vector
        Corrupted SSP
    '''
    # noise function
    locs = np.atleast_2d(loc)

    def get_noise(n):
        if corruption_dist == 'gaussian':
            return n*np.random.normal(size=(A.shape[0],len(locs)))
        elif corruption_dist == 'uniform':
            return np.random.uniform(-n/2, n/2, size=(A.shape[0],len(locs)))
        else:
            raise NotImplementedError()
        
    ssp = theta(A, locs.T)
    ssp += (get_noise(noise) + 1.j*get_noise(noise))
    ssp = ssp / abs(ssp)
    return ssp.T.squeeze(axis=0) if len(locs) == 1 else ssp.T

def get_corrupted_ssp(N, n_axes, loc=None,
                      conjugate_symm=True,
                      location_sample='unif_sphere',
                      loc_bds=5,
                      corruption_method='bundle',
                      corruption_dist='gaussian',
                      noise=2,
                      **kwargs):
    ''' 
    Create a corrupted SSP using corruption_method

    Parameters
    ----------
    N : int
        Dimension of ssp
    n_axes : int
        Dimension of continuous value    
    loc : None | (n_axes,) vector
        Continuous values to encode in a SSP (if None, generate
        random locations)
    conjugate_symm : bool
        Conjugate symmetry
    loc_bds : float | (d,) vector
        Bound for each dimension
    location_sample : str ('unif_sphere'|'unif_cube')
        Options: 'unif_sphere' (to sample from a uniform sphere with radius
        given by `loc_bds`) or 'unif_cube' (to sample from a uniform cube
        with dimensions given by `loc_bds`)
    corruption_method : str ('bundle'|'noise')
        Options: 'bundle' (corrupt by bundling) or 'noise' (corrupt using
        noise)
    corruption_dist : str ('gaussian'|'ssp')
        Options: 'phases' (corrupt the phases with the
        noise) or 'ssp' (corrupt the real and complex portions
        of the ssp each with noise/2)
    noise : float
        Amount of noise

    Returns
    -------
    A : (N,n_axes) matrix
        Base phases
    corrupted_ssp : (N,) vector 
        Corrupted FHRR vector encoding loc
    clean_ssp : (N,) vector
        Clean FHRR vector encoding loc
    loc : (axes,) vector
        Encoding value
    '''
    # create axes
    A = create_axes(N, n_axes, conjugate_symm=conjugate_symm)
    
    # use or sample location
    if loc is not None:
        loc = np.atleast_1d(loc)
    else:
        loc = get_random_location(n_axes, loc_bds=loc_bds,
                                  location_sample=location_sample)
        
    if corruption_method=='bundle':
        # corrupt using bundles
        M, indiv_ssps, binded_ssps, _ = create_ssp_bundle(A, loc, loc_bds=loc_bds,
                                                          location_sample=location_sample,
                                                          n_bundles=int(noise),
                                                          **kwargs)
        
        # extract using bundles
        corrupted_ssp = M.copy()
        clean_ssp = binded_ssps[0].copy()
        for f in indiv_ssps[0]:
            corrupted_ssp /= f
            clean_ssp /= f
    elif corruption_method=='noise':
        # corrupt using distribution
        corrupted_ssp = corrupt_ssp_distribution(A, loc,
                                                 corruption_dist=corruption_dist,
                                                 noise=noise)
        
        # encode clean ssp
        clean_ssp = theta(A, loc)
    else:
        raise NotImplementedError()

    return A, corrupted_ssp, clean_ssp, loc

def generate_ssp_phases_plot(A, ax, corrupted_ssp, true_locs,
                    est_locs=None,
                    cleaned_ssp =None,
                    use_true=False,
                    plot_true=True,
                    show_legend=True):
    '''
    Generate plot of phases for each axis

    Parameters:
    A : (N,d) vector
        Base phases
    ax : (d,) array of matplotlib.axes.Axes
        Matplotlib axes for plotting
    corrupted_ssp  : (N,) vector
        Corrupted SSP
    true_locs :     (d,) vector 
        True locations
    est_locs : (d,) vector
        Estimated locations
    cleaned_ssp : (N,) vector
        Clean SSP
    use_true : bool
        Use true location to compute phases of
        cleaned SSP
    plot_true : bool
        Whether to plot the true phases
    show_legend : bool
        Only show a legend if true
    '''
    # reshape ax for plotting
    ax = np.atleast_1d(ax).flatten()

    # dimensions
    N = A.shape[0]
    n_axes = A.shape[1]

    # plot the projects of the phases onto each axis
    for a in range(n_axes):

        # get projection of phases onto each axis
        # for different axes
        ax[a].plot(A.T[a], np.angle(extract_ssp(corrupted_ssp, true_locs, A, a)), 'g.', label='Corrupted SSP')

        if plot_true:
            ax[a].plot(A.T[a], wrap_phases(A.T[a], true_locs[a]), 'b.', label='True SSP')
        
        if cleaned_ssp is not None:
            locs = est_locs if not use_true and est_locs is not None else true_locs
            ax[a].plot(A.T[a], np.angle(extract_ssp(cleaned_ssp, locs, A, a)), '.', color='purple', label='Cleaned SSP')

        if est_locs is not None:
            ax[a].plot(A.T[a], wrap_phases(A.T[a], est_locs[a]), 'r.', label='Returned Value')
        
        if show_legend:
            ax[a].legend()
        
        # labels
        ax[a].set_xlabel('Base Phases')
        ax[a].set_ylabel('Measured Phases')

def generate_ssp_similarity_plot(A, ax, ssp, xrange=[-5.,5], xest=None):
    '''
    Plots the similarity curve for the given SSP, using the vector A
    as the base phases.

    Parameters
    ----------
    A : (N,d) vector
        Base phases
    ax  :  matplotlib.axes.Axes
    ssp  : (N,) vector
    xrange : list or tuple of length 2
        Low and high bound for the value for each dimension
    xest : None | (d,) vector
        Estimated locations
    '''
    N, D = A.shape
    P = 200

    if D==1:
        # Create a (1,D) array of x values
        xs = np.linspace(xrange[0], xrange[1], P)[np.newaxis,:]
    elif D==2:
        # Create a (2,D) array of x vectors
        yg, xg = np.mgrid[xrange[0]:xrange[1]:0.1, xrange[0]:xrange[1]:0.1]
        xs = np.array(list(zip(xg.flatten(),yg.flatten()))).T
    else:
        raise NotImplementedError("Not implemented for dimensions > 2")
        
    Ax = A@xs
    ssps = np.exp(-1j*Ax)

    sims = np.real(ssp @ ssps) / N

    if D==2:
        sims = np.real(sims.reshape(xg.shape))
        ax.imshow(sims, extent=[xrange[0],xrange[1],xrange[1],xrange[0]])
        if xest is not None:
            ax.plot(xest[0], xest[1], color=[0.9,0,0,0.5], marker='+', ms=10)
    elif D==1:
        ax.plot(xs[0,:], sims)
        if xest is not None:
            ax.axvline(xest, color='r', ls='--')
