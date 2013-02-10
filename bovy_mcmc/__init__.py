import copy
import numpy
from scipy import stats
import _bovy_mcmc as bovy_mcmc_oned
import _bovy_mv_mcmc as bovy_mcmc_multid
try:
    import markovpy as mpy
    _MARKOVPYENABLED= True
except ImportError:
    _MARKOVPYENABLED= False
def sample_gaussian_proposal(mean,stddev):
    return stats.norm.rvs()*stddev+mean
def eval_ln_gaussian_proposal(new,old,stddev):
    return -0.5*numpy.log(2.*numpy.pi*stddev**2.)-0.5*(old-new)**2./stddev**2.
def markovpy(initial_theta,step,lnpdf,pdf_params,
             isDomainFinite=[False,False],domain=[0.,0.],
             nsamples=1,nwalkers=None,threads=None,
             sliceinit=False,skip=0):
    """
    NAME:
       markovpy
    PURPOSE:
       wrapper around markovpy in the bovy_mcmc style
    INPUT:
       initial_theta - initial sample
       step - stepping out step w, only used to set up initial walkers
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf
       nsamples - number of samples desired
       nwalkers - number of walkers to use
       threads - number of threads to use
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
       sliceinit= if True, initialize by doing slice sampling
       skip= number of samples to skip when initializing using slice sampling
    OUTPUT:
       list of samples, number if nsamples=1
    REVISION HISTORY:
       2011-07-06 - Written - Bovy (NYU)
    DOCTEST:
    >>> import numpy as nu
    >>> nu.random.seed(1)
    >>> def lngaussian(x,mean,var):
    ...     return -.5*nu.log(2.*nu.pi*var)-0.5*(x-mean)**2./var
    >>> pdf_params= (0.,1.)
    >>> isDomainFinite= [False,False]
    >>> domain= [0.,0.]
    >>> nsamples= 200000
    >>> nwalkers= None
    >>> samples= markovpy(0.1,.05,lngaussian,pdf_params,isDomainFinite,domain,nsamples=nsamples,nwalkers=nwalkers,threads=1,sliceinit=True,skip=20)
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -1.
    >>> assert (nu.mean(samples)-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(samples)-1.)**2. < 10.**(logprecision*2.)
    >>> from scipy import stats
    >>> assert (stats.moment(samples,3)-0.)**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples,4)-stats.norm.moment(4))**2. < 10.**(logprecision)
    >>> import scipy as sc
    >>> from scipy import linalg
    >>> def lngaussian(x,params):
    ...        return -sc.log(2.*sc.pi)+.5*sc.log(linalg.det(params[1]))-0.5*sc.dot(x-params[0],sc.dot(params[1],x-params[0]))
    >>> lnpdf= lngaussian
    >>> mean= nu.array([0.,0.])
    >>> variance= nu.array([[1.,1],[1,4.]])
    >>> invvar= linalg.inv(variance)
    >>> pdf_params= ([mean,invvar],)
    >>> isDomainFinite= [False,False]
    >>> domain= [0.,0.]
    >>> nsamples= 200000
    >>> samples= markovpy(nu.array([0.1,0.1]),1.,lnpdf,pdf_params,
    ...        isDomainFinite,domain,
    ...        nsamples=nsamples,sliceinit=True)
    >>> samples= nu.array(samples)
    >>> logprecision= -1.5
    >>> assert (nu.mean(samples[:,0])-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.mean(samples[:,1])-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(samples[:,0])-1.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(samples[:,1])-2.)**2. < 10.**(logprecision*2.)
    >>> from scipy import stats
    >>> assert (stats.moment(samples[:,0],3)-0.)**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples[:,1],3)-0.)**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples[:,0],4)-stats.norm.moment(4))**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples[:,1]/2.,4)-stats.norm.moment(4))**2. < 10.**(logprecision)
    >>> assert (nu.corrcoef(samples.T)[0,1]-0.5)**2. < 10.**(logprecision)
    """
    if not _MARKOVPYENABLED:
        print "'markovy' import failed ..."
        return None
    try:
        ndim = len(initial_theta)
    except TypeError:
        ndim= 1
        if not sliceinit:
            initial_theta= numpy.array([initial_theta])
            isDomainFinite= [isDomainFinite]
            domain= [domain]
            step= [step]
    if not isinstance(isDomainFinite,numpy.ndarray):
        isDomainFinite= numpy.array(isDomainFinite)
    if not isinstance(domain,numpy.ndarray):
        domain= numpy.array(domain)
    if isinstance(step,list): step= numpy.array(step)
    if isinstance(step,(int,float)) or len(step) == 1:
        step= numpy.ones(ndim)*step
    if len(isDomainFinite.shape) == 1 and ndim > 1 and not sliceinit:
        dFinite= []
        for ii in range(ndim):
            dFinite.append(isDomainFinite)
        isDomainFinite= dFinite
    if len(domain.shape) == 1 and ndim > 1 and not sliceinit:
        dDomain= []
        for ii in range(ndim):
            dDomain.append(domain)
        domain= dDomain
    #Set-up walkers
    if nwalkers is None:
        nwalkers = numpy.amax([5,2*ndim])
    if threads is None:
        threads= 1
    nmarkovsamples= int(numpy.ceil(float(nsamples)/nwalkers))
    #Set up initial position
    initial_position= []
    for ww in range(nwalkers):
        if sliceinit:
            thisparams= slice(initial_theta,step,lnpdf,pdf_params,
                              create_method='step_out',
                              isDomainFinite=isDomainFinite,domain=domain,
                              nsamples=1+skip)
            if skip > 0: thisparams= thisparams[-1]
            if ndim == 1:
                thisparams= numpy.array([float(thisparams)])
            initial_theta= copy.copy(thisparams)
        else:
            thisparams= []
            for pp in range(ndim):
                prop= initial_theta[pp]+numpy.random.normal()*step[pp]
                if (isDomainFinite[pp][0] and prop < domain[pp][0]):
                    prop= domain[pp][0]
                elif (isDomainFinite[pp][1] and prop > domain[pp][1]):
                    prop= domain[pp][1]
                thisparams.append(prop)
        initial_position.append(numpy.array(thisparams))
    if ndim == 1: lambdafunc= lambda x: lnpdf(x[0],*pdf_params)
    else: lambdafunc= lambda x: lnpdf(x,*pdf_params)
    #Set up sampler
    sampler = mpy.EnsembleSampler(nwalkers,ndim,
                                  lambdafunc,
                                  threads=threads)
    #Sample
    pos, prob, state= sampler.run_mcmc(initial_position,
                                       numpy.random.mtrand.RandomState().get_state(),
                                       nmarkovsamples)
    #Get chain
    chain= sampler.get_chain()
    samples= []
    for ww in range(nwalkers):
        for ss in range(nmarkovsamples):
            thisparams= []
            for pp in range(ndim):
                thisparams.append(chain[ww,pp,ss])
            samples.append(numpy.array(thisparams))
    if len(samples) > nsamples:
        samples= samples[-nsamples:len(samples)]
    if nsamples == 1:
        return samples[0]
    else:
        return samples

def slice(initial_theta,step,lnpdf,pdf_params,create_method='step_out',
          isDomainFinite=[False,False],domain=[0.,0.],
          nsamples=1,randomize_directions=True,callback=None):
    """
    NAME:
       slice
    PURPOSE:
       simple slice sampling function (e.g., Neal 2003,Mackay 2003)
    INPUT:
       initial_theta - initial sample
       step - stepping out step w
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf
       create_method - 'step_out', 'double', or 'whole' (whole only works if the domain is finite; defaults to 'double')
       nsamples - number of samples desired
       randomize_directions - (bool) pick a random coordinate to update
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
       callback - function of current parameters to call after each new sample
    OUTPUT:
       list of samples, number if nsamples=1
    REVISION HISTORY:
       2009-10-29 - Written - Bovy (NYU)
    """
    try:
        ndim = len(initial_theta)
    except TypeError:
        ndim= 1
    if ndim == 1: #1D
        return bovy_mcmc_oned.slice(initial_theta,step,lnpdf,pdf_params,
                                    create_method=create_method,
                                    isDomainFinite=isDomainFinite,
                                    domain=domain,
                                    nsamples=nsamples,
                                    callback=callback)
    else: #multi-D
        return bovy_mcmc_multid.slice(initial_theta,step,lnpdf,pdf_params,
                                      create_method=create_method,
                                      isDomainFinite=isDomainFinite,
                                      domain=domain,
                                      nsamples=nsamples,
                                      randomize_directions=\
                                          randomize_directions,
                                      callback=callback)

def metropolis(initial_theta,proposal_params,lnpdf,pdf_params,symmetric=False,
               sample_proposal=sample_gaussian_proposal,
               eval_ln_proposal=eval_ln_gaussian_proposal,
               nsamples=1,randomize_directions=True,callback=None):
    """
    NAME:
       metropolis
    PURPOSE:
       metropolis mcmc
    INPUT:
       initial_theta - initial sample
       proposal_params - parameters for the proposal function 
                         (e.g., typical steps)
                         (single for all dimensions or list of 
                         functions)
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf (tuple)
       sample_proposal - given x and proposal_params, sample a proposal 
                         using this function DEFAULT: GAUSSIAN
                         (single function for all dimensions or list of 
                         functions)
       eval_ln_proposal - given x and proposal_params, evaluate the log of 
                          the proposal density DEFAULT: GAUSSIAN
                         (single function for all dimensions or list of 
                         functions)
       symmetric - (bool) if True, the proposal distribution is symmetric and will not be evaluated (bool or list of bools
       randomize_directions - (bool) pick a random coordinate to update
       nsamples - number of samples desired
       callback - function of current parameters to call after each new sample
    OUTPUT:
       tuple consisting of
          list of samples, number if nsamples=1
          acceptance ratio, 1 or 0 if nsamples=1
    REVISION HISTORY:
       2011-07-27 - Written - Bovy (NYU)
    """
    try:
        ndim = len(initial_theta)
    except TypeError:
        ndim= 1
    if ndim == 1: #1D
        return bovy_mcmc_oned.metropolis(initial_theta,
                                         sample_proposal,
                                         eval_ln_proposal,
                                         proposal_params,lnpdf,
                                         pdf_params,symmetric=symmetric,
                                         nsamples=nsamples,callback=callback)
    else: #multi-D
        return bovy_mcmc_multid.metropolis(initial_theta,
                                           sample_proposal,
                                           eval_ln_proposal,
                                           proposal_params,lnpdf,
                                           pdf_params,symmetric=symmetric,
                                           nsamples=nsamples,
                                           randomize_directions=\
                                               randomize_directions,
                                           callback=callback)
if __name__ == '__main__':
    import doctest
    doctest.run_docstring_examples(markovpy,globals())
    import subprocess, sys
    if len(sys.argv) < 2:
        subprocess.check_call(['python','_bovy_mcmc.py'])
        subprocess.check_call(['python','_bovy_mv_mcmc.py'])
    else:
        if sys.argv[1] == '-v': sys.exit()
        subprocess.check_call(['python',sys.argv[1],'-v'])
