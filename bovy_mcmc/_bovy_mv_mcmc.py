import copy
import scipy as sc
import scipy.stats as stats
from . import _bovy_mcmc as oned_mcmc
##############################################################################
#
#  bovy_mv_mcmc.py: general multivariate mcmc methods based on 1D methods in
#                   bovy_mcmc.py
#
##############################################################################
def oned_lnpdf(x,params):
    """
    NAME:
       oned_lnpdf
    PURPOSE:
       packages a multi-dimensional pdf to sample one-dimensional slices of it
       (conditioned on k-1 of the other parameters)
    INPUT:
       x - one-d sample
       params - parameter vector: contains
                1) lnpdf function
                2) regular pdf parameters
                3) dimension that is being updated
                4) remaining parameters (list)
    OUTPUT:
       lnpdf evaluated at x conditioned on the other parameters
    HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    """
    theta= copy.deepcopy(params[3])
    theta.insert(params[2],x)
    return params[0](sc.array(theta),*params[1])

def slice(initial_theta,step,lnpdf,pdf_params,create_method='step_out',randomize_directions=True,isDomainFinite=[False,False],domain=[0.,0.],
          nsamples=1,callback=None):
    """
    NAME:
       slice
    PURPOSE:
       simple slice sampling function (e.g., Neal 2003,Mackay 2003)
       performs random, coordinate aligned updates
    INPUT:
       initial_theta - ([k]) initial sample
       step - (1 or [k]) stepping out step w
       lnpdf - function evaluating the log of the pdf to be sampled,
               arguments are initial_theta + pdf_params
       pdf_params - parameters to pass to the pdf
       create_method - 'step_out', 'double', or whole (string or array of D strings)
       randomize_directions - (bool) pick a random coordinate to update
       isDomainFinite - is the domain finite? [[bool,bool],...]
       domain - the domain if it is finite (has no effect if the domain is not finite) [[0.,0.],...]
       nsamples - number of samples desired
       callback - function of parameter to call after new sample
    OUTPUT:
       list of samples, number if nsamples=1
    REVISION HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    DOCTEST:
    >>> import numpy as nu
    >>> import scipy as sc
    >>> from scipy import linalg
    >>> nu.random.seed(1)
    >>> def lngaussian(x,params):
    ...        return -sc.log(2.*sc.pi)+.5*sc.log(linalg.det(params[1]))-0.5*sc.dot(x-params[0],sc.dot(params[1],x-params[0]))
    >>> lnpdf= lngaussian
    >>> mean= nu.array([0.,0.])
    >>> variance= nu.array([[1.,1],[1,4.]])
    >>> invvar= linalg.inv(variance)
    >>> pdf_params= ([mean,invvar],)
    >>> isDomainFinite= [False,False]
    >>> domain= [0.,0.]
    >>> create_method= 'double'
    >>> nsamples= 10000
    >>> samples= slice(nu.array([0.1,0.1]),1.,lnpdf,pdf_params,create_method,
    ...        True,isDomainFinite,domain,
    ...        nsamples=nsamples)
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
    out= []
    d= len(initial_theta)
    if not isinstance(isDomainFinite,sc.ndarray):
        isDomainFinite= sc.array(isDomainFinite)
    if not isinstance(domain,sc.ndarray):
        domain= sc.array(domain)
    if isinstance(step,list): step= sc.array(step)
    if isinstance(step,(int,float)) or len(step) == 1:
        step= sc.ones(d)*step
    if len(isDomainFinite.shape) == 1:
        dFinite= []
        for ii in range(d):
            dFinite.append(isDomainFinite)
        isDomainFinite= dFinite
    if len(domain.shape) == 1:
        dDomain= []
        for ii in range(d):
            dDomain.append(domain)
        domain= dDomain
    if not isinstance(create_method,list) or len(create_method) == 1:
        tmp_method= []
        for ii in range(d):
            if isinstance(create_method,list):
                tmp_method.append(create_method[0])
            else:
                tmp_method.append(create_method)
        create_method= tmp_method
    current_sample= initial_theta.copy()
    params= []
    params.append(lnpdf)
    params.append(pdf_params)
    for ii in range(nsamples):
        if randomize_directions:
            permuterange= sc.random.permutation(d)
        else:
            permuterange= range(d)
        for dd in permuterange:
            thisparams= copy.copy(params)
            thisparams.append(dd)
            thisparams.append([current_sample[jj] for jj in range(d) if jj != dd])#list comprehensions are supposedly faster than numpy slicing
            new_up_sample= oned_mcmc.slice(current_sample[dd],step[dd],oned_lnpdf,(thisparams,),create_method[dd],
                                           isDomainFinite[dd],domain[dd],nsamples=1)
            current_sample= current_sample.copy()
            current_sample[dd]= new_up_sample
        if not callback is None: callback(current_sample)
        out.append(current_sample)
    if nsamples == 1:
        return out[0]
    else:
        return out

def metropolis(initial_theta,sample_proposal,eval_ln_proposal,
               proposal_params,lnpdf,pdf_params,symmetric=False,
               nsamples=1,randomize_directions=True,callback=None):
    """
    NAME:
       metropolis
    PURPOSE:
       metropolis mcmc
    INPUT:
       initial_theta - initial sample
       sample_proposal - given x and proposal_params, sample a proposal
                         using this function
                         (single function for all dimensions or list of
                         functions)
       eval_ln_proposal - given x and proposal_params, evaluate the log of
                          the proposal density
                         (single function for all dimensions or list of
                         functions)
       proposal_params - parameters for the proposal function
                         (e.g., typical steps)
                         (single for all dimensions or list of
                         functions)
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf (tuple)
       symmetric - (bool) if True, the proposal distribution is symmetric and will not be evaluated (bool or list of bools
       randomize_directions - (bool) pick a random coordinate to update
       nsamples - number of samples desired
    OUTPUT:
       tuple consisting of
          list of samples, number if nsamples=1
          acceptance ratio, 1 or 0 if nsamples=1
    REVISION HISTORY:
       2011-07-27 - Written - Bovy (NYU)
    DOCTEST:
    >>> import numpy as nu
    >>> nu.random.seed(1)
    >>> import scipy as sc
    >>> from scipy import linalg
    >>> from scipy import stats
    >>> def lngaussian(x,params):
    ...     return -sc.log(2.*sc.pi)+.5*sc.log(linalg.det(params[1]))-0.5*sc.dot(x-params[0],sc.dot(params[1],x-params[0]))
    >>> def sample_gaussian_proposal(mean,stddev):
    ...     return stats.norm.rvs()*stddev+mean
    >>> def eval_ln_gaussian_proposal(new,old,stddev):
    ...     return -0.5*sc.log(2.*sc.pi*stddev**2.)-0.5*(old-new)**2./stddev**2.
    >>> lnpdf= lngaussian
    >>> pdf_params= ([sc.array([0.,1.]),sc.array([[1.,0.],[0.,4.]])],)
    >>> sample_proposal= sample_gaussian_proposal
    >>> eval_ln_proposal= eval_ln_gaussian_proposal
    >>> proposal_params= (2.,)
    >>> symmetric=False
    >>> initial_theta= nu.array([5.,-3])
    >>> nsamples= 40000
    >>> samples,faccept= metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,lnpdf,pdf_params,symmetric=symmetric,nsamples=nsamples)
    >>> print "%4.1f%% of the samples were accepted" % (100.*nu.mean(faccept))
    39.6% of the samples were accepted
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> xs= [s[0] for s in samples]
    >>> ys= [s[1] for s in samples]
    >>> logprecision= -2.
    >>> assert (nu.mean(xs)-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.mean(ys)-1.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(xs)-1.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(ys)-.5)**2. < 10.**(logprecision*2.)
    >>> proposal_params= [(2.,),(4.,)]
    >>> initial_theta= nu.array([5.,-3])
    >>> nsamples= 80000
    >>> samples,faccept= metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,lnpdf,pdf_params,symmetric=symmetric,nsamples=nsamples)
    >>> print "%4.1f%% of the samples were accepted" % (100.*nu.mean(faccept))
    32.6% of the samples were accepted
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> xs= [s[0] for s in samples]
    >>> ys= [s[1] for s in samples]
    >>> logprecision= -2.
    >>> assert (nu.mean(xs)-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.mean(ys)-1.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(xs)-1.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(ys)-.5)**2. < 10.**(logprecision*2.)
    """
    out= []
    d= len(initial_theta) #dimensionality
    if isinstance(proposal_params,tuple):
        proposal_params= [proposal_params for ii in range(d)]
    elif isinstance(proposal_params,list) \
            and isinstance(proposal_params[0],(int,float)):
        proposal_params= [(proposal_params[ii],) for ii in range(d)]
    elif isinstance(proposal_params,(int,float)):
        proposal_params= [(proposal_params,) for ii in range(d)]
    if not isinstance(sample_proposal,list):
        sample_proposal= [sample_proposal for ii in range(d)]
    if not isinstance(eval_ln_proposal,list):
        eval_ln_proposal= [eval_ln_proposal for ii in range(d)]
    if isinstance(symmetric,bool):
        symmetric= [symmetric for ii in range(d)]
    current_sample= initial_theta.copy()
    params= []
    params.append(lnpdf)
    params.append(pdf_params)
    naccept= sc.zeros(d)
    for ii in range(nsamples):
        if randomize_directions:
            permuterange= sc.random.permutation(d)
        else:
            permuterange= range(d)
        for dd in permuterange:
            thisparams= copy.copy(params)
            thisparams.append(dd)
            thisparams.append([current_sample[jj] for jj in range(d) if jj != dd])#list comprehensions are supposedly faster than numpy slicing
            new_up_sample,accepted= oned_mcmc.metropolis(current_sample[dd],
                                                         sample_proposal[dd],
                                                         eval_ln_proposal[dd],
                                                         proposal_params[dd],
                                                         oned_lnpdf,(thisparams,),
                                                         symmetric=symmetric[dd],
                                                         nsamples=1)
            current_sample= current_sample.copy()
            current_sample[dd]= new_up_sample
            naccept[dd]+= accepted
        if not callback is None: callback(current_sample)
        out.append(current_sample)
    if nsamples == 1:
        return (out[0],naccept)
    else:
        return (out,naccept/float(nsamples))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
