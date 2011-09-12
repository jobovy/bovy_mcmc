import scipy as sc
import scipy.stats as stats
##############################################################################
#
#  bovy_mcmc.py: general mcmc methods
#
##############################################################################
def hmc(initial_theta,nleap,stepsize,obj_func,grad_func,func_params,nsamples=1):
    """
    NAME:
       hmc
    PURPOSE:
       general HMC routine (uses the standard kinetic energy)
    INPUT:
       initial_theta - initial state of the parameters
       nleap - (int) number of leapfrog steps per HMC step
       stepsize - (double) size of the steps to take in the orbit integration
       obj_func - (function pointer) the objective function E(x,params) as in p(x) ~ exp(-E)
       grad_func - (function pointer) the gradient of the objective function gradE(x,params)
       func_params - (tuple) the parameters of the objective function 
       nsamples - (int) desired number of HMC samples
    OUTPUT:
       (a set of samples,acceptance fraction)
    BUGS:
       - does not use masses
       - only uses the last sample
    REVISION HISTORY:
       2009-10-08 - Written - Bovy (NYU)
       2009-10-29 - Rewritten and added to bovy_mcmc.py
    """
    out=[]
    try:
        ntheta= len(initial_theta)
    except TypeError:
        ntheta= 1
    E= obj_func(initial_theta,func_params)
    grad= grad_func(initial_theta,func_params)
    theta= initial_theta.copy()
    naccept= 0.
    for ii in range(nsamples):
        p= stats.norm.rvs(size=ntheta)
        H= 0.5*sc.dot(p,p) + E
        newtheta= theta.copy()
        newgrad= grad
        #First move the momentum
        p-= 0.5*newgrad*stepsize
        for kk in range(nleap):
            newtheta+= stepsize*p
            newgrad= grad_func(newtheta,func_params)
            p-= stepsize*newgrad/(1.+ (kk == (nleap-1)))#Full steps except for the last one
        Enew= obj_func(newtheta,func_params)
        Hnew= 0.5*sc.dot(p,p)+Enew
        dH= Hnew - H
        dH= dH * ( dH > 0 ) 
        #Metropolis accept
        if stats.uniform.rvs() < sc.exp(-dH):
            theta= newtheta.copy()
            E= Enew
            grad= newgrad
            naccept+= 1.
        out.append(theta)
    if nsamples == 1:
        return (out[0],naccept)
    else:
        return (out,naccept/nsamples)

def metropolis(initial_theta,sample_proposal,eval_ln_proposal,
               proposal_params,lnpdf,pdf_params,symmetric=False,
               nsamples=1,callback=None):
    """
    NAME:
       metropolis
    PURPOSE:
       metropolis mcmc
    INPUT:
       initial_theta - initial sample
       sample_proposal - given x and proposal_params, sample a proposal 
                         using this function
       eval_ln_proposal - given x and proposal_params, evaluate the log of 
                          the proposal density
       proposal_params - parameters for the proposal function 
                         (e.g., typical steps)
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf (tuple)
       symmetric - (bool) if True, the proposal distribution is symmetric and will not be evaluated
       nsamples - number of samples desired
       callback - function of parameter to be called after each new sample
    OUTPUT:
       tuple consisting of
          list of samples, number if nsamples=1
          acceptance ratio, 1 or 0 if nsamples=1
    REVISION HISTORY:
       2009-10-30 - Written - Bovy (NYU)
       2011-06-18 - Added doctest - Bovy
    DOCTEST:
    >>> import numpy as nu
    >>> nu.random.seed(1)
    >>> import scipy as sc
    >>> from scipy import stats
    >>> def lngaussian(x,mean,var):
    ...     return -.5*sc.log(2.*sc.pi*var)-0.5*(x-mean)**2./var
    >>> def sample_gaussian_proposal(mean,stddev):
    ...     return stats.norm.rvs()*stddev+mean
    >>> def eval_ln_gaussian_proposal(new,old,stddev):
    ...     return -0.5*sc.log(2.*sc.pi*stddev**2.)-0.5*(old-new)**2./stddev**2.
    >>> lnpdf= lngaussian
    >>> pdf_params= (0.,1.)
    >>> sample_proposal= sample_gaussian_proposal
    >>> eval_ln_proposal= eval_ln_gaussian_proposal
    >>> proposal_params= (2.,)
    >>> symmetric=False
    >>> initial_theta= 5.
    >>> nsamples= 200000
    >>> (samples,faccept)= metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,lnpdf,pdf_params,symmetric=symmetric,nsamples=nsamples)
    >>> print "%4.1f%% of the samples were accepted" % (100.*faccept)
    50.0% of the samples were accepted
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(samples)-1.)**2. < 10.**(logprecision*2.)
    >>> assert (stats.moment(samples,3)-0.)**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples,4)-stats.norm.moment(4))**2. < 10.**(logprecision)
    >>> from scipy import special
    >>> def lnbeta(x,a,b):
    ...     return (a-1.)*nu.log(x)+(b-1.)*nu.log(1-x)-special.betaln(a,b)
    >>> def sample_beta_proposal(x):
    ...     return nu.random.uniform()
    >>> def eval_ln_beta_proposal(new,old):
    ...     return 0.
    >>> lnpdf= lnbeta
    >>> pdf_params= (.5,.5)
    >>> sample_proposal= sample_beta_proposal
    >>> eval_ln_proposal= eval_ln_beta_proposal
    >>> proposal_params= ()
    >>> symmetric=False
    >>> initial_theta= 0.5
    >>> nsamples= 100000
    >>> nu.random.seed(1)
    >>> (samples,faccept)= metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,lnpdf,pdf_params,symmetric=symmetric,nsamples=nsamples)
    >>> print "%4.1f%% of the samples were accepted" % (100.*faccept)
    72.5% of the samples were accepted
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-stats.beta.moment(1,pdf_params[0],pdf_params[1]))**2. < 10.**(logprecision*2.)
    >>> assert (nu.var(samples)-stats.beta.moment(2,pdf_params[0],pdf_params[1])+stats.beta.moment(1,pdf_params[0],pdf_params[1])**2.)**2. < 10.**(logprecision*2.)
    """
    out= []
    naccept= 0.
    theta= initial_theta
    logp= lnpdf(theta,*pdf_params)
    for ii in range(nsamples):
        newtheta= sample_proposal(theta,*proposal_params)
        newlogp= lnpdf(newtheta,*pdf_params)
        if symmetric:
            extra_proposal_factor= 0.
        else:
            fromoldtonew= eval_ln_proposal(newtheta,theta,*proposal_params)
            fromnewtoold= eval_ln_proposal(theta,newtheta,*proposal_params)
            extra_proposal_factor= fromnewtoold-fromoldtonew
        u=stats.uniform.rvs()
        comp= newlogp-logp+extra_proposal_factor
        comp*= (comp < 0)
        if sc.log(u) < comp:
            theta= newtheta
            logp= newlogp
            naccept+= 1.
        if not callback is None: callback(theta)
        out.append(theta)
    if nsamples == 1:
        return (out[0],naccept)
    else:
        return (out,naccept/nsamples)

def slice_double(x,u,step,lnpdf,pdf_params,isDomainFinite,domain):
    """
    NAME:
       slice_double
    PURPOSE:
       doubling technique to create the interval in slice sampling (Neal 2003)
    INPUT:
       x          - current sample
       u          - current (log) height of the slice
       step       - step to take in stepping out
       lnpdf      - function evaluating the log of the pdf
       pdf_params - parameters to be passed to the pdf
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
    OUTPUT:
       (xl,xr) - lower and upper bound to the interval
    REVISION HISTORY:
       2009-10-29 - Written - Bovy (NYU)
    """
    r= stats.uniform.rvs()
    xl= x-r*step
    xr= x+(1-r)*step
    logpxl= lnpdf(xl,*pdf_params)
    logpxr= lnpdf(xr,*pdf_params)
    while logpxl > u or logpxr > u:
        v= stats.uniform.rvs()
        if v < .5:
            xl-= (xr-xl)
            if isDomainFinite[0] and xl < domain[0]:
                xl= domain[0]
                logpxl= u-1
            else:
                logpxl= lnpdf(xl,*pdf_params)
        else:
            xr+= (xr-xl)
            if isDomainFinite[1] and xr > domain[1]:
                xr= domain[1]
                logpxr= u-1
            else:
                logpxr= lnpdf(xr,*pdf_params)
    return (xl,xr)
    
def slice_step_out(x,u,step,lnpdf,pdf_params,isDomainFinite,domain):
    """
    NAME:
       slice_step_out
    PURPOSE:
       stepping out technique to create the interval in slice sampling (Mackay 2003)
    INPUT:
       x          - current sample
       u          - current (log) height of the slice
       step       - step to take in stepping out
       lnpdf      - function evaluating the log of the pdf
       pdf_params - parameters to be passed to the pdf
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
    OUTPUT:
       (xl,xr) - lower and upper bound to the interval
    REVISION HISTORY:
       2009-10-29 - Written - Bovy (NYU)
    """
    r= stats.uniform.rvs()
    xl= x-r*step
    xr= x+(1-r)*step
    while lnpdf(xl,*pdf_params) > u:
        xl-= step
        if isDomainFinite[0] and xl < domain[0]:
            xl= domain[0]
            break
    while lnpdf(xr,*pdf_params) > u:
        xr+= step
        if isDomainFinite[1] and xr > domain[1]:
            xr= domain[1]
            break
    return (xl,xr)

def slice_whole(x,u,step,lnpdf,pdf_params,isDomainFinite,domain):
    """
    NAME:
       slice_whole
    PURPOSE:
       create the interval in slice sampling by using the whole, finite domain
    INPUT:
       x          - current sample
       u          - current (log) height of the slice
       step       - step to take in stepping out
       lnpdf      - function evaluating the log of the pdf
       pdf_params - parameters to be passed to the pdf
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
    OUTPUT:
       (xl,xr) - lower and upper bound to the interval
    REVISION HISTORY:
       2009-11-03 - Written - Bovy (NYU)
    """
    return (domain[0],domain[1])

def slice_shrink(xp,x,interval):
    """
    NAME:
       slice_shrink
    PURPOSE:
       shrink the interval in slice sampling (Mackay 2003)
    INPUT:
       xp       - proposed sample
       x        - current sample
       interval - the current interval
    OUTPUT:
       new interval
    REVISION HISTORY:
       2009-10-29 - Written - Bovy (NYU)
    """
    if xp > x:
        xr= xp
        xl= interval[0]
    else:
        xl= xp
        xr= interval[1]
    return (xl,xr)

def slice(initial_theta,step,lnpdf,pdf_params,create_method='step_out',isDomainFinite=[False,False],domain=[0.,0.],
          nsamples=1,callback=None):
    """
    NAME:
       slice
    PURPOSE:
       simple slice sampling function (e.g., Neal 2003,Mackay 2003)
    INPUT:
       initial_theta - initial sample
       step - stepping out step w
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf (tuple)
       create_method - 'step_out', 'double', or 'whole' (whole only works if the domain is finite; defaults to 'double')
       nsamples - number of samples desired
       isDomainFinite - is the domain finite? [bool,bool]
       domain - the domain if it is finite (has no effect if the domain is not finite)
       callback - function of parameter to be called after each new sample
    OUTPUT:
       list of samples, number if nsamples=1
    REVISION HISTORY:
       2009-10-29 - Written - Bovy (NYU)
    DOCTEST:
    >>> import numpy as nu
    >>> nu.random.seed(1)
    >>> def lngaussian(x,mean,var):
    ...     return -.5*nu.log(2.*nu.pi*var)-0.5*(x-mean)**2./var
    >>> pdf_params= (0.,1.)
    >>> isDomainFinite= [False,False]
    >>> domain= [0.,0.]
    >>> create_method= 'double'
    >>> nsamples= 100000
    >>> samples= slice(0.1,1.,lngaussian,pdf_params,create_method,isDomainFinite,domain,nsamples=nsamples)
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-0.)**2. < 10.**(logprecision*2.)
    >>> assert (nu.std(samples)-1.)**2. < 10.**(logprecision*2.)
    >>> from scipy import stats
    >>> assert (stats.moment(samples,3)-0.)**2. < 10.**(logprecision)
    >>> assert (stats.moment(samples,4)-stats.norm.moment(4))**2. < 10.**(logprecision)
    >>> from scipy import special
    >>> def lnbeta(x,a,b):
    ...     return (a-1.)*nu.log(x)+(b-1.)*nu.log(1-x)-special.betaln(a,b)
    >>> pdf_params= (.5,.5)
    >>> isDomainFinite= [True,True]
    >>> domain= [0.,1.]
    >>> create_method= 'double'
    >>> samples= slice(0.1,1.,lnbeta,pdf_params,create_method,isDomainFinite,domain,nsamples=nsamples)
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-stats.beta.moment(1,pdf_params[0],pdf_params[1]))**2. < 10.**(logprecision*2.)
    >>> assert (nu.var(samples)-stats.beta.moment(2,pdf_params[0],pdf_params[1])+stats.beta.moment(1,pdf_params[0],pdf_params[1])**2.)**2. < 10.**(logprecision*2.)
    >>> create_method= 'step_out'
    >>> samples= slice(0.1,1.,lnbeta,pdf_params,create_method,isDomainFinite,domain,nsamples=nsamples)
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-stats.beta.moment(1,pdf_params[0],pdf_params[1]))**2. < 10.**(logprecision*2.)
    >>> assert (nu.var(samples)-stats.beta.moment(2,pdf_params[0],pdf_params[1])+stats.beta.moment(1,pdf_params[0],pdf_params[1])**2.)**2. < 10.**(logprecision*2.)
    >>> create_method= 'whole'
    >>> samples= slice(0.1,1.,lnbeta,pdf_params,create_method,isDomainFinite,domain,nsamples=nsamples)
    >>> samples= samples[nsamples/2:-1] #discard burn-in
    >>> logprecision= -2.
    >>> assert (nu.mean(samples)-stats.beta.moment(1,pdf_params[0],pdf_params[1]))**2. < 10.**(logprecision*2.)
    >>> assert (nu.var(samples)-stats.beta.moment(2,pdf_params[0],pdf_params[1])+stats.beta.moment(1,pdf_params[0],pdf_params[1])**2.)**2. < 10.**(logprecision*2.)
    """
    if create_method == 'step_out':
        create_interval= slice_step_out
        accept= slice_step_out_accept
    elif create_method == 'double':
        create_interval= slice_double
        accept= slice_double_accept
    else:
        if isDomainFinite[0] and isDomainFinite[1]:
            create_interval= slice_whole
            accept= slice_step_out_accept
        else:
            create_interval= slice_double
            accept= slice_double_accept
    modify_interval= slice_shrink
    x= initial_theta
    logp= lnpdf(x,*pdf_params)
    out= []
    for ii in range(nsamples):
        u= logp-stats.expon.rvs()#p.712 in Neal (2003)
        (xl,xr)= create_interval(x,u,step,lnpdf,pdf_params,isDomainFinite,domain)
        while True:
            xp= stats.uniform.rvs()*(xr-xl)+xl
            logpxp= lnpdf(xp,*pdf_params)
            if logpxp >= u and accept(xp,x,u,step,(xl,xr),lnpdf,pdf_params):#Equal sign from Neal comment on Gelman blog
                break
            (xl,xr)= modify_interval(xp,x,(xl,xr))
        if not callback is None: callback(xp)
        out.append(xp)
        x=xp
        logp= logpxp
    if nsamples == 1:
        return out[0]
    else:
        return out

def slice_double_accept(xp,x,u,step,interval,lnpdf,pdf_params):
    """
    NAME:
       slice_double_accept
    PURPOSE:
       accept a step when using the doubling procedure
    INPUT:
       xp         - proposed point
       x          - current point
       u          - log of the height of the slice
       step       - step parameter w
       interval   - (xl,xr)
       lnpdf      - function that evaluates the log of the pdf
       pdf_params - parameters of the pdf
    OUTPUT:
       Whether to accept or not (Neal 2003)
    BUGS:
       Not as efficient as possible with lnpdf evaluations
    HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    """
    (xl,xr) = interval
    d= False
    acceptable= True
    while xr-xl > 1.1*step:
        m= (xl+xr)*.5
        if (x < m and xp >= m) or (x >= m and xp < m):
            d= True
        if xp < m:
            xr= m
        else:
            xl= m
        if d and lnpdf(xl,*pdf_params) <= u and lnpdf(xr,*pdf_params) <= u:
            acceptable= False
            break
    return acceptable

def slice_step_out_accept(xp,x,u,step,interval,lnpdf,pdf_params):
    """
    NAME:
       slice_step_out_accept
    PURPOSE:
       accept a step when using the stepping out procedure
    INPUT:
       xp         - proposed point
       x          - current point
       u          - log of the height of the slice
       step       - step parameter w
       interval   - (xl,xr)
       lnpdf      - function that evaluates the log of the pdf
       pdf_params - parameters of the pdf
    OUTPUT:
       True
    HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    """
    return True

if __name__ == '__main__':
    import doctest
    doctest.testmod()
