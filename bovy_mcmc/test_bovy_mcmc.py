import scipy as sc
import scipy.special as special
import sys
from bovy_mcmc import *
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
##############################################################################
#
# test_bovy_mcmc: test suite for bovy_mcmc.py
#
# Use: python test_bovy_mcmc.py method pdf
#      method =  mcmc method to test: 'slice', 'hmc', 'metropolis'
#      pdf = pdf to test it on: 'gaussian' or 'mixgaussian'
#      More options to follow
#
##############################################################################
def sample_gaussian_proposal(mean,stddev):
    """
    NAME:
       sample_gaussian_proposal
    PURPOSE:
       sample from a Gaussian proposal distribution for Metropolis sampling
    INPUT:
       mean - mean of the Gaussian
       stddev - standard deviation of the Gaussian
    OUTPUT:
       a sample from the Gaussian
    REVISION HISTORY:
       2009-10-30 - Written - Bovy
    """
    return stats.norm.rvs()*stddev+mean

def eval_ln_gaussian_proposal(x,mean,stddev):
    """
    NAME:
       eval_ln_gaussian_proposal
    PURPOSE:
       evaluate the probability of a sample from a Gaussian distribution
    INPUT:
       x - sample the evaluate the probability of
       mean - mean of the Gaussian
       stddev - standard deviation of the Gaussian
    OUTPUT:
       the log of N(x|mean,stddev^2)
    HISTORY:
       2009-10-30 - Written - Bovy
    """
    return -0.5*sc.log(2.*sc.pi*stddev**2.)-0.5*(x-mean)**2./stddev**2.

def minuslngaussian(x,params):
    """
    NAME:
       minuslngaussian
    PURPOSE:
       minus the log of a Gaussian for use in testing the HMC routine
    INPUT:
       x - evaluate ln gaussian at this point
       params - params[0] = mean; params[1]= variance
    OUTPUT:
       minus the ln of G(params)
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    return -lngaussian(x,params)

def grad_minuslngaussian(x,params):
    """
    NAME:
       grad_minuslngaussian
    PURPOSE:
       return the derivative of minus the log of a gaussian for use in
       testing the HMC routine
    INPUT:
       x - evaluate ln gaussian at this point
       params - params[0] = mean; params[1]= variance
    OUTPUT:
       the gradient
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    return (x-params[0])/params[1]

def lngaussian(x,params):
    """
    NAME:
       lngaussian
    PURPOSE:
       the log of a Gaussian
    INPUT:
       x - evaluate ln gaussian at this point
       params - params[0] = mean; params[1]= variance
    OUTPUT:
       the ln of G(params)
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    return -.5*sc.log(2.*sc.pi*params[1])-0.5*(x-params[0])**2./params[1]

def lnmixgaussian(x,params):
    """
    NAME:
       lnmixgaussian
    PURPOSE:
       the log of a Gaussian
    INPUT:
       x - evaluate ln gaussian at this point
       params - params[0] = mean1; params[1]= variance1;
                params[2] = mean2; params[3]= variance2;
                params[4]= amp1
    OUTPUT:
       the ln of G(params)
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    return sc.log(params[4]/sc.sqrt(2.*sc.pi*params[1])*sc.exp(-0.5*(x-params[0])**2./params[1])+(1.-params[4])/sc.sqrt(2.*sc.pi*params[3])*sc.exp(-0.5*(x-params[2])**2./params[3]))

def lnbeta(x,params):
    """
    NAME:
       lnbeta
    PURPOSE:
       the log of the beta distribution
    INPUT:
       x - evaluate ln beta at this point
       params - parameters of the beta distribution
                params[0]= alpha
                params[1]= beta (as on wikipedia)
    OUTPUT:
      the log of B(x;alpha,beta)
    REVISION HISTORY:
       2009-11-03 - Written - Bovy (NYU)
    """
    return (params[0]-1.)*sc.log(x)+(params[1]-1.)*sc.log(1-x)-special.betaln(params[0],params[1])

def test_slice(initial_theta,step,lnpdf,pdf_params,create_method,isDomainFinite,domain,
               plotfilename,nsamples=1000):
    """
    NAME:
       test_slice
    PURPOSE:
       test the slice sampling routine
    INPUT:
       initial_theta - initial sample
       step - stepping out step w
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf
       create_method - 'step_out' or 'double'
       plotfilename - filename for plot
       nsamples - number of samples to use
    OUTPUT:
       plot
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    samples= slice(initial_theta,step,lnpdf,pdf_params,create_method,isDomainFinite,domain,nsamples=nsamples)
    samples= sc.array(samples)
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    rcparams = {'axes.labelsize': 16,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.minor.size' : 2,
              'ytick.minor.size' : 2}
    rcParams.update(rcparams)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    figure()
    thishist=hist(samples[nsamples/2:-1],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    nabcissae= 1001
    abcissae= sc.linspace(thishist[1][0],thishist[1][-1],nabcissae)
    plotthese= sc.zeros(nabcissae)
    for ii in range(nabcissae):
        plotthese[ii]= lnpdf(abcissae[ii],pdf_params)
    plotthese= sc.exp(plotthese)
    plot(abcissae,plotthese,'k-')
    xlabel('$x$')
    yaxislabel='$p(x|'
    ylim(0.,sc.amax(thishist[0])*1.1)
    for ii in range(len(pdf_params)):
        yaxislabel+= str(pdf_params[ii])
        if ii < len(pdf_params)-1:
            yaxislabel+= ','
    yaxislabel+= ')$'
    ylabel(yaxislabel)
    savefig(plotfilename,format='png')
    
def test_metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,
                    lnpdf,pdf_params,symmetric,plotfilename,nsamples=1000):
    """
    NAME:
       test_metropolis
    PURPOSE:
       test the metropolis sampling routine
    INPUT:
       initial_theta - initial sample
       sample_proposal - given x and proposal_params, sample a proposal
       eval_ln_proposal - given x and proposal_params, evaluate the log of the proposal density
       proposal_params - parameters for the proposal function (e.g., typical steps)
       lnpdf - function evaluating the log of the pdf to be sampled
       pdf_params - parameters to pass to the pdf
       symmetric - (bool) if True, the proposal distribution is symmetric and will not be evaluated
       plotfilename - filename for plot
       nsamples - number of samples to use
    OUTPUT:
       plot
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    (samples,faccept)= metropolis(initial_theta,sample_proposal,eval_ln_proposal,proposal_params,lnpdf,pdf_params,symmetric=symmetric,nsamples=nsamples)
    print "%4.1f%% of the samples were accepted" % (100.*faccept)
    samples= sc.array(samples)
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    rcparams = {'axes.labelsize': 16,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.minor.size' : 2,
              'ytick.minor.size' : 2}
    rcParams.update(rcparams)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    figure()
    thishist=hist(samples[nsamples/2:-1],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    nabcissae= 1001
    abcissae= sc.linspace(thishist[1][0],thishist[1][-1],nabcissae)
    plotthese= sc.zeros(nabcissae)
    for ii in range(nabcissae):
        plotthese[ii]= lnpdf(abcissae[ii],pdf_params)
    plotthese= sc.exp(plotthese)
    plot(abcissae,plotthese,'k-')
    xlabel('$x$')
    yaxislabel='$p(x|'
    for ii in range(len(pdf_params)):
        yaxislabel+= str(pdf_params[ii])
        if ii < len(pdf_params)-1:
            yaxislabel+= ','
    yaxislabel+= ')$'
    ylabel(yaxislabel)
    savefig(plotfilename,format='png')
    
def test_hmc(initial_theta,nleap,stepsize,obj_func,grad_func,func_params,
             plotfilename,nsamples=1000):
    """
    NAME:
       test_hmc
    PURPOSE:
       test the Hamiltonian/Hybrid Monte Carlo sampling routine
    INPUT:
       initial_theta - initial state of the parameters
       nleap - (int) number of leapfrog steps per HMC step
       stepsize - (double) size of the steps to take in the orbit integration
       obj_func - (function pointer) the objective function E(x,params) as in p(x) ~ exp(-E)
       grad_func - (function pointer) the gradient of the objective function gradE(x,params)
       func_params - (tuple) the parameters of the objective function 
       plotfilename - filename for plot
       nsamples - number of samples to use
    OUTPUT:
       plot
    REVISION HISTORY:
       2009-10-29 - Written - Bovy
    """
    (samples,faccept)= hmc(initial_theta,nleap,stepsize,obj_func,grad_func,
                 func_params,nsamples=nsamples)
    print "%4.1f%% of the samples were accepted" % (100.*faccept)
    samples= sc.array(samples)
    fig_width=5
    fig_height=5
    fig_size =  [fig_width,fig_height]
    rcparams = {'axes.labelsize': 16,
              'text.fontsize': 11,
              'legend.fontsize': 12,
              'xtick.labelsize':10,
              'ytick.labelsize':10,
              'text.usetex': True,
              'figure.figsize': fig_size,
              'xtick.minor.size' : 2,
              'ytick.minor.size' : 2}
    rcParams.update(rcparams)
    rc('text.latex', preamble=r'\usepackage{amsmath}')
    figure()
    thishist=hist(samples[nsamples/2:-1],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    nabcissae= 1001
    abcissae= sc.linspace(thishist[1][0],thishist[1][-1],nabcissae)
    plotthese= sc.zeros(nabcissae)
    for ii in range(nabcissae):
        plotthese[ii]= -obj_func(abcissae[ii],pdf_params)
    plotthese= sc.exp(plotthese)
    plot(abcissae,plotthese,'k-')
    xlabel('$x$')
    yaxislabel='$p(x|'
    for ii in range(len(pdf_params)):
        yaxislabel+= str(pdf_params[ii])
        if ii < len(pdf_params)-1:
            yaxislabel+= ','
    yaxislabel+= ')$'
    ylabel(yaxislabel)
    savefig(plotfilename,format='png')
    
if __name__=='__main__':
    if sys.argv[1] == 'slice':
        print "Testing slice sampling..."
        if sys.argv[2] == 'gaussian':
            lnpdf= lngaussian
            pdf_params= [0.,1.]
            plotfilename='slice_gaussian.png'
            isDomainFinite= [False,False]
            domain= [0.,0.]
        elif sys.argv[2] == 'mixgaussian':
            lnpdf= lnmixgaussian
            pdf_params= [0.,1.,5.,2.,.4]
            plotfilename='slice_mixgaussian.png'
            isDomainFinite= [False,False]
            domain= [0.,0.]
        else:
            lnpdf= lnbeta
            #pdf_params= [2.,5.]
            pdf_params= [.5,.5]
            plotfilename='slice_beta.png'
            isDomainFinite= [True,True]
            domain= [0.,1.]
        if len(sys.argv) < 4:
            create_method= 'double'
        else:
            create_method= sys.argv[3]
        test_slice(0.1,1.,lnpdf,pdf_params,create_method,isDomainFinite,domain,
                   plotfilename=plotfilename,nsamples=10000)
    if sys.argv[1] == 'metropolis':
        print "Testing Metropolis sampling..."
        if sys.argv[2] == 'gaussian':
            lnpdf= lngaussian
            pdf_params= [0.,1.]
            sample_proposal= sample_gaussian_proposal
            eval_ln_proposal= eval_ln_gaussian_proposal
            proposal_params= 2.
            symmetric=False
            plotfilename='metropolis_gaussian.png'
        else:
            lnpdf= lnmixgaussian
            pdf_params= [0.,1.,5.,2.,.4]
            sample_proposal= sample_gaussian_proposal
            eval_ln_proposal= eval_ln_gaussian_proposal
            proposal_params= 2.
            symmetric=False
            plotfilename='metropolis_mixgaussian.png'
        test_metropolis(0.,sample_proposal,eval_ln_proposal,
                        proposal_params,lnpdf,pdf_params,
                        symmetric,plotfilename=plotfilename,
                        nsamples=100000)
    if sys.argv[1] == 'hmc':
        print "Testing HMC sampling..."
        if sys.argv[2] == 'gaussian':
            obj_func= minuslngaussian
            grad_func= grad_minuslngaussian
            pdf_params= [0.,1.]
            plotfilename='hmc_gaussian.png'
        else:
            print "This function has not been implemented (yet)"
            print "Returning..."
            sys.exit(-1)
        test_hmc(0,20,.05,obj_func,grad_func,pdf_params,
                 plotfilename=plotfilename,nsamples=10000)
