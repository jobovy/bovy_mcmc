import scipy as sc
import scipy.linalg as linalg
import scipy.special as special
import sys
from bovy_mv_mcmc import *
from logsum import logsum
import matplotlib
matplotlib.use('Agg')
from pylab import *
from matplotlib.pyplot import *
from matplotlib import rc
##############################################################################
#
# test_bovy_mv_mcmc: test suite for bovy_mv_mcmc.py
#
# Just plots 1D marginals for now
#
# Use: python test_bovy_mv_mcmc.py method pdf
#      method =  mcmc method to test: 'slice'
#      pdf = pdf to test it on: 'gaussian' or 'mixgaussian'
#      More options to follow
#
##############################################################################
def lngaussian(x,params):
    """
    NAME:
       lngaussian
    PURPOSE:
       returns the log of a two-dimensional gaussian
    INPUT:
       x - 2D point to evaluate the Gaussian at
       params - mean and variances ([mean_array,inverse variance matrix])
    OUTPUT:
       log N(mean,var)
    HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    """
    return -sc.log(2.*sc.pi)+.5*sc.log(linalg.det(params[1]))-0.5*sc.dot(x-params[0],sc.dot(params[1],x-params[0]))

def lnmixgaussian(x,params):
    """
    NAME:
       lnmixgaussian
    PURPOSE:
       returns the log of a mixture of two two-dimensional gaussian
    INPUT:
       x - 2D point to evaluate the Gaussian at
       params - mean and variances ([mean_array,inverse variance matrix, mean_array, inverse variance, amp1])
    OUTPUT:
       log N(mean,var)
    HISTORY:
       2009-10-30 - Written - Bovy (NYU)
    """
    return sc.log(params[4]/2./sc.pi*sc.sqrt(linalg.det(params[1]))*sc.exp(-0.5*sc.dot(x-params[0],
                                                                                       sc.dot(params[1],x-params[0])))
                  +(1.-params[4])/2./sc.pi*sc.sqrt(linalg.det(params[3]))*sc.exp(-0.5*sc.dot(x-params[2],
                                                                                             sc.dot(params[3],x-params[2]))))

def lnbetagaussian(x,params):
    """
    NAME:
       lnbetagaussian
    PURPOSE:
       2D distribution for which x has a beta distribution and y a gaussian
    INPUT:
       x - 2D point to evaluate the distribution at
       params - parameters of the distribution (params[0]= alpha, params[1]= beta, params[2]= mu, params[3]= var)
    OUTPUT:
       the log of the probability distribution
    HISTORY:
       2009-11-03 - Written - Bovy (NYU)
    """
    return (params[0]-1.)*sc.log(x[0])+(params[1]-1.)*sc.log(1-x[0])-special.betaln(params[0],params[1])-.5*sc.log(2.*sc.pi*params[1])-0.5*(x[1]-params[2])**2./params[3]

def test_slice(initial_theta,step,lnpdf,pdf_params,create_method,randomize_directions,isDomainFinite,domain,
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
       randomize_directions - (bool) pick a random coordinate to update
       isDomainFinite - is the domain finite? [[bool,bool],...]
       domain - the domain if it is finite (has no effect if the domain is not finite) [[0.,0.],...]
       plotfilename - filename for plot
       nsamples - number of samples to use
    OUTPUT:
       plot
    REVISION HISTORY:
       2009-10-30 - Written - Bovy
    """
    samples= slice(initial_theta,step,lnpdf,pdf_params,create_method,randomize_directions,isDomainFinite,domain,
                   nsamples=nsamples)
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
    #First marginal
    thishistx=hist(samples[nsamples/2:-1,0],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    thishisty=hist(samples[nsamples/2:-1,1],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    print("Plotting...")
    nabcissae= 101
    abcissaex= sc.linspace(thishistx[1][0],thishistx[1][-1],nabcissae)
    abcissaey= sc.linspace(thishisty[1][0],thishisty[1][-1],nabcissae)
    plotthese= sc.zeros((nabcissae,nabcissae))
    for ii in range(nabcissae):
        for jj in range(nabcissae):
            plotthese[ii,jj]= lnpdf([abcissaex[ii],abcissaey[jj]],pdf_params)
    plotthesex= sc.zeros(nabcissae)
    for ii in range(nabcissae):
        plotthesex[ii]= logsum(plotthese[ii,:])
    plotthesex= sc.exp(plotthesex)
    #normx= sc.sum(plotthesex)*(thishistx[1][-1]-thishistx[1][0])/nabcissae
    normx= (thishisty[1][-1]-thishisty[1][0])/nabcissae
    plotthesex*= normx
    figure()
    hist(samples[nsamples/2:-1,0],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    plot(abcissaex,plotthesex,'k-')
    xlabel(r'$x$')
    ylabel(r'$p(x)$')
    ylim(0.,1.1*sc.amax(thishistx[0]))
    basefilename=plotfilename.split('.')
    thisplotfilename=basefilename[0]+'_x.'+basefilename[1]
    savefig(thisplotfilename,format='png')
    #Second marginal
    plotthesey= sc.zeros(nabcissae)
    for ii in range(nabcissae):
        plotthesey[ii]= logsum(plotthese[:,ii])
    plotthesey= sc.exp(plotthesey)
    normy= sc.sum(plotthesey)*(thishisty[1][-1]-thishisty[1][0])/nabcissae
    plotthesey/= normy
    figure()
    hist(samples[nsamples/2:-1,1],bins=.2*sc.sqrt(nsamples),ec='k',fc='None',normed=True)
    plot(abcissaey,plotthesey,'k-')
    xlabel(r'$y$')
    ylabel(r'$p(y)$')
    ylim(0.,1.1*sc.amax(thishisty[0]))
    basefilename=plotfilename.split('.')
    thisplotfilename=basefilename[0]+'_y.'+basefilename[1]
    savefig(thisplotfilename,format='png')


if __name__=='__main__':
    if sys.argv[1] == 'slice':
        print("Testing slice sampling...")
        if sys.argv[2] == 'gaussian':
            lnpdf= lngaussian
            mean= sc.array([0.,0.])
            variance= sc.array([[1.,1],[1,4.]])
            invvar= linalg.inv(variance)
            pdf_params= [mean,invvar]
            plotfilename='slice_mv_gaussian.png'
            isDomainFinite= [False,False]
            domain= [0.,0.]
        elif sys.argv[2] == 'mixgaussian':
            lnpdf= lnmixgaussian
            mean1= sc.array([0.,0.])
            variance1= sc.array([[1.,1],[1.,4.]])
            invvar1= linalg.inv(variance1)
            mean2= sc.array([8.,7.])
            variance2= sc.array([[12.,3],[3.,4.]])
            invvar2= linalg.inv(variance2)
            pdf_params= [mean1,invvar1,mean2,invvar2,.2]
            plotfilename='slice_mv_mixgaussian.png'
            isDomainFinite= [False,False]
            domain= [0.,0.]
        else:
            lnpdf= lnbetagaussian
            pdf_params= [0.5,.5,0.,1.]
            plotfilename='slice_mv_betagaussian.png'
            isDomainFinite= [[True,True],[False,False]]
            domain= [[0.,1.],[0.,0.]]
        if len(sys.argv) < 4:
            create_method= 'double'
        else:
            create_method= sys.argv[3]
        test_slice(sc.array([0.1,0.1]),1.,lnpdf,pdf_params,create_method,True,
                   isDomainFinite,domain,plotfilename=plotfilename,nsamples=10000)
