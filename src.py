# import logging
# logging.getLogger('redback').setLevel(logging.ERROR)
# logging.getLogger('bilby').setLevel(logging.ERROR)
import redback
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import dill
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import afterglowpy as grb
import bilby
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os.path
from pathlib import Path
import swifttools.ukssdc.data as ud #https://www.swift.ac.uk/API/ukssdc/data/GRB.md
import swifttools.ukssdc.query as uq #add link
import swifttools.ukssdc.xrt_prods as ux #add link
import swifttools.ukssdc.data.GRB as udg # https://www.swift.ac.uk/API/ukssdc/data/GRB.md
from swifttools.ukssdc import plotLightCurve #https://www.swift.ac.uk/API/ukssdc/commonFunc.md#plotlightcurve
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import Pool
from tqdm import tqdm
from uncertainties import unumpy
from uncertainties import ufloat
import sys
num_cpus = os.cpu_count()
num_cpus=num_cpus+2 #ensure saturation
import tqdm
import tqdm.auto
def tqdm_plain(*args, **kwargs):
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("file", sys.stdout)
    return tqdm.std.tqdm(*args, **kwargs)
tqdm.auto.tqdm = tqdm_plain 
mpl.rcParams.update(mpl.rcParamsDefault) #fix a LaTeX bug with matplotlib in redback
mpl.rcParams['text.usetex'] = False
#runfile('/Users/sam/Documents/Project_xray/Updated/x_ray_functions_updated.py')
file_path_inputs = ('/Users/sam/Documents/Project_lc_modeling/Inputs/')
file_path_outputs=('/Users/sam/Documents/Project_lc_modeling/Outputs/Automation/')
file_path_fits='/Users/sam/Documents/Project_lc_modeling/Outputs/Automation/Fits/'
paper_figures_outputs='/Users/sam/Documents/Project_lc_modeling/Paper_figures/'

sample=False

#get data 
data=pd.read_csv('260924A_data.txt',sep=',',header=0,names=['Time','TimePos','TimeNeg','Rate','RatePos','RateNeg'])
data['Redshift']=3.51 #https://gcn.nasa.gov/circulars/45740?view=index&query=260924a&startDate=&endDate=&sort=circularID
data['ECF']=3.84e-11 #erg /cm2 /ct https://www.swift.ac.uk/xrt_spectra/01513082/
data['gamma']=2.04 #https://www.swift.ac.uk/xrt_spectra/01513082/
data['frequency']=2.418000e+17
data['band']='X-ray'

#append symmetric errors
data['RateErr']=(np.sqrt(np.power(data['RatePos'],2))+np.sqrt(np.power(data['RateNeg'],2)))/2
data['TimeErr']=(np.sqrt(np.power(data['TimePos'],2))+np.sqrt(np.power(data['TimeNeg'],2)))/2

#remove flares and steep decay
data=data[(data['Time']<=115) | (data['Time']>=127)]
data=data[(data['Time']<=311) | (data['Time']>=3647)]
data=data[data['Time']>=1e3]


#convert to unabsorbed flux (erg/cm2/s)
data['Native band flux'] = data['Rate']*data['ECF']
data['Native band flux error'] = data['RateErr']*data['ECF']

#convert to flux density at 1 keV (erg/cm2/s/Hz)
flux=unumpy.uarray(data['Native band flux'].values,data['Native band flux error']) #append into uncertainty array
E,E_2,E_1=1,10,0.3 #we convert to 1 keV and define a upper and lower values of bandpass
numerator=4.13*(10**11)*(2-data['gamma'])*(np.power(E,1-data['gamma'])) 
denominator=((E_2**(2-data['gamma']))-(E_1**(2-data['gamma'])))
flux=flux*(numerator/denominator) #calculates flux in microJy
flux=flux*1e-29 #converting from microJy to erg/s/cm2/Hz
data['Flux density'],data['Flux density error (symmetric)']=unumpy.nominal_values(flux),unumpy.std_devs(flux)

#convert into redback units
data['Time_d']=data['Time'].values/(60*60*24) #convert from s to d
data['Time_e']=data['TimeErr'].values/(60*60*24) #convert from s to d
data['Flux_mjy']=data['Flux density'].values*1e26 #convert from erg/s/cm2/hz to mJy
data['Flux_err_mjy']=data['Flux density error (symmetric)'].values*1e26 #convert from erg/s/cm2/hz to mJy

#plot light curve
plt.errorbar(x=data['Time_d'],xerr=data['Time_e'],y=data['Flux_mjy'],yerr=data['Flux_err_mjy'],fmt='o',color='black',ms=2,fillstyle=None)
plt.xscale('log')
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=16)
plt.ylabel('Flux density [mJy]',fontsize=16)
plt.xlabel('Time since explosion [d]',fontsize=16)
plt.title('GRB 260924A - no steep decay or flares')
plt.show()

#model kwargs
kwargs={}
kwargs['sampler']='dynamic_dynesty'
kwargs['sample']='rslice'
kwargs['bound']='multi'
kwargs['slices']=15 #lowered from 25 
kwargs['dlogz_init']=0.1 #threshold used in the baseline static run prior to the dynamic loop
kwargs['nlive_init']=250 #number of live points used as the baseline static run prior to the dynamic loop
# kwargs['enlarge']=1.5 #enlarge the bounds 
kwargs['maxbatch']=20 #maximum number of batches 
kwargs['nlive_batch']=250 #number of live points added per batch in the dynamic loop
kwargs['use_stop']=True #use default stopping criteria
kwargs['check_point']=False
kwargs['verbose']=True
kwargs['queue_size']=num_cpus
sampler=kwargs['sampler']
del(kwargs['sampler'])

#sampling priors
priors = bilby.core.prior.PriorDict()
priors['thvthc']=bilby.core.prior.Uniform(minimum=0,maximum=1.5, name='thvthc', latex_label='$\\theta_{\\mathrm{obs}}/\\theta_{\\mathrm{obs}}~(\\mathrm{rad})$', unit=None, boundary=None)
priors['ksin']=1 #fixed due to degeneracy with other params (see https://www.aanda.org/articles/aa/full_html/2024/10/aa50116-24/aa50116-24.html)
priors['loge0']=bilby.core.prior.Uniform(minimum=46, maximum=55, name='loge0', latex_label='$\\log_{10}~E_{0}/{\\mathrm{erg}}$', unit=None, boundary=None)
priors['thc']=bilby.core.prior.Uniform(minimum=0.01, maximum=0.2, name='thc', latex_label='$\\theta_{\\mathrm{core}}~({\\mathrm{rad}})$', unit=None, boundary=None)
priors['logn0']=bilby.core.prior.Uniform(minimum=-5, maximum=2, name='logn0', latex_label='$\\log_{10}~n_{\\mathrm{ism}}/{\\mathrm{cm}}^{-3}$', unit=None, boundary=None)
priors['p']=bilby.core.prior.Uniform(minimum=2, maximum=3, name='p', latex_label='$p$', unit=None, boundary=None)
priors['logepse']=bilby.core.prior.Uniform(minimum=-5, maximum=0, name='logepse', latex_label='$\\log_{10}~\\epsilon_{e}$', unit=None, boundary=None)
priors['logepsb']=bilby.core.prior.Uniform(minimum=-5, maximum=0, name='logepsb', latex_label='$\\log_{10}~\\epsilon_{B}$', unit=None, boundary=None)
priors['g0']=bilby.core.prior.Uniform(minimum=50, maximum=500, name='g0', latex_label='$g0$', unit=None, boundary=None)
priors['thw']=6
priors['beta']=bilby.core.prior.Uniform(minimum=0.5, maximum=10, name='beta', latex_label='$\\beta$', unit=None, boundary=None)
priors['redshift']=data['Redshift'].unique()[0]

#model wrappers
class convert_thvthc_to_thv(object):

    def __init__(self, thvthc, thc):
        self.thvthc = thvthc
        self.thc = thc
        self.thv = self.calculate_thv

    @property
    def calculate_thv(self):
        thv=(self.thvthc)*(self.thc)
        return thv
def wrapped_tophat(time,**kwargs):
    #convert thvthc into thv before passing into the model
    kwargs['thv']=convert_thvthc_to_thv(kwargs['thvthc'],kwargs['thc']).thv
    #call model
    y=redback.transient_models.afterglow_models.tophat(time,**kwargs)
    return(y)


wrapped_model=wrapped_tophat

#setup final arguments
time_d,flux_density,flux_density_err,frequency,bands,redshift,eiso,time_e=data['Time_d'].values,data['Flux_mjy'].values,data['Flux_err_mjy'].values,data['frequency'].values,data['band'].values,data['Redshift'].unique()[0],0,data['Time_e']
afterglow = redback.transient.Afterglow(name='260924A', data_mode='flux_density', time=time_d,flux_density=flux_density, flux_density_err=flux_density_err, frequency=frequency)
model_kwargs = dict(frequency=frequency, output_format='flux_density')
model_kwargs['cosmology']=FlatLambdaCDM(H0=70, Om0=0.3)

#sample
if sample==True:
    if __name__ == "__main__":
        bilby.core.utils.logger.setLevel("ERROR")
        redback.utils.logger.setLevel("ERROR")
        result = redback.fit_model(transient=afterglow, model=wrapped_model, sampler=sampler, model_kwargs=model_kwargs,prior=priors,npool=num_cpus,resume=False,clean=True,plot=False,**kwargs)
        
        #save
        results_dictionary={}
        results_dictionary['Tophat']=result
        with open("260924A_results_object.pkl", "wb") as f:
            dill.dump(results_dictionary, f)
            
#read in
else:
    with open("260924A_results_object.pkl", "rb") as f:
        results_dictionary = dill.load(f)

      
#plot redback default corner result
fig=results_dictionary['Tophat'].plot_corner(save=False,smooth=0.99,title_kwargs={'fontsize': 20},label_kwargs={'fontsize': 22})
plt.show()

#plot redback default light curve result
ax=results_dictionary['Tophat'].plot_lightcurve(model=wrapped_tophat,random_models=100,show=False,linewidth=2,xlim_low_multiplier=0.1,xlim_high_multiplier=10,uncertainty_mode='credible_intervals')
ax.set_xscale('log')
ax.set_ylim(1e-7,1)
plt.show()

#retrieve median values inferred
loge0=results_dictionary['Tophat'].posterior['loge0'].median()
eiso_gamma=np.log10(9.79e54) #https://arxiv.org/pdf/2610.32861
logn0=results_dictionary['Tophat'].posterior['logn0'].median()
results_dictionary['Tophat'].posterior['thv']=results_dictionary['Tophat'].posterior['thvthc']*results_dictionary['Tophat'].posterior['thc']
thc=results_dictionary['Tophat'].posterior['thc'].median()
thv=results_dictionary['Tophat'].posterior['thv'].median()


#Derive expected jet break time
coeff=3.9
z=(1+data['Redshift'].unique()[0])
eiso= 10**loge0/1e53 #this is not gamma-ray this is kinetic energy
eiso=eiso**(1/3)
n0=10**logn0
n0=n0**(-1/3)
theta=(thc+thv)/0.2
theta=theta**(8/3)
tj=coeff*z*eiso*n0*theta
t_j_sec=tj*24*60*60

#get the xmm limit in flux density (same as done for the xrt data)
xmm_limit=0.9e-15 #erg/cm-2/s in the 2-10 keV band https://xmm-tools.cosmos.esa.int/external/xmm_user_support/documentation/uhb/epicsens.html
E,E_2,E_1=1,10,2 #we convert to 1 keV and define a upper and lower values of bandpass
numerator=4.13*(10**11)*(2-data['gamma'].unique()[0])*(np.power(E,1-data['gamma'].unique()[0])) 
denominator=((E_2**(2-data['gamma'].unique()[0]))-(E_1**(2-data['gamma'].unique()[0])))
xmm_limit=xmm_limit*(numerator/denominator) #calculates flux in microJy
xmm_limit=xmm_limit*1e-29 #converting from microJy to erg/s/cm2/Hz
xmm_limit_mjy=xmm_limit*1e26 #convert from erg/s/cm2/hz to mJy


#plot expected jet break time with XMM limit
ax=results_dictionary['Tophat'].plot_lightcurve(model=wrapped_tophat,random_models=100,show=False,linewidth=2,xlim_low_multiplier=0.1,xlim_high_multiplier=10,uncertainty_mode='credible_intervals')
ax.set_xscale('log')
ax.set_ylim(5e-8,1e-1)
ax.axvline(tj,c='red',label='Expected jet break \ntime {} d'.format(round(tj,2)))
ax.axhline(xmm_limit_mjy,c='black',label='Approx. XMM sensitivity limit \n(converted to mJy)')
ax.axvline(55,c='blue',label='Today (nov 18th)')
plt.legend(loc='center left')
plt.show()

#inferred gamma ray efficiency (not geometrically corrected)
eiso_gamma_linear=np.power(10,eiso_gamma)
ezero=np.power(10,loge0)
efficiency=eiso_gamma_linear/(eiso_gamma_linear+ezero)
