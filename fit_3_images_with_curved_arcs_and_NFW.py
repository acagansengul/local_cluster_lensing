import numpy as np
import matplotlib.pyplot as plt
import warnings
import matplotlib

from lenstronomy.Util import util
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Plots import lens_plot, plot_util
from lenstronomy.Util import simulation_util as sim_util
from lenstronomy.Util import param_util, image_util
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
import lenstronomy.Util.param_util as param_util
from multiprocessing import Pool


from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
import lenstronomy.ImSim.de_lens as de_lens
import dynesty
from dynesty import utils as dyfunc
from dynesty import plotting as dyplot
# Import PySwarms
import pyswarms as ps
import copy
import pickle

from astropy.cosmology import Planck15 as cosmo

nmax1 = 8
nmax2 = 8

cnk = 1

font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 50}
matplotlib.rc('font', **font)

deltaPix = 0.031230659851709842*cnk  #  pixel size in arcsec (area per pixel = deltaPix**2)
exp_time = 1.
sigma_bkg = 1.

array200sci = np.load('smacs_0723/mockimageA1_10_80_16_POSA.npy')
array200err = np.load('smacs_0723/f200w_error1_.npy')

array200sci2 = np.load('smacs_0723/mockimageA2_10_80_16_POSA.npy')
array200err2 = np.load('smacs_0723/f200w_error2_.npy')

array200sci3 = np.load('smacs_0723/mockimageA3_10_80_16_POSA.npy')
array200err3 = np.load('smacs_0723/f200w_error3_.npy')


##############################
for i in range(len(array200err)):
    for j in range(len(array200err)):
        if array200err[i,j] == 0:
            array200err[i,j] = np.inf
        if array200err2[i,j] == 0:
            array200err2[i,j] = np.inf
        if array200err3[i,j] == 0:
            array200err3[i,j] = np.inf
##############################

def rechunk(array2d,nchunk):
    shp = np.shape(array2d)
    shpnew = [int(q/nchunk) for q in shp]
    arraynew = np.zeros(shpnew)
    
    for i in range(shpnew[0]):
        for j in range(shpnew[1]):
            for k in range(nchunk):
                for l in range(nchunk):
                    arraynew[i,j] += array2d[i*nchunk+k,j*nchunk+l]/(nchunk**2.) 
    return arraynew

def rechunkerr(array2d,nchunk):
    shp = np.shape(array2d)
    shpnew = [int(q/nchunk) for q in shp]
    arraynew = np.zeros(shpnew)
    
    for i in range(shpnew[0]):
        for j in range(shpnew[1]):
            acc = 0.
            for k in range(nchunk):
                for l in range(nchunk):
                    acc += (array2d[i*nchunk+k,j*nchunk+l]/(nchunk**2.))**2.
            arraynew[i,j] = np.sqrt(acc)
    return arraynew

arytst = rechunk(array200sci,cnk)
arytst2 = rechunk(array200sci2,cnk)
arytst3 = rechunk(array200sci3,cnk)

arytsterr = rechunkerr(array200err,cnk)
arytsterr2 = rechunkerr(array200err2,cnk)
arytsterr3 = rechunkerr(array200err3,cnk)

numPix2 = int(120/cnk)

kernel_cut = np.load('jwst_psf_200.npy')

psf_type = 'PIXEL'  # 'GAUSSIAN', 'PIXEL', 'NONE'

kwargs_psf = {'psf_type': psf_type, 'pixel_size': deltaPix/cnk,'kernel_point_source': kernel_cut}
#kwargs_psf = sim_util.psf_configure_simple(psf_type=psf_type, fwhm=fwhm, kernelsize=kernel_size, deltaPix=deltaPix, kernel=kernel)

psf_class = PSF(**kwargs_psf)
kwargs_numerics = {'supersampling_factor': 1}

kwargs_data2 = sim_util.data_configure_simple(numPix2, deltaPix, exp_time, sigma_bkg)
data_class2 = ImageData(**kwargs_data2)

def makemask(array,a,b,angle):
    #makes an elliptical mask of a size and angle
    shp = np.shape(array)
    
    likemask = np.zeros(shp,dtype=np.bool)
    for i in range(shp[0]):
        for j in range(shp[1]):
            xprim = np.cos(angle)*(i-shp[0]/2.) + np.sin(angle)*(j-shp[1]/2.)
            yprim = np.cos(angle)*(j-shp[1]/2.) - np.sin(angle)*(i-shp[0]/2.)
            sqrsum = (xprim/a)**2 + (yprim/b)**2. 
            if sqrsum < 1:
                likemask[i,j] = True
    return likemask

def convert_params(mur,mut):
    #take in the tangential and radial magnification
    #return kappa and gamma
    kappa = 1. - 0.5*(1./mur + 1./mut)
    gamma = 0.5*(1./mur - 1./mut)
    return kappa,gamma

likemask = makemask(arytst,35./cnk,85./cnk,1.85)
likemask2 = makemask(arytst2,35/cnk,85./cnk,2.25)
likemask3 = makemask(arytst3,35/cnk,45./cnk,2.25)

def flatten2d(arrays,likemasks):
    #flattens a set of arrays according to likemasks
    flatarr = []
    for i in range(len(likemasks)):
        for p in range(len(likemasks[i])):
            for q in range(len(likemasks[i])):
                if likemasks[i][p,q]:
                    flatarr.append(arrays[i][p,q])  
    return flatarr

def unflatten(flatimg,likemasks):
    arrays = np.zeros([len(likemasks),len(likemasks[0]),len(likemasks[0])])
    k = 0
    for i in range(len(likemasks)):
        for p in range(len(likemasks[i])):
            for q in range(len(likemasks[i])):
                if likemasks[i][p,q]:
                    arrays[i,p,q] = flatimg[k]
                    k+=1
    return arrays

imagearr = np.array([arytst3,arytst2,arytst])
noises = np.array([arytsterr3,arytsterr2,arytsterr])
likemasks = np.array([likemask3,likemask2,likemask])
likearr = likemasks

flatarr = flatten2d(imagearr,likemasks)
flaterror = flatten2d(noises,likemasks)

indices = [0,1,2]

for i in range(len(flaterror)):
    if flaterror[i] == 0:
        flaterror[i] = np.inf

unflatimg = unflatten(flatarr,likemasks)

def model_curved_fit_shapelet_sers(data,kappashear_params,source_params,likemask_list,indices):   
    A_list = []
    C_D_response_list = []
    d_list = []
    for i in indices:
        if i == 1:
            lens_model_list_new = ['SHIFT','CURVED_ARC_SIS_MST','NFW']
        else:
            lens_model_list_new = ['SHIFT','CURVED_ARC_SIS_MST']
        kwargs_lens_true_new = kappashear_params[i]

        lens_model_class = LensModel(lens_model_list=lens_model_list_new)

        source_model_list = ['SHAPELETS','SHAPELETS']
        source_model_class = LightModel(light_model_list=source_model_list)
        
        lensLightModel_reconstruct = LightModel([])

        data_class2.update_data(data[i])
        
        imageModel = ImageLinearFit(data_class=data_class2, psf_class=psf_class, kwargs_numerics=kwargs_numerics, 
                                lens_model_class=lens_model_class, source_model_class=source_model_class,
                                lens_light_model_class = lensLightModel_reconstruct,likelihood_mask=likemask_list[i])
        
        
        A = imageModel._linear_response_matrix(kwargs_lens_true_new, source_params, kwargs_lens_light=[], kwargs_ps=None)
        C_D_response, model_error = imageModel._error_response(kwargs_lens_true_new, kwargs_ps=None,kwargs_special=None)
        d = imageModel.data_response
        
        A_list.append(A)
        
        Ashp = np.shape(A)
        
        C_D_response_list.append(C_D_response)
        d_list.append(d)
    
    Ashp = np.shape(A)
        
    
    Atot = np.concatenate((A_list),axis=1)
    Ctot = np.concatenate((C_D_response_list))
    Dtot = np.concatenate((d_list))
    
    param, cov_param, wls_model2 = de_lens.get_param_WLS(Atot.T, 1./Ctot, Dtot, inv_bool=False)
    
    return wls_model2,param

ndeg = np.count_nonzero(likemasks)

maxlens1 = [3.,1.,1.]
minlens1 = [1.,-1.,0.]

maxlens2 = [3.,-3.,1.,1.]
minlens2 = [0.,-20.,-1.,0.]

maxlens3 = [10.,3.,1.,1.]
minlens3 = [1., 0.,-1.,0.]

maxshift = [5e-2,5e-2]
minshift = [-5e-2,-5e-2]

maxshape = [0.7,2e-1,2e-1]
minshape = [0.2,-2e-1,-2e-1]

maxshape2 = [0.2,2e-1,2e-1]
minshape2 = [0.0,-2e-1,-2e-1]


xpp = 0.02995819
ypp = 0.0234981


maxnfw = [200.,100.,xpp+0.3,ypp+0.3]
minnfw = [0.,1.,xpp-0.3,ypp-0.3]

maxparam = maxlens1 + maxlens2 + maxshift + maxlens3 + maxshift + maxshape + maxshape2 + maxnfw
minparam = minlens1 + minlens2 + minshift + minlens3 + minshift + minshape + minshape2 + minnfw

npar = len(minparam)

def lnlike(params):      

    mur1, mut1 = 1.,params[0]
    mur2, mut2 = params[3],params[4]
    mur3, mut3 = params[9],params[10]
    
    cv1 = np.abs(params[1])
    cv2 = np.abs(params[5])
    cv3 = np.abs(params[11])

    psi_ext1 = params[2]*np.pi - (np.sign(params[1])+1.)*np.pi/2.
    psi_ext2 = params[6]*np.pi - (np.sign(params[5])+1.)*np.pi/2.
    psi_ext3 = params[12]*np.pi - (np.sign(params[11])+1.)*np.pi/2.
    
    # lensing quantities
    Mpert = params[21]*1e7
    Cpert = params[22]

    z_s = 1.449
    z_l = 0.3877

    lens_cosmo = LensCosmo(z_lens=z_l, z_source=z_s, cosmo=cosmo)
    Rs_angle, alpha_Rs = lens_cosmo.nfw_physical2angle(M=Mpert, c=Cpert)

    kwargs_kapshe = [[{'alpha_x':0.,'alpha_y':0.},
                      {'tangential_stretch': mut1, 'radial_stretch': mur1, 'curvature': cv1, 'direction': psi_ext1, 
                       'center_x': 0., 'center_y': 0.}],
                     [{'alpha_x':params[7],'alpha_y':params[8]},
                      {'tangential_stretch': mut2, 'radial_stretch': mur2, 'curvature': cv2, 'direction': psi_ext2, 
                       'center_x': 0., 'center_y': 0.},
                       {'Rs':Rs_angle, 'alpha_Rs':alpha_Rs, 'center_x':params[24], 'center_y':-params[23]}],
                     [{'alpha_x':params[13],'alpha_y':params[14]},
                      {'tangential_stretch': mut3, 'radial_stretch': mur3, 'curvature': cv3, 'direction': psi_ext3, 
                       'center_x': 0., 'center_y': 0.}]]

    source_shape = [{'n_max': nmax1, 'beta': params[15]/np.power(nmax1+1,0.5), 'center_x': params[16], 'center_y': params[17]},
                    {'n_max': nmax2, 'beta': params[18]/np.power(nmax2+1,0.5), 'center_x': params[19], 'center_y': params[20]}]

    fit,paramq = model_curved_fit_shapelet_sers(imagearr,kwargs_kapshe,source_shape,likearr,indices)

    return -0.5*np.sum((((fit-flatarr)/flaterror)**2.))


# Define our uniform prior.
def ptform(u):
    """Transforms samples `u` drawn from the unit cube to samples to those
    from our uniform prior within [-10., 10.) for each variable."""
    return minparam*(1.-u) + maxparam*u
nparal = 32
pool = Pool(nparal)

# "Dynamic" nested sampling.
dsampler = dynesty.DynamicNestedSampler(lnlike, ptform, npar, pool=pool, queue_size=nparal)
dsampler.run_nested()
dresults = dsampler.results

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
            
save_obj(dresults,'POINT_MASS_NFW_FIT_POSA_with_10_8_PERTURBED_FIXmur=1_3_images_SHAPE_SHAPE' + 'nmax1' + str(nmax1) + 'nmax2' + str(nmax2))