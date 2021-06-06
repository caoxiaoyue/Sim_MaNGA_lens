import numpy as np 
import scipy.signal as signal
import os 
from matplotlib import pyplot as plt 

def make_grid_2d(numPix, deltapix, subgrid_res=1):
    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a))
    x_grid = matrix[:, :, 0] * deltapix_eff
    y_grid = matrix[:, :, 1] * deltapix_eff
    shift = np.sum(x_grid) / numPix_eff**2  #find mean x-coord value, minus it
    return x_grid - shift, y_grid - shift

def bin_image(arr, sub_grid=None):
    new_shape = (arr.shape[0]//sub_grid,arr.shape[1]//sub_grid)
    shape = (new_shape[0], sub_grid,
             new_shape[1], sub_grid)
    return arr.reshape(shape).mean(axis=(-1,1)) #the mean operation happen at axis -1 and 1


def auto_mkdir_path(path_dir):
    if not os.path.exists(path_dir):
        abs_path = os.path.abspath(path_dir) 
        os.makedirs(path_dir)

def add_noise_to_image(ideal_image=None,skylevel=0.5,exposure=600,add_noise=False):
    image_with_noise = np.copy(ideal_image)
    image_= image_with_noise + skylevel 
    counts = image_ * exposure  #ideal mean counts
    if add_noise:
        counts = np.random.poisson(counts, image_.shape) #possion counts base on ideal counts
    image_with_noise = counts*1.0/exposure - skylevel #subtract the skylevel in image. this work can be done in pre-processing step
    poisson_noise = np.sqrt(counts)
    poisson_noise /= exposure  #-> counts/s
    return image_with_noise, poisson_noise

def gen_rand_src_pos(mean=0.0,sigma=1.0,size=100,seed=1):
    np.random.seed(seed=seed)
    xpos = np.random.normal(mean, sigma, size=size)
    ypos = np.random.normal(mean, sigma, size=size)
    return xpos,ypos

def gen_rand_pa(low=0.0, high=180.0, size=100, seed=1):
    np.random.seed(seed=seed)
    return np.random.uniform(low=0.0, high=180.0, size=size)

def test_ran_src():
    xsrc,ysrc = gen_rand_src_pos(mean=0.0, sigma=1.0, size=10000)
    plt.figure()
    plt.scatter(xsrc,ysrc,s=0.5)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('square')
    plt.show()

def estimate_thetaE_from_images_pos(ximages, yimages):
    """roughly estimate the einstien radius from images position

    Args:
        ximages (array[float]): images x-coordinates
        yimages (array[float]): images y-coordinates
    """
    r = np.sqrt(ximages**2 + yimages**2)
    return np.mean(r)


def output_images_position_dat(ximages,yimages, ouputfile=None):
    dat_list = list(zip(yimages, ximages)) #ensure the coordiates order is compatible to autolens
    dat_list = str(dat_list)
    with open(ouputfile,'w') as f:
        f.write(dat_list)

def images_from_position_dat(ouputfile=None):
    with open(ouputfile,'r') as f:
        dat_list = f.readline()
    dat_list = eval(dat_list)
    return dat_list

def circular_mask_from_image(mask_radius=0.3, dpix=None, image=None):
    x, y = make_grid_2d(len(image), dpix, subgrid_res=1)
    r = np.sqrt(x**2 + y**2)
    ind = (r < mask_radius)
    return ind


from solve_image_postion.mge_lens import mge_lens_fast
def interpol_alpha_from_map(
    alphax_map,
    alphay_map, 
    xgrid_map,
    ygrid_map,
    eval_xgrid,
    eval_ygrid,
):
    sis_lens_fast = mge_lens_fast(xgrid_map,ygrid_map,alphax_map,alphay_map)
    alphax_interpol = np.zeros_like(eval_xgrid).reshape(-1)
    alphay_interpol = np.zeros_like(eval_ygrid).reshape(-1)
    eval_xgrid_1d = eval_xgrid.reshape(-1)
    eval_ygrid_1d = eval_ygrid.reshape(-1)

    for ii in range(eval_xgrid_1d.size): 
        sis_lens_fast.deflect(eval_xgrid_1d[ii],eval_ygrid_1d[ii])
        alphax_interpol[ii] = sis_lens_fast.deflected_x
        alphay_interpol[ii] = sis_lens_fast.deflected_y

    alphax_interpol = alphax_interpol.reshape(eval_xgrid.shape)
    alphay_interpol = alphay_interpol.reshape(eval_ygrid.shape)
    return alphax_interpol, alphay_interpol


def cut_image(image,shape):
    n1,n2 = image.shape
    c1,c2 = int(n1/2), int(n2/2)
    hw = int(shape[0]/2)
    return image[c1-hw:c1+hw,c2-hw:c2+hw]


from astropy.io import fits
def cut_image_fits_multi_hdu(infile, outfile, shape=None):
    with fits.open(infile) as hdul:
        for ind in range(len(hdul)):
            hdul[ind].data = cut_image(hdul[ind].data, shape)
        hdul.writeto(outfile, overwrite=True)
        
        
def gauss_2d(x, y, xc, yc, r_eff, norm):
    (xnew, ynew) =  (x-xc, y-yc)
    r = np.sqrt(xnew**2 + ynew**2)
    return norm*np.exp(-0.5*(r/r_eff)**2)
