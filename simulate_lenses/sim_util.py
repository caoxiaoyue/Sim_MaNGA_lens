import numpy as np 
import scipy.signal as signal
import os 
from matplotlib import pyplot as plt 
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from astropy.io import fits


class pix_lens(object):
    def __init__(self, xgrid, ygrid, alphax, alphay):
        """Initialize pixelized mass model, wihch delfection angle is evaluated based on a 
        set of precompute delfection angle map

        Args:
            xgrid ([array]): xgrid of precomputed delfection angle map
            ygrid ([array]): ygrid of precomputed delfection angle map
            alphax ([array]): x-delfectoin map
            alphay ([array]): y-delfectoin map
        """
        self.xgrid_1d = xgrid.reshape(-1)
        self.ygrid_1d = ygrid.reshape(-1)
        self.alphax_1d = alphax.reshape(-1)
        self.alphay_1d = alphay.reshape(-1)

        self.tri = Delaunay(list(zip(self.xgrid_1d, self.ygrid_1d)))

    def eval_alphax(self, x ,y):
        interpol = griddata(
            self.tri, 
            self.alphax_1d, 
            (x , y), 
            method='linear',
            fill_value=0.0
        ) 
        return interpol

    def eval_alphay(self, x ,y):
        interpol = griddata(
            self.tri, 
            self.alphay_1d, 
            (x , y), 
            method='linear',
            fill_value=0.0
        ) 
        return interpol

    def deflect(self, x,y):
        self.alpha_x = self.eval_alphax(x,y)
        self.alpha_y = self.eval_alphay(x,y)
        
    def ray_shoot(self,x, y):
        self.deflect(x,y)
        return x - self.alpha_x, y - self.alpha_y


def make_grid_2d(numPix, dpix, nsub=1):
    npix_eff = numPix*nsub
    dpix_eff = dpix/float(nsub)
    coord_1d = np.arange(npix_eff) * dpix_eff
    coord_1d -= np.mean(coord_1d)
    return np.meshgrid(coord_1d, coord_1d)

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
    x, y = make_grid_2d(len(image), dpix, nsub=1)
    r = np.sqrt(x**2 + y**2)
    ind = (r < mask_radius)
    return ind


# def interpol_alpha_from_map(
#     alphax_map,
#     alphay_map, 
#     xgrid_map,
#     ygrid_map,
#     eval_xgrid,
#     eval_ygrid,
# ):
#     this_lens = pix_lens(xgrid_map,ygrid_map,alphax_map,alphay_map)
#     alphax_interpol = np.zeros_like(eval_xgrid).reshape(-1)
#     alphay_interpol = np.zeros_like(eval_ygrid).reshape(-1)
#     eval_xgrid_1d = eval_xgrid.reshape(-1)
#     eval_ygrid_1d = eval_ygrid.reshape(-1)

#     for ii in range(eval_xgrid_1d.size): 
#         this_lens.deflect(eval_xgrid_1d[ii],eval_ygrid_1d[ii])
#         alphax_interpol[ii] = this_lens.deflected_x
#         alphay_interpol[ii] = this_lens.deflected_y

#     alphax_interpol = alphax_interpol.reshape(eval_xgrid.shape)
#     alphay_interpol = alphay_interpol.reshape(eval_ygrid.shape)
#     return alphax_interpol, alphay_interpol

def cut_image(image,shape):
    n1,n2 = image.shape
    c1,c2 = int(n1/2), int(n2/2)
    hw = int(shape[0]/2)
    return image[c1-hw:c1+hw,c2-hw:c2+hw]


        
def gauss_2d(x, y, xc, yc, r_eff, norm):
    (xnew, ynew) =  (x-xc, y-yc)
    r = np.sqrt(xnew**2 + ynew**2)
    return norm*np.exp(-0.5*(r/r_eff)**2)
