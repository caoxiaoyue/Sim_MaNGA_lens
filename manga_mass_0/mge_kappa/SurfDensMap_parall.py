import os
import re
import subprocess
import time

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits
from astropy.table import Table

import util_dm
import util_mge
from multiprocessing import Pool

def make_grid_2d(numPix, deltapix, subgrid_res=1):
    """
    creates pixel grid (in 2d arrays of x- and y- positions)
    default coordinate frame is such that (0,0) is in the center of the coordinate grid

    :param numPix: number of pixels per axis
    :param deltapix: pixel size
    :param subgrid_res: sub-pixel resolution (default=1)
    :return: x, y position information in two 1d arrays
    """

    numPix_eff = numPix*subgrid_res
    deltapix_eff = deltapix/float(subgrid_res)
    a = np.arange(numPix_eff)
    matrix = np.dstack(np.meshgrid(a, a))
    x_grid = matrix[:, :, 0] * deltapix_eff
    y_grid = matrix[:, :, 1] * deltapix_eff
    
    shift = np.sum(x_grid) / numPix_eff**2  #find mean x-coord value, minus it

    return x_grid - shift, y_grid - shift

def SurfDens(x,y,mge2d,dist,ml,cosinc,logrho_s,rs,gamma):
    # For oblate, x is the major axis while y is the minor axis
    # x,y in unit [pc]
    LmMge = util_mge.mge(mge2d,inc=np.arccos(cosinc),shape='oblate',dist=dist)
    Surf_stellar = LmMge.surfaceBrightness(x,y)*ml  #M_solar/pc^2
    DhMge = util_dm.gnfw1d(10**logrho_s,rs,gamma)
    Surf_dark = DhMge.surfaceDensity(np.sqrt(x**2+y**2)/1e3)/1e6  #M_solar/pc^2
    Surf_total = Surf_stellar+Surf_dark
    return Surf_total,Surf_stellar,Surf_dark

def cal_critical_surface_density(z_l,z_s):
    v_light = const.c
    grav_const = const.G 
    Ds = cosmo.angular_diameter_distance(z_s)
    Dd = cosmo.angular_diameter_distance(z_l)
    Dds = cosmo.angular_diameter_distance_z1z2(z_l,z_s)

    crit_surface_density = v_light**2 / \
                    (4*np.pi*grav_const) * \
                    Ds/Dd/Dds
    output_unit = u.solMass / (u.pc*u.pc)
    return crit_surface_density.to(output_unit)  #critical density in unit of M_solar/pc^2

def return_file_list_info(folder='./mge_fit_res/'):
    dir1 = os.getcwd()
    os.chdir(folder)
    output = subprocess.Popen(['ls','-l', ],stdout=subprocess.PIPE,shell=True)
    lines = output.stdout.read().splitlines()
    lines = [line.decode("utf-8") for line in lines]
    index = []
    plate_info = []
    for line in lines:
        if 'npy' not in line: continue
        output = re.search(r"mge_(\d+)_(.*).npy", line)
        index.append(output.group(1))
        plate_info.append(output.group(2))
    dir2 = os.chdir(dir1)
    return index,plate_info

def generate_kappa(
    gal_index, 
    gal_plateifu,
    shape=[300,300], 
    dpix=0.05, 
    z_l=0.2,
    z_s=0.6, 
    return_kappa=True, 
    return_kpc_size=False,
    folder='./mge_fit_res/'
    ):
    '''
    Purpose: this function output the `lens' convergence map based on the dynamics modeling results of MaNGA Early type galaxies
    gal_index and gal_plateifu: the identifier of MaNGA dynamics reconstruction results 
    shape: Npixel * Npixel, the image shape
    dpix: pixel size in arcsec 
    z_l: the redshift of lens which you want to simulate
    z_s: the source redshift
    '''
    #load luminosity mge
    sol = np.load(f'{folder}mge_{gal_index}_{gal_plateifu}.npy',allow_pickle=True, encoding='bytes')
    mge2d = sol[0]

    #load JAM result
    hdu=fits.open('cat_JAM_SPS_MPL7.fits')
    #hdu[1].header
    index = hdu[1].data['Index']  #automatically loaded with int data-type
    plateifu = hdu[1].data['plateIFU']
    dist = hdu[1].data['Dist_Mpc']
    cosinc = hdu[1].data['cosinc']
    ml = hdu[1].data['ml']
    logrho_s = hdu[1].data['logrho_s']
    rs = hdu[1].data['rs']
    gamma = hdu[1].data['gamma']
    z_objs = hdu[1].data['z']
    vdisp = hdu[1].data['SigmaRe']
    idd = np.where(index==gal_index)
    z_l_origin = z_objs[idd][0]  #the original redshift of manga galaxy, scale to higher redshift z_l

    D_ang = cosmo.angular_diameter_distance(z_l).value * 1e6 #pc units
    factor = 180.0/np.pi*3600 / D_ang #factor to change pc to arcsec
    pixsize = dpix/factor  #lens-plane pixel size in physical pc scale
    # print('----pixel size in pc', pixsize)
    # print('-------velocity dispersion within RE',hdu[1].data['SigmaRe'][idd][0])

    crit_density = cal_critical_surface_density(z_l,z_s).value

    xpos, ypos = make_grid_2d(shape[0],deltapix=pixsize)  #grid in physical scale [pc]
    surf_total = np.zeros_like(xpos)
    surf_stellar = np.zeros_like(xpos)
    surf_dark = np.zeros_like(xpos)
    ind = index == gal_index
    for i in range(shape[0]):
        for j in range(shape[1]):
            surf_total[i,j],\
            surf_stellar[i,j],\
            surf_dark[i,j] = SurfDens(xpos[i,j],ypos[i,j],mge2d,dist[ind],
                                      ml[ind],cosinc[ind],logrho_s[ind],rs[ind],gamma[ind])
    
    if return_kappa:
        surf_total = surf_total/crit_density
        surf_stellar = surf_stellar/crit_density
        surf_dark = surf_dark/crit_density

    if return_kpc_size:
        ext = (np.min(xpos)/1000,np.max(xpos)/1000,np.min(ypos)/1000,np.max(ypos)/1000)
        size_units = '[kpc]'
    else:
        xpos, ypos = make_grid_2d(shape[0],deltapix=dpix)  #grid in arcsec unit
        ext = (np.min(xpos),np.max(xpos),np.min(ypos),np.max(ypos))
        size_units = '[arcsec]'
    # plot figures
    fig=plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(surf_total,origin='lower',extent=ext)
    ax1.set_xlabel(size_units)
    ax1.set_ylabel(size_units)
    cbar1 = plt.colorbar(im1)
    #cbar1.set_label(r'$M_{\odot}pc^{-2}$')
    ax1.set_title('Total surface density')
    
    ax2 = fig.add_subplot(132)
    im2 = ax2.imshow(surf_stellar,origin='lower',extent=ext)
    ax2.set_xlabel(size_units)
    ax2.set_ylabel(size_units)
    cbar2 = plt.colorbar(im2)
    #cbar2.set_label(r'$M_{\odot}pc^{-2}$')
    ax2.set_title('Stellar surface density')
    
    ax3 = fig.add_subplot(133)
    im3 = ax3.imshow(surf_dark,origin='lower',extent=ext)
    ax3.set_xlabel(size_units)
    ax3.set_ylabel(size_units)
    cbar3 = plt.colorbar(im3)
    #cbar3.set_label(r'$M_{\odot}pc^{-2}$')
    ax3.set_title('Dark surface density')
    plt.tight_layout()
    #plt.show()
    fig.savefig('./kappa_png/SurfaceDensity_%s_%s.png'%(gal_index,gal_plateifu))
    return surf_total, surf_stellar, surf_dark

if __name__ == "__main__":
    #load mge-fiting results files name
    index_list,plate_info_list = return_file_list_info()
    index_list = index_list[0:]
    plate_info_list = plate_info_list[0:]
    
    #the overall-mge-fitting ccatalog infomation
    tab = Table.read('./cat_JAM_SPS_MPL7.fits')
    grade = tab['Flag_classA']
    vdisp = tab['SigmaRe']
    index = tab['Index']

    #run the kappa-map generation in parallel!!!
    nsubgrid = 2
    dpix = 0.05 # arcsec per pix
    gal_index  = list(map(int,index_list))  #convert the index from string to int type
    gal_plateifu = plate_info_list[:]

    par_list= []
    for ii in range(len(gal_index)):
        par = [
            gal_index[ii], 
            gal_plateifu[ii],
            (800*nsubgrid,800*nsubgrid),
            dpix/nsubgrid
        ]
        par_list.append(par)

    pool = Pool(processes=5)
    t1= time.time()
    results = pool.starmap(generate_kappa,par_list[0:4])   
    #here, we only generate 4 kappa map as an example; change to par_list[:] to generate all samples
    t2= time.time()
    print('total time elapse:',t2-t1)   #take about 3 hour on my local PC

    #save the results into fits file
    for jj,item in enumerate(results):
        surf_total, surf_stellar, surf_dark = item
        primary_hdu = fits.PrimaryHDU(surf_total)
        primary_hdu.header['dpix'] = dpix/nsubgrid
        primary_hdu.header['nsub'] = nsubgrid
        image_hdu = fits.ImageHDU(surf_stellar)
        image_hdu2 = fits.ImageHDU(surf_dark)
        hdul = fits.HDUList([primary_hdu, image_hdu, image_hdu2])
        hdul.writeto(f'./kappa/{gal_index[jj]}_{gal_plateifu[jj]}.fits',overwrite=True)
