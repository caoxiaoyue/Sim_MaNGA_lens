import numpy as np
import scipy.signal as signal
from lenstronomy.LensModel.convergence_integrals import (
    potential_from_kappa_grid, potential_from_kappa_grid_adaptive,
    deflection_from_kappa_grid_adaptive
    )

# lens-text book: p37, eq 53
# potential == thetaE*|theta|


def sis_kappa(x, y, xc=0.0, yc=0.0, r_sis=1.0):
    xnew = x - xc
    ynew = y - yc
    r = np.sqrt(xnew**2 + ynew**2)
    return r_sis/(r+(r == 0))/2.0


def sis_defl(x, y, xc=0.0, yc=0.0, r_sis=1.0):
    xnew = x - xc
    ynew = y - yc
    r = np.sqrt(xnew**2 + ynew**2)
    alpha_x = xnew/(r+(r == 0))
    alpha_y = ynew/(r+(r == 0))
    return r_sis*alpha_x, r_sis*alpha_y


def sis_potential(x, y, xc=0.0, yc=0.0, r_sis=1.0):
    xnew = x - xc
    ynew = y - yc
    r = np.sqrt(xnew**2 + ynew**2)
    return r_sis*r


def psi_from_kappa(kappa, dpix):
    return potential_from_kappa_grid(kappa, dpix)


def psi_from_kappa_adp(kappa_high_res, dpix_high_res, bining_size):
    kernel_size = kernel_size_from_kappa(kappa_high_res, bining_size)
    psi = potential_from_kappa_grid_adaptive(
        kappa_high_res,
        grid_spacing=dpix_high_res,
        low_res_factor=bining_size,
        high_res_kernel_size=kernel_size,
    )
    return psi

def alpha_from_kappa_adp(kappa_high_res, dpix_high_res, bining_size):
    kernel_size = kernel_size_from_kappa(kappa_high_res, bining_size)
    alphax, alphay = deflection_from_kappa_grid_adaptive(
        kappa_high_res,
        grid_spacing=dpix_high_res,
        low_res_factor=bining_size,
        high_res_kernel_size=kernel_size,
    )
    return alphax, alphay

def kernel_size_from_kappa(kappa_high_res, bining_size):
    """
    size of high resolution kernel in units of degraded pixels
    2 times of the kappa_high_res size
    """
    kernel_size = int(len(kappa_high_res)*2/bining_size + 1)
    if not isinstance(kernel_size, int):
        raise Exception("kernel size must be and odd integer",kernel_size)
    return kernel_size
