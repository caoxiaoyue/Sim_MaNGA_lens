#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : JAM/utils/util_mge.py
# Author            : Hongyu Li <lhy88562189@gmail.com>
# Date              : 10.09.2017
# Last Modified Date: 11.09.2017
# Last Modified By  : Hongyu Li <lhy88562189@gmail.com>
import numpy as np
from scipy.integrate import quad, dblquad
from scipy.misc import derivative
from scipy import special

TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
SQRT_TOW_PI = np.sqrt(2.0*np.pi)


def _inte_2dencloseM(r, theta,
                     surf, twoSigma2, q2):
    cosTheta2 = np.cos(theta)**2
    rst = 0.0
    for i in range(len(surf)):
        rst += surf[i] * np.exp(- r*r / twoSigma2[i] *
                                (cosTheta2 + (1-cosTheta2)/q2[i])) * r
    return rst


def _inte_3dencloseM(r, u,
                     dens, twoSigma2, q2):
    rst = 0.0
    for i in range(len(twoSigma2)):
        rst += dens[i] * np.exp(- r*r / twoSigma2[i] *
                                ((1 - u*u) + u*u / q2[i])) * r*r
    return FOUR_PI * rst


def _Hu_inte(u,
             R, z,
             M, sigma, q2):
    rst = 0.0
    twoSigma2 = 2.0 * sigma * sigma
    for i in range(len(M)):
        rst += M[i] * np.exp(-u*u / twoSigma2[i] *
                             (R*R + z*z / (1.0 - (1.0 - q2[i]) * u*u))) /\
            np.sqrt(1.0 - (1.0 - q2[i]) * u*u) / sigma[i]
    return rst


def _Re(mge2d, lower=None, upper=None):
    '''
    calculate the effective radius using Cappellari 2013 MNRAS 432,1709
      Equation (11)
    '''
    if lower is None:
        lower = 0.3 * mge2d[:, 1].min()
    if upper is None:
        upper = 3.0 * mge2d[:, 1].max()
    L = 2*np.pi * mge2d[:, 0] * mge2d[:, 1]**2 * mge2d[:, 2]
    R = np.logspace(np.log10(lower), np.log10(upper), 5000)
    enclosedL = R.copy()
    for i in range(R.size):
        enclosedL[i] = \
            np.sum(L * (1-np.exp(-(R[i])**2 / (2.0 * mge2d[:, 1]**2 *
                                               mge2d[:, 2]))))
    halfL = np.sum(L) / 2.0
    ii = np.abs(enclosedL-halfL) == np.min(np.abs(enclosedL-halfL))
    return np.mean(R[ii])


def _enclosed3D(mge3d, r):
    lum = mge3d[:, 0] * (SQRT_TOW_PI*mge3d[:, 1])**3 * mge3d[:, 2]
    e = np.sqrt(1. - mge3d[:, 2]**2)
    h = r[:, np.newaxis]/(np.sqrt(2.)*mge3d[:, 1]*mge3d[:, 2])
    mass_r = np.sum(lum*(special.erf(h) - np.exp(-(h*mge3d[:, 2])**2) *
                         special.erf(h*e)/e), axis=1)
    return mass_r


def _r_half_cir(mge2d, lower=None, upper=None):
    '''
    calculate the 3D half light radius using Cappellari 2013 MNRAS 432,1709
      Equation (15)
    '''
    if lower is None:
        lower = 0.3 * mge2d[:, 1].min()
    if upper is None:
        upper = 3.0 * mge2d[:, 1].max()
    L = 2*np.pi * mge2d[:, 0] * mge2d[:, 1]**2 * mge2d[:, 2]
    r = np.logspace(np.log10(lower), np.log10(upper), 5000)
    enclosedL = r.copy()
    scale = (2**0.5 * mge2d[:, 1] * mge2d[:, 2]**(1.0/3.0))
    for i in range(r.size):
        h = r[i] / scale
        enclosedL[i] = \
            np.sum(L * (special.erf(h) - 2.0*h*np.exp(-h**2)/np.pi**0.5))
    halfL = np.sum(L) / 2.0
    ii = np.abs(enclosedL-halfL) == np.min(np.abs(enclosedL-halfL))
    return np.mean(r[ii])


def _r_half(mge3d, lower=None, upper=None):
    '''
    calculate the 3D half light radius using Cappellari 2013 MNRAS 432,1709
      Equation (15)
    '''
    if lower is None:
        lower = 0.3 * mge3d[:, 1].min()
    if upper is None:
        upper = 3.0 * mge3d[:, 1].max()
    halfL = np.sum(mge3d[:, 0] * (SQRT_TOW_PI*mge3d[:, 1])**3 * mge3d[:, 2])/2
    r = np.logspace(np.log10(lower), np.log10(upper), 5000)
    enclosedL = _enclosed3D(mge3d, r)
    ii = np.abs(enclosedL-halfL) == np.min(np.abs(enclosedL-halfL))
    return np.mean(r[ii])


def projection(mge3d, inc, shape='oblate'):
    '''
    Convert 3d mges to 2d mges
    inc in radians
    '''
    dens = mge3d[:, 0]
    sigma3d = mge3d[:, 1]
    qint = mge3d[:, 2]
    mge2d = mge3d.copy()
    if shape == 'oblate':
        qobs = np.sqrt(qint**2 * np.sin(inc)**2 + np.cos(inc)**2)
        surf = dens * qint / qobs * (SQRT_TOW_PI * sigma3d)
        mge2d[:, 0] = surf
        mge2d[:, 2] = qobs
    elif shape == 'prolate':
        qobs = np.sqrt(1.0/(qint**2 * np.sin(inc)**2 + np.cos(inc)**2))
        sigma2d = sigma3d/qobs
        surf = (SQRT_TOW_PI * sigma2d * qobs**2 * qint) * dens
        mge2d[:, 0] = surf
        mge2d[:, 1] = sigma2d
        mge2d[:, 2] = qobs
    else:
        raise ValueError('shape {} not supported'.format(shape))
    return mge2d


def ml_gradient_gaussian(sigma, delta, ml0=1.0, lower=0.4):
    '''
    Create a M*L gradient
    sigma: Gaussian sigma in [Re]
    delta: Gradient value
    ml0: Central stellar mass to light ratio
    lower: the ratio between the central and the outer most M*/L
    '''
    sigma = np.atleast_1d(sigma)
    sigma = sigma - sigma[0]
    ML = ml0 * (lower + (1-lower)*np.exp(-0.5 * (sigma * delta)**2))
    return ML


def expsin(x):
    yfunctuion = lambda t: np.exp(-x*np.sin(t)**2.)
    return 4.*quad(yfunctuion, 0., np.pi/2.)[0]

def secexpsin(radius, sigma_j, para_j):
    secyfunction = lambda t: sigma_j*np.sin(t)**2.*para_j*(np.exp(-radius**2./(2.*sigma_j)*(1+para_j*np.sin(t)**2.))-1)/(1.+(para_j*np.sin(t)**2.))
    return 4.*quad(secyfunction, 0., np.pi/2.)[0]

def enclosed2Dluminosity_formula(mge2d, r, dh2D_mge=None, dist=None, ml=1.0):
    kpt = mge2d.copy()
    if dist is not None:
        pcc = dist * np.pi / 0.648
        kpt[:, 1] *= pcc
    kpt[:, 0] = kpt[:, 0] * ml
    if dh2D_mge is not None:
        kkt = dh2D_mge.copy()
        kpt = np.append(kpt, kkt, axis=0)
    surf = kpt[:, 0]
    twoSigma2_j = 2.0*kpt[:, 1]**2.
    q2_paraj = (1 - kpt[:, 2]**2.)/kpt[:, 2]**2.
    mass_j = 0.
    for index_j in range(np.size(surf)):
        mass_j += surf[index_j]*(twoSigma2_j[index_j]/2.*\
                      (2.*np.pi-np.exp(-r**2./twoSigma2_j[index_j])*expsin(r**2./twoSigma2_j[index_j]*q2_paraj[index_j])) \
                      + secexpsin(r, twoSigma2_j[index_j]/2., q2_paraj[index_j]))
    return mass_j
    
def density_2D_formula(mge2d, r, dh2D_mge=None, dist=None, ml=1.0):
    kpt = mge2d.copy()
    if dist is not None:
        pcc = dist * np.pi / 0.648
        kpt[:, 1] *= pcc
    kpt[:, 0] = kpt[:, 0] * ml
    if dh2D_mge is not None:
        kkt = dh2D_mge.copy()
        kpt = np.append(kpt, kkt, axis=0)
    surf = kpt[:, 0]
    twoSigma2_j = 2.0*kpt[:, 1]**2.
    q2_paraj = (1 - kpt[:, 2]**2.)/kpt[:, 2]**2.
    dens_hh = 0.
    for i in range(np.size(surf)):
        dens_hh += surf[i]/(2.*np.pi)*np.exp(-r**2./twoSigma2_j[i])*expsin(r**2./twoSigma2_j[i]*q2_paraj[i])
    return dens_hh


class mge:
    '''
    The default units are [L_solar/pc^2]  [pc]  [none]
    inc in radians
    All the length unit is pc, 2d density is in [L_solar/pc^2],
      3d density is in [L_solar/pc^3]
    '''
    def __init__(self, mge2d, inc, shape='oblate', dist=None):
        self.mge2d = mge2d.copy()  #array,  [5,3] here, 5 gauss mge, each gauss 3 par
        #[density,sigma,q]
        if dist is not None:
            pc = dist * np.pi / 0.648  #????
            self.mge2d[:, 1] *= pc
        self.inc = inc
        self.shape = shape
        self.ngauss = mge2d.shape[0]

    def deprojection(self):
        '''
        Return the 3D deprojected MGE coefficients
        '''
        mge3d = np.zeros_like(self.mge2d)
        if self.shape == 'oblate':
            qintr = self.mge2d[:, 2]**2 - np.cos(self.inc)**2
            if np.any(qintr <= 0):
                raise RuntimeError('Inclination too low q < 0')
            qintr = np.sqrt(qintr)/np.sin(self.inc)
            if np.any(qintr < 0.05):
                raise RuntimeError('q < 0.05 components')
            dens = self.mge2d[:, 0]*self.mge2d[:, 2] /\
                (self.mge2d[:, 1]*qintr*SQRT_TOW_PI)
            mge3d[:, 0] = dens
            mge3d[:, 1] = self.mge2d[:, 1]
            mge3d[:, 2] = qintr
        elif self.shape == 'prolate':
            qintr = np.sqrt(1.0/self.mge2d[:, 2]**2 -
                            np.cos(self.inc)**2)/np.sin(self.inc)
            if np.any(qintr > 10):
                raise RuntimeError('q > 10.0 conponents')
            sigmaintr = self.mge2d[:, 1]*self.mge2d[:, 2]
            dens = self.mge2d[:, 0] / (SQRT_TOW_PI*self.mge2d[:, 1] *
                                       self.mge2d[:, 2]**2*qintr)
            mge3d[:, 0] = dens
            mge3d[:, 1] = sigmaintr
            mge3d[:, 2] = qintr
        return mge3d

    def luminosity(self):
        '''
        Return the total luminosity of all the Gaussians (in L_solar)
        '''
        return (self.mge2d[:, 0]*TWO_PI*self.mge2d[:, 1]**2 *
                self.mge2d[:, 2]).sum()

    def luminosityDensity(self, R, z):
        '''
        Return the luminosity density at coordinate R, z (in L_solar/pc^3)
        '''
        rst = 0.0
        mge3d = self.deprojection()
        for i in range(self.ngauss):
            rst += mge3d[i, 0] * np.exp(-0.5/mge3d[i, 1]**2 *
                                        (R**2 + (z/mge3d[i, 2])**2))
        return rst

    def meanDensity_inte(self, r):
        '''
        Return the mean density at give spherical radius r
        r in pc, density in L_solar/pc^3
        meanDensity is recommended to ues
        '''
        r = np.atleast_1d(r)[:, np.newaxis]
        cosTheta = np.linspace(0.0, 1.0, 500)
        sinTheta = np.sqrt(1 - cosTheta**2)
        R = r * sinTheta
        z = r * cosTheta
        density = self.luminosityDensity(R, z)
        return np.average(density, axis=1)

    def meanDensity(self, r):
        '''
        Return the mean density at give spherical radius r
        r in pc, density in L_solar/pc^3
        See Cappellari 2015, APJ 804,21
        '''
        r = np.atleast_1d(r)[:, np.newaxis]
        mge2d = self.mge2d
        mge3d = self.deprojection()
        sigma = mge3d[:, 1]
        q = mge3d[:, 2]
        lum = mge2d[:, 0]*TWO_PI*sigma**2 * mge2d[:, 2]
        rho = np.sum(lum * np.exp(-r**2/(2*sigma**2)) *
                     special.erf(r*np.sqrt(1-q**2)/(q*sigma*np.sqrt(2.0))) /
                     (FOUR_PI*sigma**2*r*np.sqrt(1-q**2)), axis=1)
        return rho

    def surfaceBrightness(self, x, y):
        '''
        Return the surface brightness at coordinate x, y (in L_solar/pc^2)
        x, y must have the same unit as mge2d (i.e. default as pc)
        x is the major axis for oblate and minor axis for prolate
        '''
        rst = 0.0
        if self.shape == 'prolate':
            for i in range(self.ngauss):
                rst += self.mge2d[i, 0] * \
                    np.exp(-0.5/self.mge2d[i, 1]**2 *
                           (y**2+(x/self.mge2d[i, 2])**2))
        elif self.shape == 'oblate':
            for i in range(self.ngauss):
                rst += self.mge2d[i, 0] * \
                    np.exp(-0.5/self.mge2d[i, 1]**2 *
                           (x**2+(y/self.mge2d[i, 2])**2))
        return rst

    def enclosed3Dluminosity_inte(self, r):
        '''
        Return the 3D enclosed luminosity within a sphere r (in L_solar)
        input r should be in pc
        enclosed3Dluminosity is recommended to use
        '''
        mge3d = self.deprojection()
        dens = mge3d[:, 0]
        twoSigma2 = 2.0 * mge3d[:, 1]**2
        q2 = mge3d[:, 2]**2
        I = dblquad(_inte_3dencloseM, 0.0, 1.0, lambda x: 0.0,
                    lambda x: r, args=(dens, twoSigma2, q2))
        return I[0]

    def enclosed3Dluminosity(self, r):
        '''
        Return the 3D enclosed luminosity within a sphere r (in L_solar)
        input r should be in pc
        see Mitzkus 2017, MNRAS 464,4789
        '''
        r = np.atleast_1d(r)
        mge3d = self.deprojection()
        mass_r = _enclosed3D(mge3d, r)
        return mass_r

    def enclosed2Dluminosity(self, R):
        '''
        Return the 2D enclosed luminosity within a circular aperture R
          (in L_solar)
        input R should be in pc
        '''
        surf = self.mge2d[:, 0]
        twoSigma2 = 2.0 * self.mge2d[:, 1]**2
        q2 = self.mge2d[:, 2]**2
        I = dblquad(_inte_2dencloseM, 0.0, 0.5*np.pi, lambda x: 0.0,
                    lambda x: R, args=(surf, twoSigma2, q2))
        return I[0] * 4.0

    def Phi(self, R, z):
        '''
        Return the gravitational potential at R, z (R, z in pc)
        '''
        G = 0.00430237
        mge3d = self.deprojection()
        sigma = mge3d[:, 1]
        q2 = mge3d[:, 2]**2
        M = mge3d[:, 0] * (SQRT_TOW_PI*sigma)**3 * mge3d[:, 2]
        I = quad(_Hu_inte, 0, 1, args=(R, z, M, sigma, q2))
        return -np.sqrt(2.0/np.pi)*G*I[0]

    def Vc(self, R, ml=1.0):
        '''
        Return the rotational velocity at R on the equatorial plane (in km/s)
        R in pc
        ml mass-to-light ratio of these Gaussians
        '''
        Force = derivative(self.Phi, R, args=([0]))
        return np.sqrt(ml*Force*R)

    def Re(self):
        return _Re(self.mge2d)

    def r_half_cir(self):
        return _r_half_cir(self.mge2d)

    def r_half(self):
        return _r_half(self.deprojection())
