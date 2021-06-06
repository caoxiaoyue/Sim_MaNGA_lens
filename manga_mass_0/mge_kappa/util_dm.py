from __future__ import division
import numpy as np
#from mge1d_fit import mge1d_fit
from scipy.integrate import quad,dblquad


TWO_PI = 2.0 * np.pi
FOUR_PI = 4.0 * np.pi
SQRT_TOW_PI = np.sqrt(2.0*np.pi)


def _mge1dfit(r, rho, q=0.999, **kwargs):
    '''
    fit a 1D profile (e.g. a spherical dark halo) with MGEs and return the
      3D-deprojected mge coefficients
    important kwargs
      imax - maximum iteration steps (5-10 is enough)
      ngauss - number of Gaussians used in fitting (~10)
      rbound - range of the MGE sigmas (usually do not need to provide)
    return
      mge3d - N*3 3d gaussian coefficients
    '''
    err = r*0.0 + np.mean(rho) * 0.1
    mge = mge1d_fit(r, rho, err, **kwargs)
    mge3d = np.zeros([mge.shape[0], 3])
    mge3d[:, 0] = mge[:, 0]/SQRT_TOW_PI/mge[:, 1]
    mge3d[:, 1] = mge[:, 1]
    mge3d[:, 2] = q
    return mge3d


class gnfw1d:
    def __init__(self, rho_s, rs, gamma):
        '''
        unit:
          rho_s [M_solar/kpc^3], is 4 times larger than usual rho_s
          rs [kpc]
          gamma [none], usually between [-2.0, 0], for NFW, gamma = -1.0
        '''
        self.rho_s = rho_s
        self.rs = rs
        self.gamma = gamma

    def densityProfile(self, r):
        '''
        Return the density values at given r
        r [kpc]
        densityProfile [M_solar/kpc^3]
        '''
        rdivrs = r / self.rs
        return self.rho_s*rdivrs**self.gamma *\
            (0.5+0.5*rdivrs)**(-self.gamma-3)
        
    def surfaceDensity(self, R):
        '''
        Return the surface density values at projected radius R(in unit M_solar/kpc^2)
        '''
        def _integrate(Z,R):
            r = np.sqrt(R**2+Z**2)
            rdivrs = r / self.rs
            rst = self.rho_s*rdivrs**self.gamma *\
                (0.5+0.5*rdivrs)**(-self.gamma-3)
            return rst
        return quad(_integrate, -np.inf, np.inf, args=(R))[0]
    
    
    def enclosed2dMass(self, R, **kwargs):
        '''
        Return the enlosed projected mass within projected radius R
        R [kpc]
        enclosed2dMass [M_solar]
        '''
        
        def _integrate(Z,R1):
            r = np.sqrt(R1**2+Z**2)
            rdivrs = r / self.rs
            rst = self.rho_s*rdivrs**self.gamma *\
                (0.5+0.5*rdivrs)**(-self.gamma-3) * TWO_PI * R1
            return rst
        enclosedmass2nd = dblquad(_integrate,0.0,R,lambda R1:-np.inf, lambda R1:np.inf)[0]
        return enclosedmass2nd

    def enclosedMass(self, R, **kwargs):
        '''
        Return the enlosed mass within R
        R [kpc]
        enclosedMass [M_solar]
        '''
        def _integrate(r):
            rdivrs = r / self.rs
            rst = self.rho_s*rdivrs**self.gamma *\
                (0.5+0.5*rdivrs)**(-self.gamma-3) * FOUR_PI * r * r
            return rst
        return quad(_integrate, 0.0, R, **kwargs)[0]
    
    def surfaceDensity_dh2ndNFW(self, R, rho_s, rs, d):
        '''
        Include a second NFW dark halo(rho_s,rs) at distance d
        Return the surface density contribution from the second dark halo at projected radius R(in unit M_solar/kpc^2)
        input:
            rho_s [M_solar/kpc^3]
            rs [kpc]
            R [kpc]
            d [kpc]
        output:
            surface mass density [M_solar/kpc^2]
        '''
        def _integrate(theta, R, d):
            '''
            Surface mass density of NFW halo come from Eq(11) in Wright & Brainerd 2000
            https://ui.adsabs.harvard.edu/abs/2000ApJ...534...34W/abstract
            '''
            R1 = np.sqrt(R**2+d**2-2*d*R*np.cos(theta))
            if R1 < rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(1-R1**2/rs**2)*np.arctanh(np.sqrt((1-R1/rs)/(1+R1/rs))))
            elif R1 == rs:
                rst = 2*rs*rho_s/3.
            elif R1 > rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(R1**2/rs**2-1)*np.arctan(np.sqrt((R1/rs-1)/(1+R1/rs))))
            return rst
        return quad(_integrate,0,np.pi,args=(R,d))[0]/np.pi
    
    def enclosed2dMass_dh2ndNFW(self, R, rho_s, rs, d, **kwargs):
        '''
        Include a second NFW dark halo(rho_s,rs) at distance d
        Return the enlosed projected mass contribution from the second dark halo within projected radius R
        R [kpc]
        enclosed2dMass [M_solar]
        '''
        def _integrate1(R1, R, d):
            '''
            Surface mass density of NFW halo come from Eq(11) in Wright & Brainerd 2000
            https://ui.adsabs.harvard.edu/abs/2000ApJ...534...34W/abstract
            '''
            if d == 0:
                d += 1e-8
            theta = np.arccos((d**2+R1**2-R**2)/(2*d*R1))
            if R1 < rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(1-R1**2/rs**2)*np.arctanh(np.sqrt((1-R1/rs)/(1+R1/rs))))
            elif R1 == rs:
                rst = 2*rs*rho_s/3.
            elif R1 > rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(R1**2/rs**2-1)*np.arctan(np.sqrt((R1/rs-1)/(1+R1/rs))))
            return rst *2*theta*R1
        def _integrate2(R1):
            if R1 < rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(1-R1**2/rs**2)*np.arctanh(np.sqrt((1-R1/rs)/(1+R1/rs))))
            elif R1 == rs:
                rst = 2*rs*rho_s/3.
            elif R1 > rs:
                rst = 2*rs*rho_s/(R1**2/rs**2-1)*(1-2./np.sqrt(R1**2/rs**2-1)*np.arctan(np.sqrt((R1/rs-1)/(1+R1/rs))))
            return rst*TWO_PI*R1
        if d>=R:
            enclosedmass2nd = quad(_integrate1,d-R,d+R,args=(R,d))[0]
        elif d<R:
            enclosedmass2nd1 = quad(_integrate1,R-d,R+d,args=(R,d))[0]
            enclosedmass2nd2 = quad(_integrate2,0.0,R-d)[0]
            enclosedmass2nd = enclosedmass2nd1 + enclosedmass2nd2
        return enclosedmass2nd

    def surfaceDensity_dh2nd(self, R, rho_s, rs, gamma, d):
        '''
        same formulae as http://jesford.github.io/cluster-lensing/
        Include a second gNFW dark halo(rho_s,rs,gamma) at distance d
        Return the surface density contribution from the second dark halo at projected radius R(in unit M_solar/kpc^2)
        input:
            rho_s [M_solar/kpc^3], is 4 times larger than usual rho_s
            rs [kpc]
            gamma [none], usually between [-2.0, 0], for NFW, gamma = -1.0
            R [kpc]
            d [kpc]
        output:
            surface mass density [M_solar/kpc^2]
        '''
        def _integrate(Z,theta,R,d):
            #theta is the angle between R and d
            r = np.sqrt(R**2+d**2-2*d*R*np.cos(theta)+Z**2)
            rdivrs = r / rs
            rst = rho_s*rdivrs**gamma *\
                (0.5+0.5*rdivrs)**(-gamma-3)
            return rst
        surfdens2nd = dblquad(_integrate,0.0,np.pi,lambda theta:-np.inf,lambda theta:np.inf,args=(R,d))[0]/np.pi
        return surfdens2nd

    def enclosed2dMass_dh2nd(self, R, rho_s, rs, gamma, d, **kwargs):
        '''
        Include a second gNFW dark halo(rho_s,rs,gamma) at distance d
        Return the enlosed projected mass contribution from the second dark halo within projected radius R
        input:
            R [kpc]
            rho_s [M_solar/kpc^3], is 4 times larger than usual rho_s
            rs [kpc]
            gamma [none], usually between [-2.0, 0], for NFW, gamma = -1.0
            d [kpc]
        output:
            enclosed2dMass [M_solar]
        '''
        def _integrate1(Z,R1,R,d):
            #theta is the angle between d and R1 (corresponding angle of R)
            #R1 is the distance between the center of 2nd dark halo and the point at R
            if d == 0:
                d += 1e-8
            theta = np.arccos((d**2+R1**2-R**2)/(2*d*R1))
            r = np.sqrt(R1**2+Z**2)
            rdivrs = r / rs
            rst = rho_s*rdivrs**gamma *\
                (0.5+0.5*rdivrs)**(-gamma-3)*2*theta*R1
            return rst
        def _integrate2(Z,R1):
            r = np.sqrt(R1**2+Z**2)
            rdivrs = r / rs
            rst = rho_s*rdivrs**gamma *\
                (0.5+0.5*rdivrs)**(-gamma-3)*TWO_PI*R1
            return rst
        if d>=R:
            enclosedmass2nd = dblquad(_integrate1,d-R,d+R,lambda R1:-np.inf,lambda R1:np.inf,args=(R,d))[0]
        elif d<R:
            enclosedmass2nd1 = dblquad(_integrate1,R-d,R+d,lambda R1:-np.inf,lambda R1:np.inf,args=(R,d))[0]
            enclosedmass2nd2 = dblquad(_integrate2,0.0,R-d,lambda R1:-np.inf, lambda R1:np.inf)[0]
            enclosedmass2nd = enclosedmass2nd1 + enclosedmass2nd2
        return enclosedmass2nd

    def mge3d(self, rrange=[0.1, 200], ngauss=10, imax=5, npoints=200):
        '''
        rrange [kpc]
        mge3d [M_solar/pc^3]  [pc]  [none]
        '''
        r = np.logspace(np.log10(rrange[0]), np.log10(rrange[1]), npoints)
        rho = self.densityProfile(r)
        return _mge1dfit(r*1e3, rho/1e9, imax=imax, ngauss=ngauss)

    def mge2d(self, rrange=[0.1, 200], ngauss=10, imax=5, npoints=200):
        '''
        fit this halo profile using MGE method and return the MGE coefficents
        rrange [kpc], within which profile is fitted.
        mge2d [M_solar/pc^2]  [pc]  [none]
        '''
        mge3d = self.mge3d(rrange=rrange, ngauss=ngauss,
                           imax=imax, npoints=npoints)
        mge3d[:, 0] *= mge3d[:, 1]*SQRT_TOW_PI
        return mge3d


    
    
class gnfw2d:
    def __init__(self, rho_s, rs, gamma, q):
        '''
        unit:
          rho_s [M_solar/kpc^3]
          rs [kpc]
          gamma [none], usually between [-2.0, 0], for NFW, gamma = -1.0
          q [none], intrinsic axis ratio
        '''
        self.rho_s = rho_s
        self.rs = rs
        self.gamma = gamma
        self.q = q

    def densityProfile(self, R, z):
        '''
        Return the density values at given (R, z)
        (R, z) [kpc]
        densityProfile [M_solar/kpc^3]
        '''
        m = np.sqrt(R**2 + (z/self.q)**2)
        rdivrs = m / self.rs
        return self.rho_s*rdivrs**self.gamma *\
            (0.5+0.5*rdivrs)**(-self.gamma-3)

    def enclosedMass(self, R, **kwargs):
        '''
        Return the enlosed mass within R
        R [kpc]
        enclosedMass [M_solar]
        '''
        print('Not implemented yet')
        return None

    def mge3d(self, rrange=[0.1, 200], ngauss=10, imax=5, npoints=200):
        '''
        rrange [kpc]
        mge3d [M_solar/pc^3]  [pc]  [none]
        '''
        m = np.logspace(np.log10(rrange[0]), np.log10(rrange[1]), npoints)
        rho = self.densityProfile(m, 0)
        return _mge1dfit(m*1e3, rho/1e9, q=self.q, imax=imax, ngauss=ngauss)

    def mge2d(self, inc, rrange=[0.1, 200], ngauss=10, imax=5,
              npoints=200):
        '''
        fit this halo profile using MGE method and return the MGE coefficents
        rrange [kpc], within which profile is fitted.
        inc: inclination in radians
        mge2d [M_solar/pc^2]  [pc]  [none]
        '''
        mge3d = self.mge3d(rrange=rrange, ngauss=ngauss,
                           imax=imax, npoints=npoints)
        dens = mge3d[:, 0]
        sigma3d = mge3d[:, 1]
        qint = mge3d[:, 2]

        mge2d = mge3d.copy()
        qobs = np.sqrt(qint**2 * np.sin(inc)**2 + np.cos(inc)**2)
        surf = dens * qint / qobs * (SQRT_TOW_PI * sigma3d)
        mge2d[:, 0] = surf
        mge2d[:, 2] = qobs
        return mge2d


# do not use this
def gnfw(r, rho_s, rs, gamma):
    rdivrs = r / rs
    return rho_s*rdivrs**gamma * (0.5+0.5*rdivrs)**(-gamma-3)
