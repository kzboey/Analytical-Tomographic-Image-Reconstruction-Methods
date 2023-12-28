import pyelsa as elsa
from reconstruction.XrayOperator import XrayOperator
from reconstruction.ramp import ramp_filter
import numpy as np
from scipy.interpolate import BSpline

class RebinFan2Par(XrayOperator):
    
    ## Initialize geometries and implement forward projection to acquire the sinogram in the parent class -> Fanbeam
    def __init__(
        self,
        vol_shape,
        sino_shape,
        thetas,
        s2c,
        c2d,
        vol_spacing=None,
        sino_spacing=None,
        cor_offset=None,
        pp_offset=None,
        projection_method="josephs",
        dtype="float32"           
    ):
        super().__init__(
            vol_shape=vol_shape,
            sino_shape=sino_shape,
            thetas=thetas,
            s2c=s2c,
            c2d=c2d,
            vol_spacing=vol_spacing,
            sino_spacing=sino_spacing,
            cor_offset=cor_offset,
            pp_offset=pp_offset,
            projection_method=projection_method,
            dtype=dtype 
        )
              
    def applyAdjoint(self, sino):
        psino = self.rebin_fan2par(sino) # psino is rebinned parallel sinogram
        img = super.applyAdjoint(psino)
        
        return img
        
    
    def rebin_fan2par(self, sino):
        ns, nbeta = sino.shape  # Assuming 2D sinogram
        ds = self.sino_spacing
        offset = self.pp_offset
        thetas = self.thetas
        
        if len(thetas) != nbeta:
            raise ValueError("Mismatch in the number of projection angles and sinogram shape")

        na, beta_start, beta_orbit = len(thetas), thetas[0], thetas[-1] - thetas[0]
        dso = self.s2c
        dsd = self.s2c + self.c2d

        nr, dr  = self.vol_shape[0], self.sino_spacing
        nphi, phi_start, phi_orbit = na, beta_start, beta_orbit 

        r_ob, phi_ob, flag180 = self.rebin_fan2par_setup(
            ns, ds, offset,
            na, beta_start, beta_orbit,
            dso, dsd, 
            nr, dr, offset,
            nphi, phi_start, phi_orbit,
        )
        
        psino = self.rebin_fan2par_arg_many(r_ob, phi_ob, flag180, sino)

        return psino

    def rebin_fan2par_setup(
        self,
        ns, ds, offset_s,
        nbeta, beta_start, beta_orbit,
        dso, dsd, 
        nr, dr, offset_r,
        nphi, phi_start, phi_orbit,
    ):

        if phi_orbit == 180 and phi_orbit == 360:
            flag180 = 1
        else:
            flag180 = 0

        phi_start = np.deg2rad(phi_start)
        phi_orbit = np.deg2rad(phi_orbit)
        beta_start = np.deg2rad(beta_start)
        beta_orbit = np.deg2rad(beta_orbit)
        phi = phi_start + phi_orbit * np.arange(nphi) / nphi
        
        # angular interpolator

        if flag180:
            phi2 = phi_start + 2 * phi_orbit * np.arange(2 * nphi) / (2 * nphi)
        else:
            if phi_orbit != np.deg2rad(360) or beta_orbit != np.deg2rad(360):
                raise NotImplementedError("Only 360 degrees orbit is supported")

            phi2 = phi
            
        s = (np.arange(ns) - (ns - 1) / 2 - offset_s) * ds
        bet = np.outer(phi2, -s / dsd)

        bet_int = nbeta / beta_orbit * (bet - beta_start)

        phi_ob = BSpline(np.arange(nbeta), bet_int, 3, extrapolate=True)

        # radial interpolator
        
        wr = (nr - 1) / 2 + offset_r
        dr = dr * np.max(np.abs([np.cos(phi), np.sin(phi)]), axis=0)

        r = (np.arange(nr) - wr) * dr
        s = dsd * np.arcsin(r / dso) # if not np.isinf(dsd) else r

        if flag180:
            if offset_s != 0.25 and offset_s != 1.25:
                raise NotImplementedError("Only offsets 0.25 and 1.25 are implemented")
            ns = 2 * ns + 4 * (offset_s - 0.25)
            offset_s = 0
            ws = (ns - 1) / 2 + offset_s
            ds /= 2

        s_int = s / ds + (ns - 1) / 2 + offset_s
        r_ob = BSpline(np.sort(s_int), np.arange(ns), 3, extrapolate=True)

        return r_ob, phi_ob, flag180
    
    def rebin_fan2par_arg_many(self, r_ob, phi_ob, flag180, fsino):
        fsino = fsino[:,:,np.newaxis]
        tmp = fsino.shape
        nt = np.prod(tmp[2:])
        fsino = np.reshape(fsino, [tmp[0], tmp[1], nt])
        # psino = np.zeros((self.nbeta, self.ns, nt), dtype=fsino.dtype)
        psino = np.zeros_like(fsino)

        for it in range(nt):
            psino[:, :, it] = self.rebin_fan2par_arg(r_ob, phi_ob, flag180, fsino[:, :, it])

        psino = np.reshape(psino, [tmp[0], tmp[1]] + list(tmp[2:]))

        return psino

    def rebin_fan2par_arg(self, r_ob, phi_ob, flag180, fsino):
        nsamples = len(fsino)

        # Evaluate phi_ob at nsamples points
        phi_values = phi_ob(np.arange(nsamples))
        phi_values = np.sort(phi_values)
        # Evaluate r_ob at nsamples points
        r_values = r_ob(np.tile(np.arange(nsamples), (nsamples,1)))
        r_values = np.clip(r_values, 0, None)

        sino = np.dot(phi_values, fsino).T

        if flag180:
            sino = self.rebin_fan2par_inlace(sino)

        sino = np.dot(r_values, sino.T).T

        return sino.T

    def rebin_fan2par_inlace(self, sino):
        ns, nphi = sino.shape
        t1 = sino[:, :nphi]
        t2 = np.flip(sino[:, nphi:], axis=0)

        if self.pp_offset == 1.25:  # trick
            t1 = np.vstack([t1, t2[-2:, :]])
            t2 = np.vstack([t1[:2, :], t2])
            ns = ns + 2
        elif self.offset != 0.25:
            raise ValueError('Invalid offset value')

        sino = np.vstack([t1.flatten(), t2.flatten()]).T.reshape(2*ns, nphi)
        return sino
    
    """
    def rebin_fan2par_arg_many(self, r_ob, phi_ob, flag180, fsino):
        ns, nbeta = fsino.shape  # Assuming 2D sinogram
        psino = np.zeros((nbeta, ns), dtype=fsino.dtype)

        # for it in range(nbeta):
        psino = self.rebin_fan2par_arg(r_ob, phi_ob, flag180, fsino)

        return psino

    def rebin_fan2par_arg(self, r_ob, phi_ob, flag180, fsino):
        nb, na = fsino.shape
    
        # Evaluate phi_ob at nsamples points
        phi_values = phi_ob(np.arange(nb))
        # print("phi_values: ",phi_values.shape)

        # Evaluate r_ob at nsamples points
        r_values = r_ob(np.arange(nb, na))
        r_values = r_ob(np.tile(np.arange(na), (na,1)))
        
        # print("r_values: ",r_values.shape)
        
        psino = np.dot(phi_values, fsino).T
        
        if flag180:
            psino = rebin_fan2par_inlace(parameters, psino)
        
        psino = np.dot(r_values, psino)

        return psino
    
    def rebin_fan2par_inlace(self, arg, sino):
        ns, nphi = sino.shape
        t1 = sino[:, :nphi]  
        t2 = np.flip(sino[:, nphi:], axis=0)  

        if self.pp_offset == 1.25:  # trick
            t1 = np.vstack([t1, t2[-2:, :]])  
            t2 = np.vstack([t1[:2, :], t2])  
            ns = ns + 2
        elif self.offset != 0.25:
            raise ValueError('Invalid offset value')

        sino = np.vstack([t1.flatten(), t2.flatten()]).T.reshape(2*ns, nphi)
        return sino
    

    """
