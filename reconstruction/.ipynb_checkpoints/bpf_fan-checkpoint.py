import pyelsa as elsa
from reconstruction.XrayOperator import XrayOperator
from reconstruction.ramp import ramp_filter
import numpy as np

# function alias 
sin = np.sin
cos = np.cos
atan = np.arctan
fft = np.fft.fft
ifft = np.fft.ifft
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
sqrt = np.sqrt
real = np.real
pi = np.pi
tile = np.tile
deg2rad = np.deg2rad

class Bpf_fan(XrayOperator):
    
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
        nb, na = sino.shape
        nx, ny = self.vol_shape
        ds = self.vol_spacing[0]
        pixel_size = 1 # unit spacing
        offset = self.pp_offset
        dso = self.s2c
        dsd = self.s2c + self.c2d # source to detector distance
        dod = self.c2d            # object to detector distance
        ia = self.thetas
        thetas = self.thetas
        orbit = deg2rad(np.max(thetas) - np.min(thetas))
        orbit_start = deg2rad(np.min(thetas))
        betas = deg2rad(ia)
        
        # Step 1: Weight the sinogram
        if np.max(thetas) - np.min(thetas)%180 == 0: # if some points of the sinogram are only sampled once
            weight = self.parker_weight(nb, na, ds, orbit, dsd, offset)
        else:
            weight = self.fan_weight(nb, na, ds, dso, dsd, offset)
            
        wsino = weight * sino
    
        # Fan beam FBP step 2: backproject the filtered sinogram
        xc, yc = np.meshgrid((np.arange(1, nx + 1) - (nx + 1) / 2) * pixel_size,
                             (np.arange(1, ny + 1) - (ny + 1) / 2) * pixel_size)
        rr = sqrt(xc**2 + yc**2)

        smax = ((nb - 1) / 2 - np.abs(offset)) * ds
        gamma_max = np.arctan(smax / dsd)
        rmax = dso * np.sin(gamma_max)
        mask = np.logical_and(rr < rmax, np.ones_like(rr, dtype=bool))
        xc, yc = xc[mask], yc[mask]

        img = np.zeros_like(rr)

        ia_values = np.arange(0, na)
        betas = orbit_start + orbit * ia_values / na

        # Loop over each projection angle
        for ia in ia_values:
            beta = betas[ia]
            d_loop = dso + xc * sin(beta) - yc * cos(beta)
            x_beta = xc * cos(beta) + yc * sin(beta)
            w2 = dsd**2 / d_loop**2 # backprojection weighting
            sprime_ds = (dsd / ds) * (x_beta - offset) / d_loop
            bb = sprime_ds + ((nb + 1) / 2 + offset)

            # Linear interpolation
            il = np.floor(bb).astype(int)
            il = np.clip(il, 0, nb - 2) 
            wr = bb - il # left weight
            wl = 1 - wr # right weight          
            img[mask] += (wl * sino[il, ia] + wr * sino[il+1, ia]) * w2

        img = 0.5 * orbit / na * img
        
        # Step 3: filter the backprojected image

        img_shape = img.shape[0]
        img_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
        pad_width = ((0, img_size_padded - img_shape), (0, 0))
        fimg = np.zeros([img_size_padded, img_shape])
        ramp = ramp_filter(img_size_padded)
        ramp = tile(ramp, img_shape)
        
        fimg = np.pad(img, pad_width, mode="constant", constant_values=0)
        fimg = fft(fimg , axis=0) * ramp
        img = real(ifft(fimg, axis=0)[:img_shape, :])

        return img
          
    @staticmethod
    def fan_weight(nb, na, ds, dso, dsd, offset):
        nn = np.arange(-(nb - 1) / 2, (nb - 1) / 2 + 1) 
        ss = ds * nn
        gam = np.arctan(ss / dsd)
        weighting = np.abs(dso * np.cos(gam) - offset * np.sin(gam)) / dsd
  
        return np.tile(weighting, (na,1)).T

    @staticmethod
    def parker_weight(nb, na, ds, bet, dsd, offset):
        nn = np.arange(-(nb - 1) / 2, (nb - 1) / 2 + 1) 
        ss = ds * nn
        gam = np.arctan(ss / dsd)
        
        smax = ((nb - 1) / 2 - np.abs(offset)) * ds
        gam_max = np.arctan(smax / dsd)
        
        gg, bb = np.meshgrid(gam, bet)
        
        fun = lambda x: sin(pi/2 * x)**2

        wt = np.zeros((nb, na))

        ii = (bb < 2 * (gam_max - gg)) & (bb >= 0)
        tmp = bb[ii] / (2 * (gam_max - gg[ii]))
        wt[ii] = fun(tmp)

        ii = (2 * (gam_max - gg) <= bb) & (bb < pi - 2 * gg)
        wt[ii] = 1

        ii = (pi - 2 * gg < bb) & (bb <= pi + 2 * gam_max)
        tmp = (pi + 2 * gam_max - bb[ii]) / (2 * (gam_max + gg[ii]))
        wt[ii] = fun(tmp)

        return wt
    
    