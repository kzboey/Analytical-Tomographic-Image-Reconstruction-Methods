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

class Fbp_fan(XrayOperator):
    
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
        dtype="float32",       
        weighting = "" # normal, parker or empty string (evaluated based on orbit)
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
        self.weighting = weighting
        
    """
        Only account for flat detector (equidistant case)
    """
    def applyAdjoint(self, sino):
        nb, na = sino.shape
        nx, ny = self.vol_shape
        ds = self.vol_spacing[0]
        pixel_size = 1 # unit spacing
        offset = self.pp_offset
        dso = self.s2c            # source to isocenter distance
        dsd = self.s2c + self.c2d # source to detector distance
        dod = self.c2d            # object to detector distance
        ia = self.thetas
        thetas = self.thetas
        weighting = self.weighting
        orbit = deg2rad(np.max(thetas) - np.min(thetas))
        orbit_start = deg2rad(np.min(thetas))
        betas = deg2rad(ia)
        images = [] # store intemediate backprojected images
        
        # Step 1: Weight the sinogram
        if weighting == "parker":
            weight = self.parker_weight(nb, na, ds, thetas, dsd, offset)
        elif weighting == "normal":
            weight = self.fan_weight(nb, na, ds, dso, dsd, offset)
        else:
            if np.max(thetas) - np.min(thetas) < 180 == 0: # if some points of the sinogram are only sampled once
                weight = self.parker_weight(nb, na, ds, orbit, dsd, offset)
            else:
                weight = self.fan_weight(nb, na, ds, dso, dsd, offset)
         
        # print("weight: ", weight)
        wsino = weight * sino
        sino = wsino
        
        # Step 2: filter the sinogram
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * nb))))
        pad_width = ((0, projection_size_padded - nb), (0, 0))
        fsino = np.zeros([projection_size_padded, na])
        ramp = ramp_filter(projection_size_padded)
        ramp = tile(ramp, na)
        
        fsino = np.pad(sino, pad_width, mode="constant", constant_values=0)
        fsino = fft(fsino , axis=0) * ramp
        fsino = real(ifft(fsino, axis=0)[:nb, :]) # real(ifft(fsino, axis=0))

        # Fan beam FBP step 3: backproject the filtered sinogram
        xc, yc = np.meshgrid((np.arange(1, nx + 1) - (nx + 1) / 2) * pixel_size,
                             (np.arange(1, ny + 1) - (ny + 1) / 2) * pixel_size)
        rr = sqrt(xc**2 + yc**2)

        smax = ((nb - 1) / 2 - np.abs(offset)) * ds
        gamma_max = atan(smax / dsd)
        rmax = dso * sin(gamma_max)
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
            
            img[mask] += (wl * fsino[il, ia] + wr * fsino[il+1, ia]) * w2
            images.append(np.copy(img))
            
        img = 0.5 * orbit / na * img

        return img, images, wsino
          
    @staticmethod
    def fan_weight(nb, na, ds, dso, dsd, offset):
        nn = np.arange(-(nb - 1) / 2, (nb - 1) / 2 + 1) 
        ss = ds * nn
        gam = np.arctan(ss / dsd)
        weighting = np.abs(dso * np.cos(gam) - offset * np.sin(gam)) / dsd
  
        return np.tile(weighting, (na,1)).T

    
    @staticmethod
    def parker_weight(nb, na, ds, bet, dsd, offset):
        # Implement equation (3.9.35) from 3.9.3 FBP for short scans
        print("parker weight used")
        nn = np.arange(-(nb - 1) / 2, (nb - 1) / 2 + 1)
        ss = ds * nn
        gam = np.arctan(ss / dsd)
        
        smax = ((nb - 1) / 2 - np.abs(offset)) * ds
        fan_angle = np.max(bet)
        gam_max = fan_angle/4

        gg, bb = np.meshgrid(bet, gam)

        wt = np.zeros((nb, na))
        
        # q(x) function used in the chapter
        fun = lambda x: sin(pi / 2 * x) ** 2
        for i in range(nb):
            for j in range(na):
                if bb[i, j] < 2 * (gam_max - gg[i, j]): # Condition 1
                    wt[i, j] = fun(bb[i, j] / (2 * (gam_max - gg[i, j]))) 
                elif 2 * (gam_max - gg[i, j]) <= bb[i, j] < np.pi - 2 * gg[i, j]: # Condition 2
                    wt[i, j] = 1
                elif np.pi - 2 * gg[i, j] < bb[i, j] <= np.pi + 2 * gam_max: # Condition 3
                    wt[i, j] = fun( (np.pi + 2 * gam_max - bb[i, j]) / (2 * (gam_max + gg[i, j]))) 

        return wt

    
   