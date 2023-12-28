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

class Feldkamp(XrayOperator):
    
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
        
    """
        Only account for flat detector (equidistant case)
    """
    def applyAdjoint(self, sino):
        # Precompute variables
        ns, nt, na = sino.shape
        nx, ny, nz = self.vol_shape
        ds, dt = self.sino_spacing 
        dx, dy, dz = self.vol_spacing
        offset_s, offset_t = self.pp_offset 
        offset_x, offset_y, offset_z = self.cor_offset
        dsd = self.s2c
        dso = self.s2c + self.c2d
        thetas = self.thetas
        orbit = deg2rad(np.max(thetas) - np.min(thetas))
        orbit_start = deg2rad(np.min(thetas))
        ss = (np.arange(-(ns-1)/2, (ns-1)/2+1) - offset_s) * ds
        tt = (np.arange(-(nt-1)/2, (nt-1)/2+1) - offset_t) * dt
        ss, tt = np.meshgrid(ss, tt)
        
        # Setup the projection weighting (3.10.5)
        ww1 = dso / np.sqrt(dsd**2 + ss**2 + tt**2)
        ww1 = dso / np.sqrt(dsd**2 + ss**2 + tt**2) * np.sqrt(1 + (tt/dsd)**2)
        
        # Setup filter
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * ns))))
        pad_width = ((0, projection_size_padded - ns), (0, 0))
        fsino = np.zeros([projection_size_padded, nt, na])
        ramp = ramp_filter(projection_size_padded)
        ramp = tile(ramp, nt)
        
        for ia in range(0, na):
            # Step 1: Weight the sinogram
            sino[:, :, ia] = sino[:, :, ia] * ww1
        
            # Step 2: filter the sinogram
            projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * ns)))) # Add paddinga
            pad_width = ((0, projection_size_padded - ns), (0, 0))

            fsino = np.pad(sino[:, :, ia], pad_width, mode="constant", constant_values=0)  
            fsino = fft(fsino , axis=0) * ramp
            sino[:, :, ia] = real(ifft(fsino, axis=0)[:nt, :])
            
        # Step 3: Backproject the sinogram
        
        # Precompute backprojection variables   
        betas = deg2rad(orbit_start + orbit * np.arange(na) / na)  # [na] source angles
        ia_skip = 1
        wx = (nx - 1) / 2 + offset_x
        wy = (ny - 1) / 2 + offset_y
        wz = (nz - 1) / 2 + offset_z
        xc, yc = np.meshgrid((np.arange(nx) - wx) * dx, (np.arange(ny) - wy) * dy)
        zc = (np.arange(nz) - wz) * dz
        
        rr = sqrt(xc**2 + yc**2)  # [nx,ny]
        smax = ((ns - 1) / 2 - abs(offset_s)) * ds  # maximum detector s coordinate
            
        gamma_max = atan(smax / dsd)  
        
        rmax = dso * sin(gamma_max)
        mask = np.logical_and(rr < rmax, np.ones_like(rr, dtype=bool))
        mask = mask & (rr < rmax)
        xc, yc = xc[mask], yc[mask]
        
        ws = (ns ) / 2 + offset_s  
        wt = (nt ) / 2 + offset_t
            
        img = np.zeros([nx, ny, nz])
        sdim = [ns + 1, nt]  # trick: extra zeros for zero extrapolation in "s"
        proj1 = np.zeros(sdim)
        
        for iz in range(nz):
            ia_min = 0
            ia_max = na

            # loop over each projection angle
            img2 = 0
            for ia in range(ia):
                beta = betas[ia]

                x_beta = xc * np.cos(beta) + yc * np.sin(beta)
                y_betas = dso - (-xc * np.sin(beta) + yc * np.cos(beta))

                # detector indices 
                mag = dsd / y_betas  # [np]
                sprime = mag * x_beta
                tprime = mag * (zc[iz]) #  - source_zs[ia])  # \tbxyz
                
                # Weight from (3.10.9)
                bs = sprime / ds + ws
                bt = tprime / dt + wt

                bs[(bs < 1) | (bs > ns)] = ns + 1  # trick for zero extrapolation in s
                bt = np.maximum(bt, 1)  # implicit extrapolation in t (detector row)
                bt = np.minimum(bt, nt)

                # bi-linear interpolation:
                is_ = np.floor(bs).astype(int)  # left bin
                it = np.floor(bt).astype(int)

                is_[is_ == ns + 1] = ns  # trick for zero extrapolation in s
                it[it == nt] = nt - 1  # trick for last row extrapolation in t

                wr = bs - is_  # left weight
                wl = 1 - wr  # right weight
                wu = bt - it  # upper weight
                wd = 1 - wu  # lower weight

                proj1[0:ns, :] = sino[:, :, ia]  # trick: 1 extra zero at ns+1
                p1 = wl * proj1[is_, it] + wr * proj1[is_ + 1, it]
                p2 = wl * proj1[is_, it + 1] + wr * proj1[is_ + 1, it + 1]

                p0 = wd * p1 + wu * p2  # vertical interpolation
                p0 = p0 * mag**2  # back-projection weighting for flat
                
                # np reshape p0 to img2
                img2 += np.reshape(p0,(nx,ny))

            img[:, :, iz] = img2

        img = (0.5 * deg2rad(abs(orbit)) / (na / ia_skip)) * img
        img[img < 0] = 0

        return img
    