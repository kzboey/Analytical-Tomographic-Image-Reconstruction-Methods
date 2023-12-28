import numpy as np
import pyelsa as elsa

class XrayOperator():
    """
    Parameters
    --------------------------------------------------------------------------------
    vol_shape : :obj:`np.ndarray`
        Size of the image/volume
    sino_shape : :obj:`np.ndarray`
        Size of the sinogram
    thetas : :obj:`np.ndarray`
        List of projection angles in degree
    s2c : :obj:`float32`
        Distance from source to center of rotation
    c2d : :obj:`float32`
        Distance from center of rotation to detector
    vol_spacing : :obj:`np.ndarray`, optional
        Spacing of the image/volume, i.e. size of each pixel/voxel. By default
        unit size is assumed.
    sino_spacing : :obj:`np.ndarray`, optional
        Spacing of the sinogram, i.e. size of each detector pixel. By default
        unit size is assumed.
    cor_offset : :obj:`np.ndarray`, optional
        Offset of the center of rotation. By default no offset is applied.
    pp_offset : :obj:`np.ndarray`, optional
        Offset of the principal point. By default no offset is applied.
    projection_method : :obj:`str`, optional
        Projection method used for the forward and backward projections. By
        default the interpolation/Joseph's method ('josephs') is used. Can also
        be 'siddons', for the line intersection length methods often referred
        to as Siddons method.
    dtype : :obj:`float32`, optional
        Type of elements in input array.
    --------------------------------------------------------------------------------
    """
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
        self.vol_shape = np.array(vol_shape)
        self.sino_shape = np.array(sino_shape)

        # Sinogram is of the same dimension as volume (i.e. it's a stack
        # of (n-1)-dimensional projection)
        if self.vol_shape.size != (self.sino_shape.size + 1):
            raise RuntimeError(
                f"Volume and sinogram must be n and (n-1) dimensional (is {self.vol_shape.size} and {self.sino_shape.size})"
            )

        self.ndim = np.size(vol_shape)

        self.thetas = np.array(thetas)

        # thetas needs to be a 1D array / list
        if self.thetas.ndim != 1:
            raise RuntimeError(
                f"angles must be a 1D array or list (is {self.thetas.ndim})"
            )

        self.s2c = s2c
        self.c2d = c2d

        self.vol_spacing = (
            np.ones(self.ndim) if vol_spacing is None else np.array(vol_spacing)
        )
        self.sino_spacing = (
            np.ones(self.ndim - 1) if sino_spacing is None else np.array(sino_spacing)
        )
        self.cor_offset = (
            np.zeros(self.ndim) if cor_offset is None else np.array(cor_offset)
        )
        self.pp_offset = (
            np.zeros(self.ndim - 1) if pp_offset is None else np.array(pp_offset)
        )
        
                # Some more sanity checking
        if self.vol_spacing.size != self.ndim:
            raise RuntimeError(
                f"Array containing spacing of volume is of the wrong size (is {self.vol_spacing.size}, expected {self.ndim})"
            )

        if self.cor_offset.size != self.ndim:
            raise RuntimeError(
                f"Array containing offset of center of rotation is of the wrong size (is {self.cor_offset.size}, expected {self.ndim})"
            )

        if self.sino_spacing.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing spacing of detector is of the wrong size (is {self.sino_spacing.size}, expected {self.ndim - 1})"
            )

        if self.pp_offset.size != self.ndim - 1:
            raise RuntimeError(
                f"Array containing principal point offset is of the wrong size (is {self.pp_offset.size}, expected {self.ndim - 1})"
            )

        self.vol_descriptor = elsa.VolumeDescriptor(self.vol_shape, self.vol_spacing)
        self.sino_descriptor = elsa.CircleTrajectoryGenerator.trajectoryFromAngles(
            thetas,
            self.vol_descriptor,
            self.s2c,
            self.c2d,
            self.pp_offset,
            self.cor_offset,
            self.sino_shape,
            self.sino_spacing,
        )
  
        if projection_method == "josephs":
            self.projector = elsa.JosephsMethod(self.vol_descriptor, self.sino_descriptor)
        elif projection_method == "siddons":
            self.projector = elsa.SiddonsMethod(self.vol_descriptor, self.sino_descriptor)
        else:
            raise RuntimeError(f"Unknown projection method '{projection_method}'")
        
    """ Implementation of forward projection """
    def apply(self, x):
        # copy/move numpy array to elsa
        ex = elsa.DataContainer(np.reshape(x, self.vol_shape, order="C"), self.vol_descriptor)
        
        sino = self.projector.apply(ex)
        
        return np.array(sino)
    
    """ 
        Default reconstruction method: backprojection
        Other reconstruction methods (rebin, fbp, bpf) to be implemented in respective derived class 
    """
    def applyAdjoint(self, sino):
        shape = np.concatenate((self.sino_shape, np.array([np.size(self.thetas)])))
        esino = elsa.DataContainer(
            np.reshape(sino, shape, order="C"), self.sino_descriptor
        )

        # perform backward projection
        bp = self.projector.applyAdjoint(esino)

        # return a numpy array
        return np.array(bp) / len(self.thetas)
        
    