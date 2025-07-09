import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation, exposure

def segment_teeth_cbct(
    vol: np.ndarray,
    spacing=(0.15, 0.15, 0.15),     # (z, y, x) mm voxel size
    smooth_sigma=1.0,               # Gaussian blur in voxel units
    enamel_thresh=1200,             # HU threshold that keeps enamel/dentine
    mandible_thresh=350,            # HU threshold for cancellous bone
    size_threshold=5_000,           # drop small CCs (< voxels)
    watershed_separate=True,
):
    """
    Classic (non-DL) tooth segmentation for 3-D CBCT volumes.

    Parameters
    ----------
    vol : ndarray, shape (Z, Y, X)
        Grayscale CBCT in (approximate) Hounsfield units.
    spacing : tuple
        Physical voxel spacing in mm (used for structuring-element sizing).
    smooth_sigma : float
        σ for isotropic Gaussian smoothing.
    enamel_thresh : int
        Lower HU bound to isolate tooth hard tissue.
    mandible_thresh : int
        Lower HU bound to mask out low-density cancellous bone
        when separating teeth from the jaw.
    size_threshold : int
        Remove connected components smaller than this.
    watershed_separate : bool
        If True, run a marker-controlled watershed to
        split touching teeth into individual labels.

    Returns
    -------
    teeth_mask : ndarray, bool
        Binary mask of all teeth.
    labels     : ndarray, int
        Instance-label image (0 = background).  Only returned if
        `watershed_separate` is True.
    """
    # 0. (optional) bias‐field correction / intensity normalisation
    vol_eq = exposure.equalize_adapthist(vol.astype(np.float32), clip_limit=0.02)

    # 1. slight Gaussian smoothing (reduces photon noise)
    vol_s = ndi.gaussian_filter(vol_eq, sigma=smooth_sigma)

    # 2. two-level thresholding
    hard = vol_s >= enamel_thresh            # keeps enamel + dentine
    bone = vol_s >= mandible_thresh          # cancellous + cortical bone

    # 3. morphological cleaning in 3-D
    #    – close small gaps inside tooth crowns
    ball2 = morphology.ball(radius= int(round(0.6 / spacing[0])))    # ≈0.6 mm
    hard_closed = morphology.closing(hard, ball2)

    #    – remove thin mandible walls by erosion/dilation
    ball3 = morphology.ball(radius= int(round(1.5 / spacing[0])))    # ≈1.5 mm
    mandible_eroded = morphology.erosion(bone, ball3)
    mask = hard_closed & ~mandible_eroded          # teeth but not jaw bone

    # 4. keep only sizeable components (crowns + roots)
    mask = morphology.remove_small_objects(mask, min_size=size_threshold)

    if not watershed_separate:
        return mask.astype(np.uint8)

    # 5. distance-transform + marker-controlled watershed for instances
    dist = ndi.distance_transform_edt(mask, sampling=spacing)
    # local maxima as markers
    loc_max = morphology.local_maxima(dist)        # boolean image
    markers = measure.label(loc_max)
    labels = segmentation.watershed(-dist, markers, mask=mask)

    # 6. tidy small spurious labels
    labels = morphology.remove_small_objects(labels, size_threshold)
    return mask.astype(np.uint8), labels.astype(np.uint16)
