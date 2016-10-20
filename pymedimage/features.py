"""
features.py

Utility functions for calculating common image features
"""
import logging
import numpy as np
import scipy.ndimage
import pywt
from utils.rttypes import BaseVolume, MaskableVolume, FrameOfReference
from utils.misc import g_indents, indent, timer
import time
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--std=c++11']

# initialize module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# indent shortnames
l3 = g_indents[3]
l4 = g_indents[4]

def image_iterator(processing_function, image_volume, radius=2, roi=None):
    """compute the pixel-wise feature of an image over a region defined by neighborhood

    Args:
        processing-function -- function that should be applied to at each voxel location with neighborhood
                                context. Function signature should match:
                                    fxn()
        image -- a flattened array of pixel intensities of type imslice or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        feature_volume as MaskableVolume with shape=image.shape
    """
    # This is an ugly way of type-checking but cant get isinstance to see both as the same
    if (MaskableVolume.__name__ in str(type(image_volume))):
        (c, r, d) = image_volume.frameofreference.size
        def get_val(image_volume, z, y, x):
            # image boundary handling is built into BaseVolume.get_val
            return image_volume.get_val(z, y, x)
        def set_val(feature_volume, z, y, x, val):
            feature_volume.set_val(z, y, x, val)

        #instantiate a blank BaseVolume of the proper size
        feature_volume = MaskableVolume().fromArray(np.zeros((d, r, c)), image_volume.frameofreference)
        # feature_volume.modality = image_volume.modality
        # feature_volume.feature_label = 'feature'
    elif isinstance(image_volume, np.ndarray):
        if image_volume.ndim == 3:
            c, r, d = image_volume.shape
        elif image_volume.ndim == 2:
            c, r, d = (1, *image_volume.shape)
            image_volume = image_volume.reshape((d, r, c))

        # instantiate a blank np.ndarray of the proper size
        feature_volume = np.zeros((d, r, c))

        def get_val(image, z, y, x):
            if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
                return 0
            else:
                return image[z, y, x]
        def set_val(image, z, y, x, val):
            image[z, y, x] = val
    else:
        logger.info('invalid image type supplied ({:s}). Please specify an image of type BaseVolume \
            or type np.ndarray'.format(str(type(image_volume))))
        return None

    # z_radius_range controls 2d neighborhood vs 3d neighborhood for 2d vs 3d images
    if d == 1:  # 2D image
        logger.debug(indent('Computing 2D feature with radius: {:d}'.format(radius), l3))
        z_radius_range = [0]
    elif d > 1:  # 3D image
        logger.debug(indent('Computing 3D feature with radius: {:d}'.format(radius), l3))
        z_radius_range = range(-radius, radius+1)

    # in plane range
    radius_range = range(-radius, radius+1)

    # timing
    start_feature_calc = time.time()

    # set calculation bounds
    cstart, cstop = 0, c-1
    rstart, rstop = 0, r-1
    dstart, dstop = 0, d-1

    # absolute max indices for imagevolume - for handling request of voxel out of bounds
    cbound = c-1
    rbound = r-1
    dbound = d-1

    # restrict calculation bounds to roi
    if (roi is not None):
        # get max extents of the mask/ROI to speed up calculation only within ROI cubic volume
        extents = roi.getROIExtents()
        cstart, rstart, dstart = image_volume.frameofreference.getIndices(extents.start)
        cstop, rstop, dstop = image_volume.frameofreference.getIndices(extents.end())
        logger.info(indent('calculation subset volume x=({xstart:d}->{xstop:d}), '
                                               'y=({ystart:d}->{ystop:d}), '
                                               'z=({zstart:d}->{zstop:d})'.format(zstart=dstart,
                                                                                  zstop=dstop-1,
                                                                                  ystart=rstart,
                                                                                  ystop=rstop-1,
                                                                                  xstart=cstart,
                                                                                  xstop=cstop-1 ), l4))
        # redefine feature_volume
        d_subset = dstop - dstart
        r_subset = rstop - rstart
        c_subset = cstop - cstart
        feature_frameofreference = FrameOfReference((extents.start),
                                                    (image_volume.frameofreference.spacing),
                                                    (c_subset, r_subset, d_subset))
        feature_volume = feature_volume.fromArray(np.zeros((d_subset, r_subset, c_subset)), feature_frameofreference)
    else:
        d_subset = dstop - dstart
        r_subset = rstop - rstart
        c_subset = cstop - cstart

    # nested loop approach -> slowest, try GPU next
    total_voxels = d * r * c
    subset_total_voxels = d_subset * r_subset * c_subset
    #onepercent = int(subset_total_voxels / 100)
    fivepercent = int(subset_total_voxels / 100 * 5)

    idx = -1
    subset_idx = -1
    z_idx = -1
    for z in range(dstart, dstop):
        z_idx += 1
        y_idx = -1
        x_idx = -1
        for y in range(rstart, rstop):
            y_idx += 1
            x_idx = -1
            for x in range(cstart, cstop):
                x_idx += 1
                idx += 1
                if (z<dstart or z>dstop or y<rstart or y>rstop or x<cstart or x>cstop):
                    # we shouldnt ever be here
                    logger.info('why are we here?!')
                    #fill 0 instead
                    set_val(feature_volume, z_idx, y_idx, x_idx, 0)
                else:
                    subset_idx += 1
                    patch_vals = np.zeros((len(z_radius_range), len(radius_range), len(radius_range)))
                    for p_z, k_z in enumerate(z_radius_range):
                        for p_x, k_x in enumerate(radius_range):
                            for p_y, k_y in enumerate(radius_range):
                                #logger.info('k_z:{z:d}, k_y:{y:d}, k_x:{x:d}'.format(z=k_z,y=k_y,x=k_x))
                                # handle out of bounds requests - replace with 0
                                request_z = z+k_z
                                request_y = y+k_y
                                request_x = x+k_x
                                if (request_z < 0 or request_z >= dbound or
                                    request_y < 0 or request_y >= rbound or
                                    request_x < 0 or request_x >= cbound):
                                    val = 0
                                else:
                                    val = get_val(image_volume, request_z, request_y, request_x)
                                # store to local image patch
                                patch_vals[p_z, p_y, p_x] = val

                    proc_value = processing_function(patch_vals)
                    set_val(feature_volume, z_idx, y_idx, x_idx, proc_value)

                    if (False and (subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug('feature value at ({x:d}, {y:d}, {z:d})= {e:f}'.format(
                            x=z*y*x + y*x + x,
                            y=z*y*x + y,
                            z=z*y*x,
                            e=proc_value))

                    if ((subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug(indent('{p:0.2%} - voxel: {i:d} of {tot:d} (of total: {abstot:d})'.format(
                            p=subset_idx/subset_total_voxels,
                            i=subset_idx,
                            tot=subset_total_voxels,
                            abstot=total_voxels), l4))

    if isinstance(image_volume, np.ndarray) and d == 1:
        # need to reshape ndarray if input was 2d
        feature_volume = feature_volume.reshape((r_subset, c_subset))


    end_feature_calc = time.time()
    logger.debug(timer('feature calculation time:', end_feature_calc-start_feature_calc, l3))
    return feature_volume

def entropy_plugin(patch_vals):
    val_counts = {}

    # get occurence counts
    for val in patch_vals.flatten().tolist():
        if val in val_counts:
            val_counts[val] += 1
        else:
            val_counts[val] = 1

    #create new dict to store class probabilities
    val_probs = np.zeros(((len(val_counts))))
    total_counts = sum(val_counts.values())
    for i, val in enumerate(val_counts.keys()):
        val_probs[i] = val_counts[val]/total_counts
    # calculate local entropy
    h = -np.sum(val_probs*np.log(val_probs)) #/ np.log(65536)

    return h


def image_entropy(image_volume, radius=2, roi=None):
    return image_iterator(entropy_plugin, image_volume, radius, roi)


def image_entropy_gpu(image_vect, radius=2):
    """Uses PyCuda to parallelize the computation of the voxel-wise image entropy using a variable neighborhood radius

    Args:
	radius -- neighborhood radius; where neighborhood size is isotropic and calculated as 2*radius+1
    """
    mod = SourceModule("""
    __global__ void image_entropy2(
        float *image_vect)
    {
        int radius = 4;
        int z_radius = 0;
        // array index for this thread
        int idx = blockIdx.y * (blockDim.x * blockDim.y) * 32
                + blockIdx.x * (blockDim.x * blockDim.y)
                + threadIdx.y * (blockDim.x)
                + threadIdx.x;
        //image_vect[idx] = idx % 255;

        for (int k_z = -z_radius; k_z <= z_radius; k_z++) {
            for (int k_x = -radius; k_x <= radius; k_x++) {
                for (int k_y = -radius; k_y <= radius; k_y++) {
                    int k_idx = blockIdx.z * (threadIdx.z + k_z * blockDim.y * blockDim.x)
                              + blockIdx.y * (threadIdx.y + k_y * blockDim.x)
                              + blockIdx.x * (threadIdx.x + k_x);

                    // Count unique pixel intensities
                    //val = fmax(0, image_vect[k_idx])
                    image_vect[idx] = idx % 255;
                }
            }
        }


    }
    """)

    func = mod.get_function('image_entropy2')

    if isinstance(image_vect, np.ndarray):
        if image_vect.ndim == 3:
            d, r, c = image_vect.shape
        elif image_vect.ndim == 2:
            d, r, c = (1, *image_vect.shape)
        image = image_vect.flatten()
    elif isinstance(image_vect, BaseVolume):
        d = image_vect.depth
        r = image_vect.rows
        c = image_vect.columns
        image = image_vect.array

    if d == 1:
        z_radius = 0
    elif d > 1:
        z_radius = radius

    # block, grid dimensions

    # allocate image on device in global memory
    image = image.astype(np.float32)
    image_gpu = cuda.mem_alloc(image.nbytes)
    result = np.empty_like(image)
    result_gpu = cuda.mem_alloc(image.nbytes)
    # transfer image to device
    cuda.memcpy_htod(image_gpu, image)
    cuda.memcpy_htod(result_gpu, result)
    # call device kernel
    func(image_gpu, block=(16,16,1), grid=(32,32,1))
    # get result from device
    cuda.memcpy_dtoh(result, image_gpu)

    logger.info(type(result))
    logger.info(result.shape)
    logger.info('GPU done')
    return result.reshape(r,c)


def wavelet_decomp_3d(image_volume, wavelet_str='db1', mode_str='smooth'):
    """perform full 3d wavelet decomp and return coefficients"""
    coeffs = pywt.wavedecn(image_volume.array, wavelet_str, mode_str)
    return coeffs

def wavelet_energy_plugin(patch_vals):
    patch_vals = patch_vals.flatten()
    return np.true_divide(np.dot(patch_vals, patch_vals), pow(len(patch_vals), 2))

def wavelet_energy(image_volume, radius=2, roi=None, wavelet_str='db1', mode_str='smooth'):
    # compute wavelet coefficients
    logger.info(indent('performing 3d wavelet decomp using wavelet: {!s}'.format(wavelet_str), g_indents[3]))
    roi_volume = image_volume.conformTo(roi.frameofreference)
    wavelet_coeffs = wavelet_decomp_3d(roi_volume, wavelet_str, mode_str)
    nlevels = len(wavelet_coeffs) - 1
    # level_results = []
    accumulator = np.zeros(roi_volume.frameofreference.size[::-1])
    # sum voxel-wise energy across all levels
    for level in range(nlevels-1, 0, -1):
        wavelet_coeffs_diag = wavelet_coeffs[level+1]['ddd']
        zoomfactors = tuple(np.true_divide(roi_volume.frameofreference.size[::-1], wavelet_coeffs_diag.shape))
        # scale low-res coefficients to image res
        upsampled_roi_volume = scipy.ndimage.interpolation.zoom(wavelet_coeffs_diag,
                                                                  zoomfactors, order=3)
        upsampled_roi_volume = MaskableVolume().fromArray(upsampled_roi_volume, roi_volume.frameofreference)
        logger.info(indent('computing energy for level {:d} of shape:{!s}'.format(level, wavelet_coeffs_diag.shape), g_indents[3]))
        result = image_iterator(wavelet_energy_plugin, upsampled_roi_volume, radius)

        # level_results.append(result)
        accumulator = np.add(accumulator, result.array)
    return MaskableVolume().fromArray(accumulator, roi_volume.frameofreference)
