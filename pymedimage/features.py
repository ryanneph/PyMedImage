"""
features.py

Utility functions for calculating common image features
"""
import logging
import numpy as np
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

def image_entropy(image_volume, radius=2, roi=None):
    """compute the pixel-wise entropy of an image over a region defined by neighborhood

    Args:
        image -- a flattened array of pixel intensities of type imslice or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        H as MaskableVolume with shape=image.shape
    """
    if (MaskableVolume.__name__ in str(type(image_volume))):  # This is an ugly way of type-checking but cant get isinstance to see both as the same
        (c, r, d) = image_volume.frameofreference.size
        def get_val(image_volume, z, y, x):
            # image boundary handling is built into BaseVolume.get_val
            return image_volume.get_val(z, y, x)
        def set_val(feature_volume, z, y, x, val):
            feature_volume.set_val(z, y, x, val)

        #instantiate a blank BaseVolume of the proper size
        H = MaskableVolume().fromArray(np.zeros((d, r, c)), image_volume.frameofreference)
        H.modality = image_volume.modality
        H.feature_label = 'entropy'
    elif isinstance(image_volume, np.ndarray):
        if image_volume.ndim == 3:
            c, r, d = image_volume.shape
        elif image_volume.ndim == 2:
            c, r, d = (1, *image_volume.shape)
            image_volume = image_volume.reshape((d, r, c))

        # instantiate a blank np.ndarray of the proper size
        H = np.zeros((d, r, c))

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
        logger.debug(indent('Computing 2D entropy with radius: {:d}'.format(radius), l3))
        z_radius_range = [0]
    elif d > 1:  # 3D image
        logger.debug(indent('Computing 3D entropy with radius: {:d}'.format(radius), l3))
        z_radius_range = range(-radius, radius+1)

    # timing
    start_entropy_calc = time.time()

    # set calculation bounds
    cstart, cstop = 0, c
    rstart, rstop = 0, r
    dstart, dstop = 0, d

    # absolute max indices for imagevolume - for handling request of voxel out of bounds
    cbound = c
    rbound = r
    dbound = d

    # restrict calculation bounds to roi
    if (roi is not None):
        # get max extents of the mask/ROI to speed up calculation only within ROI cubic volume
        extents = roi.getROIExtents()
        cstart, rstart, dstart = image_volume.frameofreference.getIndices(extents.start)
        cstop, rstop, dstop = image_volume.frameofreference.getIndices(extents.end())
        logger.info(indent('calculation subset volume x=({xstart:d}->{xstop:d}), '
                                               'y=({ystart:d}->{ystop:d}), '
                                               'z=({zstart:d}->{zstop:d})'.format(zstart=dstart,
                                                                                  zstop=dstop,
                                                                                  ystart=rstart,
                                                                                  ystop=rstop,
                                                                                  xstart=cstart,
                                                                                  xstop=cstop ), l4))
        # redefine H
        d_subset = dstop - dstart
        r_subset = rstop - rstart
        c_subset = cstop - cstart
        entropy_frameofreference = FrameOfReference((extents.start),
                                                    (image_volume.frameofreference.spacing),
                                                    (c_subset, r_subset, d_subset))
        H = H.fromArray(np.zeros((d_subset, r_subset, c_subset)), entropy_frameofreference)
    else:
        d_subset = dstop - dstart
        r_subset = rstop - rstart
        c_subset = cstop - cstart

    # nested loop approach -> slowest, try GPU next
    total_voxels = d * r * c
    subset_total_voxels = d_subset * r_subset * c_subset
    onepercent = int(subset_total_voxels / 100)
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
                    set_val(H, z_idx, y_idx, x_idx, 0)
                else:
                    subset_idx += 1
                    if ((subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug(indent('{p:0.2%} - voxel: {i:d} of {tot:d} (of total: {abstot:d})'.format(
                            p=subset_idx/subset_total_voxels,
                            i=subset_idx,
                            tot=subset_total_voxels,
                            abstot=total_voxels), l4))
                    val_counts = {}
                    for k_z in z_radius_range:
                        for k_x in range(-radius, radius+1):
                            for k_y in range(-radius, radius+1):
                                #logger.info('k_z:{z:d}, k_y:{y:d}, k_x:{x:d}'.format(z=k_z,y=k_y,x=k_x))
                                # handle out of bounds requests - replace with 0
                                request_z = z+k_z
                                request_y = y+k_y
                                request_x = x+k_x
                                if (request_z >= dbound or
                                    request_y >= rbound or
                                    request_x >= cbound):
                                    val = 0
                                else:
                                    val = get_val(image_volume, request_z, request_y, request_x)

                                # Calculate probabilities
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
                    set_val(H, z_idx, y_idx, x_idx, h)
                    if (False and (subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        logger.debug('total counts: ' + str(total_counts))
                        logger.debug('val_probs = ' + str(val_probs))
                        logger.debug('entropy at ({x:d}, {y:d}, {z:d})= {e:f}'.format(
                            x=z*y*x + y*x + x,
                            y=z*y*x + y,
                            z=z*y*x,
                            e=h))
    if isinstance(image_volume, np.ndarray) and d == 1:
        # need to reshape ndarray if input was 2d
        H = H.reshape((r_subset, c_subset))


    end_entropy_calc = time.time()
    logger.debug(timer('entropy calculation time:', end_entropy_calc-start_entropy_calc, l3))
    return H


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

