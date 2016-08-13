"""
features.py

Utility functions for calculating common image features
"""
import numpy as np
from utils.logging import print_indent, g_indents, print_timer
import utils
from utils import rttypes
from utils.rttypes import imvolume as imvolume, featvolume as featvolume
import time
import sys
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.compiler
from pycuda.compiler import SourceModule
pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--std=c++11']

#indent shortnames
l3 = g_indents[3]
l4 = g_indents[4]

def image_entropy(image_volume, radius=2, mask=False, ROIName=None, verbose=False):
    """compute the pixel-wise entropy of an image over a region defined by neighborhood
    
    Args:
        image -- a flattened array of pixel intensities of type imslice or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        H as imvolume with shape=image.shape
    """
    if True or type(image_volume) == 'utils.rttypes.imvolume':
        d = image_volume.numberOfSlices
        r = image_volume.rows
        c = image_volume.columns

        #prepare mask vector within image_volume

        def get_val(image_volume, z, y, x):
            # image boundary handling is built into imvector.get_val
            return image_volume.get_val(z,y,x, ROIName=ROIName)
        def set_val(image_volume, z, y, x, val):
            image_volume.set_val(z,y,x,val)

        #instantiate a blank imvector of the proper size
        H = featvolume((d, r, c))
    elif isinstance(image_volume, np.ndarray):
        if image_volume.ndim == 3:
            d, r, c = image_volume.shape
        elif image_volume.ndim == 2:
            d, r, c = (1, *image_volume.shape)
            image_volume = image_volume.reshape((d,r,c))

        #instantiate a blank np.ndarray of the proper size
        H = np.zeros((d, r, c))

        def get_val(image, z, y, x):
            if (z<0 or y<0 or x<0) or (z>=d or y>=r or x>=c):
                return 0
            else:
                return image[z, y, x]
        def set_val(image, z, y ,x, val):
            image[z, y, x] = val
    else:
        print('invalid image type supplied ({:s}). Please specify an image of type imvector \
            or type np.ndarray'.format(str(type(image_volume))))
        return None

    # z_radius_range controls 2d neighborhood vs 3d neighborhood for 2d vs 3d images
    if d == 1: #2D image
        if verbose:
            print_indent('Computing 2D entropy with radius: {:d}'.format(radius), l3)
        z_radius_range = [0]
    elif d>1: # 3D image
        if verbose:
            print_indent('Computing 3D entropy with radius: {:d}'.format(radius), l3)
        z_radius_range = range(-radius, radius+1)

    # timing
    start_entropy_calc = time.time()

    # crop to ROI then expand to image size with fill 0 later
    if (mask):
        global_limits = {'xmax': -4000,
                         'ymax': -4000,
                         'zmax': -4000,
                         'xmin': 4000,
                         'ymin': 4000,
                         'zmin': 4000 }
        for i in range(1, image_volume.numberOfSlices+1):
            maskds = image_volume.getMaskSlice(i, asdataset=True, ROIName=ROIName, vectorize=False)
            if maskds is None:
                continue
            #convert coords list to ndarray
            coords = np.array(maskds.contour_points)
            (xmin, ymin, zmin) = tuple(coords.min(axis=0, keepdims=False))
            (xmax, ymax, zmax) = tuple(coords.max(axis=0, keepdims=False))

            #update limits
            if xmin < global_limits['xmin']:
                global_limits['xmin'] = xmin
            if ymin < global_limits['ymin']:
                global_limits['ymin'] = ymin
            if zmin < global_limits['zmin']:
                global_limits['zmin'] = zmin
            if xmax > global_limits['xmax']:
                global_limits['xmax'] = xmax
            if ymax > global_limits['ymax']:
                global_limits['ymax'] = ymax
            if zmax > global_limits['zmax']:
                global_limits['zmax'] = zmax

        (x_rel_start, y_rel_start, z_rel_start) = image_volume.imagePositionPatient
        (y_space, x_space) = image_volume.pixelSpacing
        z_space = image_volume.sliceThickness
        dstart = int(round((global_limits['zmin']-z_rel_start)/z_space))
        dstop = int(round((global_limits['zmax']-z_rel_start)/z_space))
        rstart = int(round((global_limits['ymin']-y_rel_start)/y_space))
        rstop = int(round((global_limits['ymax']-y_rel_start)/y_space))
        cstart = int(round((global_limits['xmin']-x_rel_start)/x_space))
        cstop = int(round((global_limits['xmax']-x_rel_start)/x_space))
    else:
        dstart = 0
        dstop = d
        rstart = 0
        rstop = r
        cstart = 0
        cstop = c

    print('calculation subset volume z=({zstart:d}->{zstop:d}), '
                                    'y=({ystart:d}->{ystop:d}), '
                                    'x=({xstart:d}->{xstop:d})'.format(
                                        zstart=dstart,
                                        zstop=dstop,
                                        ystart=rstart,
                                        ystop=rstop,
                                        xstart=cstart,
                                        xstop=cstop ) )
    # nested loop approach -> slowest, try GPU next
    idx = -1
    total_voxels = d * r * c
    subset_idx = -1
    subset_total_voxels = (dstop-dstart+1) * (cstop-cstart+1) * (rstop-rstart+1)
    onepercent = subset_total_voxels / 100
    fivepercent = 5*onepercent
    for z in range(d):
        for y in range(r):
            for x in range(c):
                idx += 1
                if (z<dstart or z>dstop or y<rstart or y>rstop or x<cstart or x>cstop):
                    #fill 0 instead
                    set_val(H,z,y,x,0)
                else:
                    subset_idx += 1
                    if ( verbose and (subset_idx % fivepercent == 0 or subset_idx == subset_total_voxels-1)):
                        print_indent('{p:0.2%} - voxel: {i:d} of {tot:d} (of total: {abstot:d})'.format(
                            p=subset_idx/subset_total_voxels,
                            i=subset_idx,
                            tot=subset_total_voxels,
                            abstot=total_voxels), l4)
                    #print('z:{z:d}, y:{y:d}, x:{x:d}'.format(z=z,y=y,x=x))
                    val_counts = {}
                    for k_z in z_radius_range:
                        for k_x in range(-radius, radius+1):
                            for k_y in range(-radius, radius+1):
                                #print('k_z:{z:d}, k_y:{y:d}, k_x:{x:d}'.format(z=k_z,y=k_y,x=k_x))
                                # Calculate probabilities
                                val = get_val(image_volume, z+k_z,y+k_y,x+k_x)
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
                    set_val(H, z, y, x, h)
    if isinstance(image_volume, np.ndarray) and d == 1:
        # need to reshape ndarray if input was 2d
        H = H.reshape((r, c))


    end_entropy_calc = time.time()
    if verbose:
        print_timer('entropy calculation time:', end_entropy_calc-start_entropy_calc, l3)
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
    elif isinstance(image_vect, imvector):
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

    print(type(result))
    print(result.shape)
    print('GPU done')
    return result.reshape(r,c)

