"""
features.py

Utility functions for calculating common image features
"""
import numpy as np
from utils.imvector import imvector
from utils.logging import print_indent, g_indents, print_timer
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

def image_entropy(image, radius=2, verbose=False):
    """compute the pixel-wise entropy of an image over a region defined by neighborhood
    
    Args:
        image -- a flattened array of pixel intensities of type imslice or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        H as imvolume with shape=image.shape
    """
    if isinstance(image, imvector):
        d = image.depth
        r = image.rows
        c = image.columns

        def get_val(image, z, y, x):
            # image boundary handling is built into imvector.get_val
            return image.get_val(z,y,x)
        def set_val(image, z, y, x, val):
            image.set_val(z,y,x,val)

        #instantiate a blank imvector of the proper size
        H = imvector(np.zeros((d, r, c)))
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            d, r, c = image.shape
        elif image.ndim == 2:
            d, r, c = (1, *image.shape)
            image = image.reshape((d,r,c))

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
        print('invalid image type supplied. Please specify an image of type imvector or type np.ndarray')
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

    # nested loop approach -> slowest, try GPU next
    idx = -1
    total_voxels = d * r * c
    for z in range(d):
        for y in range(r):
            for x in range(c):
                idx += 1
                if ( verbose and (idx % 50000 == 0 or idx == total_voxels-1)):
                    print_indent('{p:0.2%} - voxel: {i:d} of {tot:d}'.format(
                        p=idx/total_voxels,
                        i=idx,
                        tot=total_voxels), l4)
                #print('z:{z:d}, y:{y:d}, x:{x:d}'.format(z=z,y=y,x=x))
                val_counts = {}
                for k_z in z_radius_range:
                    for k_x in range(-radius, radius+1):
                        for k_y in range(-radius, radius+1):
                            #print('k_z:{z:d}, k_y:{y:d}, k_x:{x:d}'.format(z=k_z,y=k_y,x=k_x))
                            # Calculate probabilities
                            val = get_val(image, z+k_z,y+k_y,x+k_x)
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
    if type(image) is np.ndarray and d == 1:
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

