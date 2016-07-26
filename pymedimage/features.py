"""
features.py

Utility functions for calculating common image features
"""
import numpy as np
from utils.imvector import imvector
#import pycuda.autoinit
#import pycuda.driver as cuda
#from pycuda.compiler import SourceModule

def image_entropy(image, radius=2):
    """compute the pixel-wise entropy of an image over a region defined by neighborhood
    
    Args:
        image -- a flattened array of pixel intensities of type imvector or a matrix shaped numpy ndarray
        radius -- describes neighborood size in each dimension. radius of 4 would be a 9x9x9
    Returns:
        H as imvector with shape=image.shape
    """
    bigTime
    
    if type(image) is imvector:
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
    elif type(image) is np.ndarray:
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
        z_radius_range = [0]
    elif d>1: # 3D image
        z_radius_range = range(-radius, radius+1) 

    # nested loop approach -> slowest, try GPU next
    for z in range(d):
        for y in range(r):
            for x in range(c):
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
                #calculate local entropy
                h = -np.sum(val_probs*np.log(val_probs)) #/ np.log(65536)
                set_val(H, z, y, x, h)
    if type(image) is np.ndarray and d == 1:
        # need to reshape ndarray if input was 2d
        H = H.reshape((r, c))
    return H


def image_entropy_gpu(image_vect, radius=2):
    """Uses PyCuda to parallelize the computation of the voxel-wise image entropy using a variable neighborhood radius

    Args:
	radius -- neighborhood radius; where neighborhood size is isotropic and calculated as 2*radius+1
    """
    mod = SourceModule("""
    #include <math.h> /* pow */

    __global__ void image_entropy(float *dest, float *image_vect, int radius)
    {
        // array index for this thread
        const int i = blockIdx.x * blockDims.x + threadIdx.x;

        // Setup hash array for storing counts and initialize all to 0
        int counts[pow(2*radius+1, 3)] = {0};

        for (int k_z = -radius; k_z <= radius; k_z++) {
            for (int k_x = -radius; k_x <= radius; k_x++) {
                for (int k_y = -radius; k_y <= radius; k_y++) {
                    // Count unique pixel intensities
                    /* THIS IS GOING TO REQUIRE SOME ADDITIONAL THOUGHT
                        ABOUT HOW TO REDUCE THREAD DIVERGENCE WHEN COUNTING THE UNIQUE
                        OCCURENCES OF EACH PIXEL.
                        COULD POTENTIALLY HASH THE PIXEL INTENSITY AND ASSIGN/INCREMENT COUNTS TO A LARGE ARRAY AT THE INDEX
                        RETURNED BY THE HASHING ALGO. THEN LATER ACCESS THE STORED ARRAY OF HASH VALUES AND COUNTS, TO COMPUTE
                        PROBABILITIES AND EVENTUAL VOXEL-WISE ENTROPY
                        */

        }
    }
    """)

