from string import Template
import logging
import math
import numpy as np
from utils.rttypes import MaskableVolume
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
pycuda.compiler.DEFAULT_NVCC_FLAGS = ['--std=c++11']

# initialize module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def image_iterator_gpu(image_volume, roi=None, radius=2, gray_levels=12, dx=1, dy=0, dz=0, ndev=2,
             feature_kernel='glcm_plugin_gpu', stat_name='glcm_stat_contrast_gpu'):
    """Uses PyCuda to parallelize the computation of the voxel-wise image entropy using a variable \
            neighborhood radius

    Args:
	radius -- neighborhood radius; where neighborhood size is isotropic and calculated as 2*radius+1
    """
    cuda_template = Template("""
    #define RADIUS $RADIUS
    #define Z_RADIUS $Z_RADIUS
    #define PATCH_SIZE (RADIUS*2+1)*(RADIUS*2+1)*(Z_RADIUS*2+1)
    #define IMAGE_WIDTH $IMAGE_WIDTH
    #define IMAGE_HEIGHT $IMAGE_HEIGHT
    #define IMAGE_DEPTH $IMAGE_DEPTH
    #define IMAGE_SIZE IMAGE_WIDTH*IMAGE_HEIGHT*IMAGE_DEPTH
    #define NBINS $NBINS
    #define GLCM_SIZE NBINS*NBINS
    #define NDEV $NDEV
    #define DX $DX
    #define DY $DY
    #define DZ $DZ

    #include "math.h"
    #include <stdio.h>

    /* UTILITIES */
    __device__ float rn_mean(float *array) {
        // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
        int size = PATCH_SIZE;

        // computes the mean of the array
        int total = 0;
        for (int i=0; i<size; i++) {
            total += array[i];
        }
        return ((float)total / size);
    }
    __device__ float rn_std(float *array) {
        // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
        int size = PATCH_SIZE;

        // get the mean
        float mu = rn_mean(array);

        // compute the std. deviation
        int total = 0;
        for (int i=0; i<size; i++) {
            total += powf( array[i] - mu, 2);
        }
        return sqrtf((float)total / size);
    }
    __device__ void quantize_gpu(float *array) {
        // quantize the array into nbins, placing lower 2.5% and upper 2.5% residuals of gaussian
        // distribution into first and last bins respectively

        // gaussian stats
        float f_mean = rn_mean(array);
        float f_std = rn_std(array);
        float f_binwidth = 2*NDEV*f_std / (NBINS-2);
        //printf("mean:%f std:%f width:%f\\n", f_mean, f_std, f_binwidth);

        // rebin values
        for (int i=0; i<PATCH_SIZE; i++) {
            array[i] = fminf(NBINS-1, fmaxf(0, floorf(((array[i] - f_mean + NDEV*f_std)/(f_binwidth+1e-9)) + 1)));
        }
    }

    /* ENTROPY ENERGY */
    __device__ float entropy_plugin_gpu(float *patch_array) {
        // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
        int size = PATCH_SIZE;

        quantize_gpu(patch_array);

        // count intensity occurences
        int hist[NBINS];
        for (int i=0; i<NBINS; i++) {
            hist[i] = 0;
        }
        for (int i=0; i<size; i++) {
            hist[(int)floorf(patch_array[i])]++;
        }

        // calculate local entropy
        float stat = 0.0;
        for (int i=0; i<NBINS; i++) {
            stat -= ((float)hist[i]/size) * (logf(((float)hist[i]/size) + 1e-9));
        }

        return stat;
    }
    __device__ float energy_plugin_gpu(float *patch_array) {
        // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
        int size = PATCH_SIZE;

        quantize_gpu(patch_array);

        // count intensity occurences
        int hist[NBINS];
        for (int i=0; i<NBINS; i++) {
            hist[i] = 0;
        }
        for (int i=0; i<size; i++) {
            hist[(int)floorf(patch_array[i])] += 1;
        }

        // calculate local energy
        float stat = 0.0;
        for (int i=0; i<NBINS; i++) {
            stat += powf((float)hist[i]/size, 2);
        }

        return sqrtf(stat);
    }

    /* GLCM FEATURES */
    __device__ void glcm_matrix_gpu(int *mat, float *array) {
        // compute glcm matrix in the specified directions
        int dx = 1;
        int dy = 0;
        int dz = 0;

        int patch_size_1d = (RADIUS*2+1);
        int patch_size_2d = pow(patch_size_1d, 2);
        for (int i=0; i<PATCH_SIZE; i++) {
            int z = i / patch_size_2d;
            int y = (i - patch_size_2d*z) / patch_size_1d;
            int x = i - patch_size_2d*z - patch_size_1d*y;
            int y_idx = array[i];
            int x_idx_query = (int)fmaxf(0,fminf(PATCH_SIZE-1, (z+dz)*patch_size_2d + (y+dy)*patch_size_1d + (x + dx)));
            int x_idx = (int)(array[x_idx_query]);

            // Logging
            //if (threadIdx.x == 50 && blockIdx.x == 20) {
            //    printf("i:%d - y:%d,x:%d - yq:%d,xq:%d - y_idx:%d,x_idx:%d\\n", i, y, x, i, x_idx_query, y_idx, x_idx);
            //}
            //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //    printf("%d:%d\\n", i, y);
            //    printf("%d:%d,%d\\n", i, y_idx, x_idx);
            //}

            int mat_idx = NBINS * y_idx + x_idx;
            mat[mat_idx] += 1;
            int mat_idx_inv = NBINS * x_idx + y_idx;
            if (x_idx != y_idx) {
                mat[mat_idx_inv] += 1;
            }

            // Logging
            //if (threadIdx.x == 50 && blockIdx.x == 20) {
            //    printf("y:%d x:%d idx:%d val:%d\\n", y_idx, x_idx, mat_idx, mat[mat_idx]);
            //    printf("y:%d x:%d inv_idx:%d inv_val:%d\\n", y_idx, x_idx, mat_idx_inv, mat[mat_idx_inv]);
            //}

        }
    }
    __device__ float glcm_stat_contrast_gpu(int* glcm_mat) {
        // calculate statistic on glcm matrix of size NBINS*NBINS
        float accum = 0;
        for (int i=0; i<GLCM_SIZE; i++) {
            int y = i / NBINS;
            int x = i - y*NBINS;
            accum += glcm_mat[i] * pow(y-x, 2);
            //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
            //}
        }
        return accum;
    }
    __device__ float glcm_stat_dissimilarity_gpu(int* glcm_mat) {
        // calculate statistic on glcm matrix of size NBINS*NBINS
        float accum = 0;
        for (int i=0; i<GLCM_SIZE; i++) {
            int y = i / NBINS;
            int x = i - y*NBINS;
            accum += glcm_mat[i] * abs(y-x);
            //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
            //}
        }
        return accum;
    }
    __device__ float glcm_stat_energy_gpu(int* glcm_mat) {
        // calculate statistic on glcm matrix of size NBINS*NBINS
        float accum = 0;
        for (int i=0; i<GLCM_SIZE; i++) {
            accum += pow(glcm_mat[i], 2);
            //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
            //}
        }
        return accum;
    }
    __device__ float glcm_stat_homogeneity_gpu(int* glcm_mat) {
        // calculate statistic on glcm matrix of size NBINS*NBINS
        float accum = 0;
        for (int i=0; i<GLCM_SIZE; i++) {
            int y = i / NBINS;
            int x = i - y*NBINS;
            accum += glcm_mat[i] / (1 + pow(y-x, 2));
            //if (threadIdx.x == 10 && blockIdx.x == 2) {
                //printf("y:%d x:%d accum:%f\\n", y, x, accum);
            //}
        }
        return accum;
    }
    __device__ float glcm_plugin_gpu(float *patch_array) {
        // quantize patch
        quantize_gpu(patch_array);

        // compute glcm matrix
        int mat[GLCM_SIZE];
        for (int i=0; i<GLCM_SIZE; i++) {
            mat[i] = 0;
        }
        glcm_matrix_gpu(mat, patch_array);

        // compute matrix statistic
        float stat = ${STAT}(mat);

        return stat;
    }

    __global__ void image_iterator_gpu(float *image_vect, float *result_vect) {
        // array index for this thread
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < IMAGE_SIZE) {
            float patch_array[PATCH_SIZE];
            int i = 0;
            for (int k_z = -Z_RADIUS; k_z <= Z_RADIUS; k_z++) {
                for (int k_y = -RADIUS; k_y <= RADIUS; k_y++) {
                    for (int k_x = -RADIUS; k_x <= RADIUS; k_x++) {
                        int k_idx = (int)(fminf(IMAGE_SIZE-1, fmaxf(0, (IMAGE_HEIGHT*IMAGE_WIDTH*k_z) + (IMAGE_WIDTH*k_y) + (idx + k_x))));

                        // Count unique pixel intensities
                        patch_array[i] = image_vect[k_idx];
                        //if (idx == 500) {
                        //    printf("i:%d - k_idx:%d - patchval:%f - imageval:%f\\n", i, k_idx, patch_array[i], image_vect[idx]);
                        //}
                        i += 1;
                    }
                }
            }
            //result_vect[idx] = image_vect[idx];
            result_vect[idx] = ${KERNEL}(patch_array);
        }
    }
    """)

    if isinstance(image_volume, np.ndarray):
        toBaseVolume = False
        logger.debug('recognized as an np.ndarray')
        if image_volume.ndim == 3:
            d, r, c = image_volume.shape
        elif image_volume.ndim == 2:
            d, r, c = (1, *image_volume.shape)
        image = image_volume.flatten()
    else:
        toBaseVolume = True
        logger.debug('recognized as a BaseVolume')
        image = image_volume.conformTo(roi.frameofreference).vectorize(roi)
        d, r, c = roi.frameofreference.size[::-1]

    logger.debug('d:{:d}, r:{:d}, c:{:d}'.format(d, r, c))
    if d == 1:
        z_radius = 0
    elif d > 1:
        z_radius = radius

    cuda_source = cuda_template.substitute({'RADIUS': radius,
                                            'Z_RADIUS': z_radius,
                                            'IMAGE_DEPTH': d,
                                            'IMAGE_HEIGHT': r,
                                            'IMAGE_WIDTH': c,
                                            'NBINS': gray_levels,
                                            'DX': dx,
                                            'DY': dy,
                                            'DZ': dz,
                                            'NDEV': ndev,
                                            'KERNEL': feature_kernel,
                                            'STAT': stat_name})
    mod2 = SourceModule(cuda_source)
    func = mod2.get_function('image_iterator_gpu')

    # allocate image on device in global memory
    image = image.astype(np.float32)
    image_gpu = cuda.mem_alloc(image.nbytes)
    result = np.zeros_like(image)
    result_gpu = cuda.mem_alloc(result.nbytes)
    # transfer image to device
    cuda.memcpy_htod(image_gpu, image)
    cuda.memcpy_htod(result_gpu, result)
    # call device kernel
    blocksize = 256
    gridsize = math.ceil(r*c*d/blocksize)
    func(image_gpu, result_gpu, block=(blocksize, 1,1), grid=(gridsize, 1,1))
    # get result from device
    cuda.memcpy_dtoh(result, result_gpu)

    logger.debug('feature result shape: {!s}'.format(result.shape))
    logger.debug('GPU done')
    # convert to int representation
    if d == 1:
        result = result.reshape(r, c)
    elif d>1:
        result = result.reshape(d,r,c)

    if toBaseVolume:
        return MaskableVolume().fromArray(result, roi.frameofreference)
    else:
        return result
