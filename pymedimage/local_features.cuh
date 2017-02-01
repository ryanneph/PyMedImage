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


/* FORWARD DECLARATIONS */
__device__ float mean_plugin_gpu(float *array);
__device__ float stddev_plugin_gpu(float *array, int dof=1);

/* UTILITIES */
namespace rn {
    __device__ float mean(float *array, unsigned int size) {
        // computes the mean of the array
        int total = 0;
        for (int i=0; i<size; i++) {
            total += array[i];
        }
        return ((float)total / size);
    }
    __device__ float variance(float* array, unsigned int size, unsigned int dof=1) {
        // get the mean
        float mu = rn::mean(array, size);

        // compute the std. deviation
        float stat = 0.0f;
        for (int i=0; i<size; i++) {
            stat += powf(array[i] - mu, 2);
        }
        return (stat / (size-dof));
    }
    __device__ float stddev(float* array, unsigned int size, unsigned int dof=1) {
        return sqrtf(rn::variance(array, size, dof));
    }
}
__device__ void quantize_fixed_gpu(float *array, unsigned int HU) {
    //WIPWIPWIPWIPWIPWIPWIPWIP

    // quantize the array into bins of width 25HU

    // gaussian stats
    float f_mean = mean_plugin_gpu(array);
    float f_std = stddev_plugin_gpu(array, 1);
    float f_binwidth = 2*NDEV*f_std / (NBINS-2);
    //printf("mean:%f std:%f width:%f\\n", f_mean, f_std, f_binwidth);

    // rebin values
    for (int i=0; i<PATCH_SIZE; i++) {
        array[i] = fminf(NBINS-1, fmaxf(0, floorf(((array[i] - f_mean + NDEV*f_std)/(f_binwidth+1e-9)) + 1)));
    }
}
__device__ void quantize_gpu(float *array) {
    // quantize the array into nbins, placing lower 2.5% and upper 2.5% residuals of gaussian
    // distribution into first and last bins respectively

    // gaussian stats
    float f_mean = mean_plugin_gpu(array);
    float f_std = stddev_plugin_gpu(array, 1);
    float f_binwidth = 2*NDEV*f_std / (NBINS-2);
    //printf("mean:%f std:%f width:%f\\n", f_mean, f_std, f_binwidth);

    // rebin values
    for (int i=0; i<PATCH_SIZE; i++) {
        array[i] = fminf(NBINS-1, fmaxf(0, floorf(((array[i] - f_mean + NDEV*f_std)/(f_binwidth+1e-9)) + 1)));
    }
}

/* FIRST ORDER STATISTICS */
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
    float stat = 0.0f;
    for (int i=0; i<NBINS; i++) {
        stat -= ((float)hist[i]/size) * (logf(((float)hist[i]/size) + 1e-9));
    }
    return stat;
}
__device__ float uniformity_plugin_gpu(float *patch_array) {
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

    // calculate local uniformity
    float stat = 0.0;
    for (int i=0; i<NBINS; i++) {
        stat += powf((float)hist[i]/size, 2);
    }

    return sqrtf(stat);
}
__device__ float energy_plugin_gpu(float *patch_array) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    // calculate local energy
    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf((float)patch_array[i]/size, 2);
    }
    return stat;
}
__device__ float mean_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += patch_array[i];
    }
    return (stat/size);
}
__device__ float min_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = patch_array[0];
    for (int i=1; i<size; i++) {
        stat = fminf(stat, patch_array[i]) ;
    }
    return stat;
}
__device__ float max_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = patch_array[0];
    for (int i=1; i<size; i++) {
        stat = fmaxf(stat, patch_array[i]) ;
    }
    return stat;
}
__device__ float kurtosis_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(patch_array);
    float stddev = stddev_plugin_gpu(patch_array, 0);

    // compute the kurtosis
    float num = 0.0f;
    for (int i=0; i<size; i++) {
        num += powf(patch_array[i] - mu, 4);
    }
    return ((1.0f/size)*num) / (powf(stddev, 2));
}
__device__ float skewness_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(patch_array);
    float stddev = stddev_plugin_gpu(patch_array, 0);

    // compute the kurtosis
    float num = 0.0f;
    for (int i=0; i<size; i++) {
        num += powf( patch_array[i] - mu, 3);
    }
    return ((1.0f/size)*num) / (powf(stddev, 3));
}
__device__ float rms_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf(patch_array[i], 2);
    }
    return sqrtf(stat/size);
}
__device__ float variance_plugin_gpu(float *array, int dof=1) {
    // PATCH_SIZE macro in division doesn't work so we introduce local stack var to overcome
    int size = PATCH_SIZE;

    // get the mean
    float mu = mean_plugin_gpu(array);

    // compute the std. deviation
    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += powf( array[i] - mu, 2);
    }
    return (stat / (size-dof));
}
__device__ float stddev_plugin_gpu(float *array, int dof) {
    return sqrtf(variance_plugin_gpu(array, dof));
}
__device__ float range_plugin_gpu(float *array) {
    return (max_plugin_gpu(array) - min_plugin_gpu(array));
}
__device__ float meanabsdev_plugin_gpu(float *patch_array) {
    int size = PATCH_SIZE;
    float mu = mean_plugin_gpu(patch_array);

    float stat = 0.0f;
    for (int i=0; i<size; i++) {
        stat += fabsf(patch_array[i] - mu);
    }
    return sqrtf(stat/size);
}

/* GLCM FEATURES */
namespace rn {
    inline __device__ unsigned int glcm_getY(unsigned int idx) {
        return (idx / NBINS);
    }
    inline __device__ unsigned int glcm_getX(unsigned int idx) {
        return (idx - glcm_getY(idx)*NBINS);
    }
    inline __device__ unsigned int glcm_getIndex(unsigned int x, unsigned int y) {
        return (NBINS * y + x);
    }
    inline __device__ float glcm_marginal_Px(float* glcm_mat, unsigned int y) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            accum += glcm_mat[ glcm_getIndex(i, y) ];
        }
        return accum;
    }
    inline __device__ float glcm_marginal_Py(float* glcm_mat, unsigned int x) {
        float accum = 0.0f;
        for (int j=0; j<NBINS; j++) {
            accum += glcm_mat[ glcm_getIndex(x, j) ];
        }
        return accum;
    }
    inline __device__ float glcm_meanP(float* glcm_mat) {
        return mean(glcm_mat, GLCM_SIZE);
    }
    inline __device__ float glcm_mean_Px(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            accum += (glcm_marginal_Px(glcm_mat, i) * i);
        }
        return accum;
    }
    inline __device__ float glcm_mean_Py(float* glcm_mat) {
        float accum = 0.0f;
        for (int j=0; j<NBINS; j++) {
            accum += (glcm_marginal_Py(glcm_mat, j) * j);
        }
        return accum;
    }
    inline __device__ float glcm_variance_Px(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                accum += (powf(i - glcm_mean_Px(glcm_mat), 2) * glcm_mat[ glcm_getIndex(i, j) ]);
            }
        }
        return accum;
    }
    inline __device__ float glcm_variance_Py(float* glcm_mat) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                accum += (powf(j - glcm_mean_Py(glcm_mat), 2) * glcm_mat[ glcm_getIndex(i, j) ]);
            }
        }
        return accum;
    }
    inline __device__ float glcm_marginal_Pxplusy(float* glcm_mat, unsigned int k) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                if (i + j == k) {
                    accum += glcm_mat[ glcm_getIndex(i, j) ];
                }
            }
        }
        return accum;
    }
    inline __device__ float glcm_marginal_Pxminusy(float* glcm_mat, unsigned int k) {
        float accum = 0.0f;
        for (int i=0; i<NBINS; i++) {
            for (int j=0; j<NBINS; j++) {
                if (fabsf(i - j) == k) {
                    accum += glcm_mat[ glcm_getIndex(i, j) ];
                }
            }
        }
        return accum;
    }
}
__device__ void glcm_matrix_gpu(float *mat, float *array) {
    //OPTIONS
    bool accumulate_inverse = false;

    // compute glcm matrix in the specified directions
    int dx = DX;
    int dy = DY;
    int dz = DZ;

    int patch_size_1d = (RADIUS*2+1);
    int patch_size_2d = pow(patch_size_1d, 2);
    for (int i=0; i<PATCH_SIZE; i++) {
        int z = i / patch_size_2d;
        int y = (i - patch_size_2d*z) / patch_size_1d;
        int x = i - patch_size_2d*z - patch_size_1d*y;
        int y_idx = array[i];
        int x_idx_query = (int)fmaxf(0,fminf(PATCH_SIZE-1, (z+dz)*patch_size_2d + (y+dy)*patch_size_1d + (x + dx))); // should consider boundary handling options
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
        mat[mat_idx] += 1.0f;

        if (accumulate_inverse) {
            // fill lower triangular - opposite direction
            int mat_idx_inv = NBINS * x_idx + y_idx;
            mat[mat_idx_inv] += 1.0f;
        }

        // Logging
        //if (threadIdx.x == 50 && blockIdx.x == 20) {
        //    printf("y:%d x:%d idx:%d val:%d\\n", y_idx, x_idx, mat_idx, mat[mat_idx]);
        //    printf("y:%d x:%d inv_idx:%d inv_val:%d\\n", y_idx, x_idx, mat_idx_inv, mat[mat_idx_inv]);
        //}
    }

    /* normalize probabilities */
    unsigned int ncounts = 0;
    if (!accumulate_inverse)
        ncounts = PATCH_SIZE;
    else
        ncounts = 2 * PATCH_SIZE;
    for (int i=0; i<GLCM_SIZE; i++) {
        mat[i] /= ncounts;
    }
}


__device__ float glcm_stat_contrast_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] * pow(y-x, 2);
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_dissimilarity_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] * fabsf(y-x);
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
        //    printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_energy_gpu(float* glcm_mat) {
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
__device__ float glcm_stat_homogeneity_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + pow(y-x, 2));
        //if (threadIdx.x == 10 && blockIdx.x == 2) {
            //printf("y:%d x:%d accum:%f\\n", y, x, accum);
        //}
    }
    return accum;
}
__device__ float glcm_stat_homogeneity1_gpu(float* glcm_mat) {
    // calculate statistic on glcm matrix of size NBINS*NBINS
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += glcm_mat[i] / (1 + fabsf(y-x));
    }
    return accum;
}
__device__ float glcm_stat_autocorrelation_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += x*y*glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clusterprominence_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 4) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clustershade_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 3) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_clustertendency_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += powf(x + y - rn::glcm_mean_Px(glcm_mat) - rn::glcm_mean_Py(glcm_mat), 2) * glcm_mat[i];
    }
    return accum;
}
__device__ float glcm_stat_correlation_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        int y = rn::glcm_getY(i);
        int x = rn::glcm_getX(i);
        accum += ((x * y * glcm_mat[i]) - (rn::glcm_mean_Px(glcm_mat)*rn::glcm_mean_Py(glcm_mat))) /
                  (sqrtf(rn::glcm_variance_Px(glcm_mat))*sqrtf(rn::glcm_variance_Py(glcm_mat)));
    }
    return accum;
}
__device__ float glcm_stat_sumentropy_gpu(float* glcm_mat) {
    float accum = 0;
    for (int k=0; k<=NBINS*2-2; k++) {
        float val = rn::glcm_marginal_Pxplusy(glcm_mat, k);
        accum -= val * logf(val+1e-12);
    }
    return accum;
}
__device__ float glcm_stat_differenceentropy_gpu(float* glcm_mat) {
    float accum = 0;
    for (int k=0; k<NBINS; k++) {
        float val = rn::glcm_marginal_Pxminusy(glcm_mat, k);
        accum -= val * logf(val+1e-12);
    }
    return accum;
}
__device__ float glcm_stat_entropy_gpu(float* glcm_mat) {
    float accum = 0;
    for (int i=0; i<GLCM_SIZE; i++) {
        float val = glcm_mat[i];
        accum -= val * log2f(val+1e-12);
    }
    return accum;
}






__device__ float glcm_plugin_gpu(float *patch_array) {
    // quantize patch
    quantize_gpu(patch_array);

    // compute glcm matrix
    float mat[GLCM_SIZE];
    for (int i=0; i<GLCM_SIZE; i++) {
        mat[i] = 0.0f;
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
