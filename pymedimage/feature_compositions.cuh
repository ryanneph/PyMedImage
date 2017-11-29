extern "C" {
    __global__ void elementwiseMean(float *image_array,
                                    float *mean_array,
                                    int   array_length, 
                                    int   num_arrays) 
    {
        unsigned long int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx<(unsigned long int)array_length) {
            float accum = 0.0f;
            for (int k=0; k<num_arrays; k++) {
                long int true_idx = array_length * k + out_idx;
                accum += image_array[true_idx];
            }
            mean_array[out_idx] = accum / num_arrays;
        }
    }
     
    __global__ void elementwiseMax(float *image_array,
                                   float *max_array,
                                   int   array_length,
                                   int   num_arrays)
    {
        unsigned long int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx<(unsigned long int)array_length) {
            float max = 0.0f;
            for (int k=0; k<num_arrays; k++) {
                long int true_idx = array_length * k + out_idx;
                if (image_array[true_idx] > max) {
                    max = image_array[true_idx];
                }
            }
            max_array[out_idx] = max;
        }
        
    }
    __global__ void elementwiseMin(float *image_array,
                                   float *min_array,
                                   int   array_length,
                                   int   num_arrays)
    {
        unsigned long int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (out_idx<(unsigned long int)array_length) {
            float min = 9999.f;
            for (int k=0; k<num_arrays; k++) {
                long int true_idx = array_length * k + out_idx;
                if (image_array[true_idx] < min) {
                    min = image_array[true_idx];
                }
            }
            min_array[out_idx] = min;
        }
        
    }
}

