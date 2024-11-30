#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#include "fft.h"
#include "cuda_fft.hpp"
#include "util.h"

#define MAX_THREADS_PER_BLOCK 512

void create_twiddle_factors_lookup(int n, complex_t* twiddle_factors) {
    complex_t w = {1,0};
    complex_t w_n = {cos(-TWO_PI/n), sin(-TWO_PI/n)};
    for (uint64_t i = 0; i < n; i++) {
        twiddle_factors[i] = w;
        float w_r = w.real*w_n.real - w.imag*w_n.imag;
        float w_i = w.real*w_n.imag + w.imag*w_n.real;

        w.real = w_r; w.imag = w_i;
    }
}

struct fft_instance* alloc_fft_instance(uint64_t size) {
    struct fft_instance* output = (struct fft_instance*) malloc(sizeof(fft_instance));

    uint64_t bytes = sizeof(complex_t)*size;

    // Allocating Host Side Twiddle Factors
    complex_t* twiddle_factors = (complex_t*) malloc(bytes);
    create_twiddle_factors_lookup(size, twiddle_factors);

    // Allocating Device Side Twiddle Factors
    complex_t* device_twiddle_factors = 0;
    cudaMalloc((void**)&device_twiddle_factors, bytes);
    cudaMemcpy(device_twiddle_factors, twiddle_factors, bytes, cudaMemcpyHostToDevice);

    complex_t* buf1;
    complex_t* buf2;
    cudaMalloc((void**)&buf1, bytes);
    cudaMalloc((void**)&buf2, bytes);

    output->size = size;
    output->twiddle_factors = twiddle_factors;
    output->device_twiddle_factors = device_twiddle_factors;
    output->buf1 = buf1;
    output->buf2 = buf2;

    return output;
}

void free_fft_instance(struct fft_instance* fft) {
    free(fft->twiddle_factors);
    cudaFree(fft->device_twiddle_factors);
    cudaFree(fft->buf1);
    cudaFree(fft->buf2);
    free(fft);
}

__device__ int bit_reverse_device(unsigned int n, int size) {
    int log_n = ceil(log2f(size));
    int limit = ceil(((float)log_n / 2));

    unsigned int output = 0;
    for (int i = 0; i < limit; i++) {
        unsigned int bit_mask1 = 1 << (log_n - i - 1);
        unsigned int bit1_loc = n << (log_n - 2*i -1);
        unsigned int bit1 = bit1_loc & bit_mask1;

        unsigned int bit_mask2 = 1 << i; 
        unsigned int bit2_loc = n >> (log_n - 2*i -1);
        unsigned int bit2 = bit2_loc & bit_mask2;

        output |= bit1 | bit2;
    }
    return output;
}

__global__ void copy_to_output(float* arr, complex_t* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rev_i = bit_reverse_device(index,n);
    output[rev_i].real = arr[index];
    output[rev_i].imag = 0;
}

__global__ void partial_butterfly_op(complex_t* input, complex_t* output, int n, int m2, int step_size, int module_mask, complex_t* twiddle_factors) 
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t local_index = index % n;

    int pos_or_neg = 1;

    if (local_index >= n/2) {
        pos_or_neg = -1;
    }

    uint64_t reach_index = index + pos_or_neg*m2;
    
    uint64_t i1, i2;
    if (local_index >= n/2) {
        i1 = index;
        i2 = reach_index;
    }
    else {
        i1 = reach_index;
        i2 = index;
    }

    uint64_t twi = (step_size * local_index) & module_mask;

    complex_t w = twiddle_factors[twi];

    float t_r = w.real*input[i1].real - w.imag*input[i1].imag;
    float t_i = w.real*input[i1].imag + w.imag*input[i1].real;
    complex_t t = {t_r, t_i};
    complex_t u = input[i2];
    
    output[index].real = u.real + pos_or_neg*t.real; 
    output[index].imag = u.imag + pos_or_neg*t.imag;

    //printf("index=%d; local_index=%d; pos_or_neg=%d; i1=%d; i2=%d; twi=%d; w=%f+%fi;\n\tinput=%f+%fi; t=%f+%fi; u=%f+%fi; output=%f+%fi\n\n", index, local_index, pos_or_neg, i1, i2, 
    //       twi, w.real, w.imag, input[i1].real, input[i1].imag, t.real, t.imag, u.real, u.imag, output[i2].real, output[i2].imag);
    
}

void cuda_fft(float* arr, complex_t* output, struct fft_instance* fft) {
    uint64_t size = fft->size;
    complex_t* device_twiddle_factors = fft->device_twiddle_factors;
    complex_t* dev_input = fft->buf1;
    complex_t* dev_output = fft->buf2;

    uint64_t blocks;
    uint64_t threads;
    if (size > MAX_THREADS_PER_BLOCK) {
        blocks = size / MAX_THREADS_PER_BLOCK;
        threads = MAX_THREADS_PER_BLOCK;
    }
    else {
        blocks = 1;
        threads = size;
    }

    uint64_t log_n = ceil(log2f(size));

    // Allocate memory and perform bit reversing on device
    uint64_t bytes =  sizeof(float)*size;
    float* device_arr;
    cudaMalloc((void**)&device_arr, bytes);
    cudaMemcpy(device_arr, arr, bytes, cudaMemcpyHostToDevice);
    copy_to_output<<<blocks,threads>>>(device_arr, dev_input, size);
    cudaDeviceSynchronize();
    cudaFree(device_arr);

    uint64_t step_size = size;
    for (int i = 0; i < log_n; i++) {
        uint64_t m = 1 << (i+1);
        uint64_t m2 = m >> 1;
        step_size = step_size >> 1;
        partial_butterfly_op<<<blocks,threads>>>(dev_input, dev_output, m, m2, step_size, size/2 - 1, device_twiddle_factors);
        cudaDeviceSynchronize();

        complex_t* tmp = dev_output;
        dev_output = dev_input;
        dev_input = tmp;
    }

    bytes = sizeof(complex_t)*size;
    cudaMemcpy(output, dev_input, bytes, cudaMemcpyDeviceToHost);
}