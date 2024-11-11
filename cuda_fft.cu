#include <stdio.h>
#include <stdlib.h>

#include "fft.h"

__device__ int bit_reverse(unsigned int n, int size) {
    int log_n = ceil(log2l(size));
    int limit = ceil(((double)log_n / 2));

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

__device__ void create_bit_reverse_lookup(int n, unsigned int* lookup) {
    for (int i = 0; i < n; i++) {
        lookup[i] = bit_reverse(i, n);
    }
}

__device__ void create_twiddle_factors_lookup(int n, complex_t twiddle_factors) {
    int log_n = ceil(log2l(n));
    for (int i = 0; i < log_n; i++) {
        int m = 1 << i;
        twiddle_factors[i] = (complex_t) {cos(-TWO_PI/m), sin(-TWO_PI/m)};
    }
}

__global__ void copy_to_output(double* arr, complex_t* output, unsigned int* rev_i_lookup) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int rev_i = rev_i_lookup[index];
    output[rev_i].real = arr[index];
}

__global__ void partial_butterfly_op(complex_t* input, complex_t* output, int n, int m2, int step_size, complex_t* twiddle_factors) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int local_index = index % n;

    int pos_or_neg = 0;
    if (local_index >= n / 2) {
        pos_or_neg = -1;
    }
    else {
        pos_or_neg = 1;
    }

    // need to change these
    int i1 = index + pos_or_neg*m2;
    int i2 = index;
    int twi = step_size * local_index;

    complex_t w = twiddle_factors[twi];

    double t_r = w.real*input[i1].real - w.imag*input[i1].imag;
    double t_i = w.real*input[i1].imag + w.imag*input[i1].real;
    complex_t t = {t_r, t_i};
    complex_t u = output[i2];
    
    output[index].real = u.real + pos_or_neg*t.real; 
    output[index].imag = u.imag + pos_or_neg*t.imag;
}

int main()
{
    int size = 8;
    double arr[] = {1, 6, 3, 8, 9, 5, 4, 2};

    int bytes =  sizeof(unsigned int)*size;
    unsigned int* rev_bit_lookup = (unsigned int*) malloc(bytes);
    create_bit_reverse_lookup(size, rev_bit_lookup);
    
    unsigned int* device_rev_bit_lookup = 0;
    cudaMalloc((void**)&device_rev_bit_lookup, bytes);
    cudaMemcpy(rev_bit_lookup, device_rev_bit_lookup, bytes, cudaMemcpyHostToDevice);

    int bytes = sizeof(complex_t)*ceil(log2l(size));
    complex_t* twiddle_factors = (complex_t*) malloc(bytes);
    create_twiddle_factors_lookup(size, twiddle_factors);

    complex_t* device_twiddle_factors = 0;
    cudaMalloc((void**)&device_twiddle_factors, bytes);
    cudaMemcpy(twiddle_factors, device_twiddle_factors, bytes, cudaMemcpyHostToDevice);

    int bytes =  sizeof(unsigned int)*size;
    double* device_arr = 0;
    cudaMalloc((void**)&device_arr, bytes);
    cudaMemcpy(arr, device_arr, bytes, cudaMemcpyHostToDevice);

    int bytes =  sizeof(complex_t)*size;
    complex_t* input =  (complex_t*) malloc(bytes);
    complex_t* device_input = 0;
    cudaMalloc((void**)&device_input, bytes);

    int bytes =  sizeof(complex_t)*size;
    complex_t* output =  (complex_t*) malloc(bytes);
    complex_t* device_output = 0;
    cudaMalloc((void**)&device_output, bytes);

    copy_to_output<<<1,size>>>(device_arr, device_input, device_rev_i_lookup);
    cudaDeviceSynchronize();

    cudaMemcpy(twiddle_factors, device_twiddle_factors, bytes, cudaMemcpyHostToDevice);



    return 0;
}