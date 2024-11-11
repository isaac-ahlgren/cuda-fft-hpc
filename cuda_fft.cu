#include <stdio.h>

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

__device__ int* create_bit_reverse_lookup(int n, unsigned int* lookup) {
    for (int i = 0; i < n; i++) {
        lookup[i] = bit_reverse(i, n);
    }
}

__global__ void copy_to_output(double* arr, complex_t* output, unsigned int* rev_i_lookup, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < n; i++) {
        unsigned int rev_i = rev_i_lookup[index];
        output[rev_i].real = arr[index];
    }
}

__global__ void fft_layer(double* arr, complex_t* output, int n, int depth, int m2, complex_t w_m) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}